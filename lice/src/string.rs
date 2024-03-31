//! TODO: what is up with latin-1 vs UTF-8 encoding???
use crate::{
    combinator::Combinator,
    integer::Integer,
    memory::{make_pair, App, Pointer, Value},
};
use alloc::ffi::CString;
use core::ffi::CStr;
use gc_arena::{Collect, Gc, Mutation};
use libc::c_void;

/// A vector-backed, GC-managed string.
///
/// TODO: this memory layout is really quite inefficient. Consider making this a `Gc<str>`,
/// a `Gc<[u8]>`, or a `Gc`-backed `ThinStr`.
#[derive(Debug, Clone, Copy, Collect, PartialEq, Eq)]
#[collect(no_drop)]
pub struct VString<'gc>(Gc<'gc, String>);

impl<'gc> VString<'gc> {
    pub fn new(m: &Mutation<'gc>, s: &str) -> Self {
        Self(Gc::new(m, s.to_string()))
    }
}

impl<'gc> AsRef<str> for VString<'gc> {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl<'gc> AsRef<[u8]> for VString<'gc> {
    fn as_ref(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

impl<'gc> core::fmt::Display for VString<'gc> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

pub(crate) fn hstring_from_utf8<'gc>(mc: &Mutation<'gc>, s: &str) -> Pointer<'gc> {
    // TODO: don't allocate new combinators
    let mut res = Pointer::new(mc, Value::Combinator(Combinator::NIL));
    if s.is_empty() {
        return res;
    }
    let cons = Pointer::new(mc, Value::Combinator(Combinator::CONS));

    for c in s.chars().rev() {
        let data = Pointer::new(mc, Value::Integer(Integer::from(c)));
        let data = Pointer::new(
            mc,
            Value::App(App {
                fun: cons,
                arg: data,
            }),
        );
        res = Pointer::new(
            mc,
            Value::App(App {
                fun: data,
                arg: res,
            }),
        );
    }
    res
}

// O x y f g = g x y
// K     f g = f
//
// Cons = O
// Nil  = K
//
// C'B f g x y = (f x) (g y)
// Seq !x y    = y
//
// NewCAStringLen s  = sseq s
//   where sseq      = P nil_case cons_case
//         nil_case  = cont
//         cons_case = C'B Seq sseq
//
// ~> P (Build s) (C'B Seq reducer) s
// ~> s (Build s) (C'B Seq reducer)
// case s of
//   Cons c s' -> O  c s' (Build s) (C'B Seq sseq)
//     ~>                            C'B Seq sseq c s'
//     ~>                            (Seq c) (sseq s')
//     ~>                            sseq s'
//    Nil      -> K       (Build s) (C'B Seq sseq)
//     ~>                  Build s
pub(crate) fn eval_hstring<'gc>(
    mc: &Mutation<'gc>,
    string: Pointer<'gc>,
    cont: Pointer<'gc>,
) -> Value<'gc> {
    let p = Pointer::new(mc, Value::Combinator(Combinator::P));
    let ccb = Pointer::new(mc, Value::Combinator(Combinator::CCB));
    let seq = Pointer::new(mc, Value::Combinator(Combinator::Seq));

    let sseq = Pointer::new(mc, /* placeholder */ Value::Integer(Integer::from(42)));
    let nil_case = cont;
    let cons_case = Pointer::new(
        mc,
        Value::App(App {
            fun: Pointer::new(mc, Value::App(App { fun: ccb, arg: seq })),
            arg: sseq,
        }),
    );

    sseq.set(
        mc,
        Value::App(App {
            fun: Pointer::new(
                mc,
                Value::App(App {
                    fun: p,
                    arg: nil_case,
                }),
            ),
            arg: cons_case,
        }),
    );

    // Instead of:
    //
    //      Value::App(App{fun: reducer, arg: s})
    //
    // Optimization: we can pre-reduce P:
    //
    //      P nil_case cons_case s ~> s nil_case cons_case
    Value::App(App {
        fun: Pointer::new(
            mc,
            Value::App(App {
                fun: string,
                arg: nil_case,
            }),
        ),
        arg: cons_case,
    })
}

pub(crate) fn read_hstring(s: Pointer<'_>) -> Option<Vec<u8>> {
    let mut bytes = Vec::<u8>::new();
    let mut cur = s.follow();
    while !matches!(cur.unpack(), Value::Combinator(Combinator::NIL)) {
        let Value::App(App {
            fun: cons_c,
            arg: next,
        }) = cur.unpack()
        else {
            return None;
        };
        let Value::App(App { fun: cons, arg: c }) = cons_c.follow().unpack() else {
            return None;
        };
        let Value::Combinator(Combinator::CONS) = cons.follow().unpack() else {
            return None;
        };
        let c =
            char::try_from(c.follow().unpack()).expect("Haskell String should be a list of Chars");
        let mut buf = [0; 4];
        bytes.extend(c.encode_utf8(&mut buf).as_bytes());
        cur = next.follow();
    }
    Some(bytes)
}

pub(crate) fn new_castringlen<'gc>(mc: &Mutation<'gc>, s: Pointer<'gc>) -> Option<Pointer<'gc>> {
    let bytes = read_hstring(s)?;
    let cstring = CString::new(bytes).unwrap();
    let cstr = cstring.as_bytes_with_nul();

    // Haskell mandates that (libc's) `free` be called on the pointer returned by `NewCAStringLen`,
    // so we need to store the string contents in a `malloc`-allocated buffer.
    let buf_ptr = unsafe { libc::malloc(cstr.len()) };
    unsafe { libc::memcpy(buf_ptr, cstr.as_ptr() as *mut c_void, cstr.len()) };
    Some(make_pair(
        mc,
        (
            Pointer::new(mc, buf_ptr.into()),
            Pointer::new(mc, cstring.as_bytes().len().into()),
        ),
    ))
}

pub(crate) fn peek_castring<'gc>(mc: &Mutation<'gc>, s: Pointer<'gc>) -> Pointer<'gc> {
    let ptr = s.unpack().unwrap_foreign_ptr().into();
    let cstr = unsafe { CStr::from_ptr(ptr) };
    hstring_from_utf8(mc, cstr.to_str().unwrap())
}

pub(crate) fn peek_castringlen<'gc>(
    mc: &Mutation<'gc>,
    s: Pointer<'gc>,
    l: Pointer<'gc>,
) -> Pointer<'gc> {
    let ptr = s.unpack().unwrap_foreign_ptr().into();
    let cstr = unsafe { CStr::from_ptr(ptr) };
    let len = l.unpack().unwrap_integer().signed().max(0) as usize;
    hstring_from_utf8(mc, &cstr.to_str().unwrap()[0..len])
}

//! TODO: what is up with latin-1 vs UTF-8 encoding???
use crate::{
    combinator::Combinator,
    integer::Integer,
    memory::{make_pair, App, IntoValue, Pointer, Value},
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

impl<'gc> IntoValue<'gc> for &str {
    fn into_value(self, mc: &Mutation<'gc>) -> Value<'gc> {
        hstring_from_utf8(mc, self)
    }
}

impl<'gc> IntoValue<'gc> for CString {
    fn into_value(self, mc: &Mutation<'gc>) -> Value<'gc> {
        hstring_from_utf8(mc, &self.to_string_lossy())
    }
}

impl<'gc> IntoValue<'gc> for &CStr {
    fn into_value(self, mc: &Mutation<'gc>) -> Value<'gc> {
        hstring_from_utf8(mc, &self.to_string_lossy())
    }
}

pub(crate) fn hstring_from_utf8<'gc>(mc: &Mutation<'gc>, s: &str) -> Value<'gc> {
    // TODO: don't allocate new combinators
    let mut res = Combinator::NIL.into_value(mc);
    if s.is_empty() {
        return res;
    }
    let cons = Pointer::new(mc, Combinator::CONS);
    for c in s.chars().rev() {
        res = Value::App(App {
            fun: Pointer::new(
                mc,
                App {
                    fun: cons,
                    arg: Pointer::new(mc, c),
                },
            ),
            arg: Pointer::new(mc, res),
        });
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
        let Value::Integer(c) = c.follow().unpack() else {
            return None;
        };
        let c = char::from(c); // TODO: complain if not UTF-8
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
            Pointer::new(mc, buf_ptr),
            Pointer::new(mc, cstring.as_bytes().len()),
        ),
    ))
}

pub(crate) fn peek_castring<'gc>(ptr: *const core::ffi::c_char) -> &'gc str {
    &unsafe { CStr::from_ptr(ptr) }.to_str().unwrap()
}

pub(crate) fn peek_castringlen<'gc>(ptr: *const core::ffi::c_char, len: usize) -> &'gc str {
    let cstr = unsafe { CStr::from_ptr(ptr) };
    &cstr.to_str().unwrap()[0..len]
}

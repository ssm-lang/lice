#![allow(non_upper_case_globals)] // Appease rust-analyzer for derive(FromPrimitive)

use crate::combinator::Combinator;
use core::{isize, marker::PhantomData, mem};
use gc_arena::{Collect, Gc};
use thin_str::ThinStr;

const _: () = assert!(
    mem::size_of::<Word>() == mem::size_of::<usize>(),
    "Word should be the size of a pointer"
);
const _: () = assert!(
    mem::size_of::<&'static PackedValue>() == mem::size_of::<usize>(),
    "&'a PackedValue should be exactly one word"
);
const _: () = assert!(
    mem::size_of::<Box<PackedValue>>() == mem::size_of::<Word>(),
    "Box<PackedValue> should be exactly one word, i.e., it is just a pointer"
);
const _: () = assert!(
    mem::size_of::<Gc<PackedValue>>() == mem::size_of::<Word>(),
    "Gc<PackedValue> should be exactly one word, i.e., it is just a pointer"
);
const _: () = assert!(
    mem::size_of::<PackedValue>() == mem::size_of::<Word>() * 2,
    "PackedValue is the size of two words (pointers)"
);

type TryFromResult<T> = Result<T, num_enum::TryFromPrimitiveError<T>>;

// Typedef for integer literals
type Integer = isize;

// TODO: only if target supports floats
#[cfg(target_pointer_width = "32")]
type Float = f32;

#[cfg(target_pointer_width = "64")]
type Float = f64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
struct Word(usize);

impl Word {
    #[inline(always)]
    fn from_pointer(pointer: Pointer) -> Self {
        // SAFETY: taking the pointer value from a reference
        let ptr: usize = unsafe { mem::transmute(pointer.ptr) };

        debug_assert!(
            ptr & Pointer::STOLEN_MASK == 0,
            "lower bits of *const PackedValue should be zero"
        );
        Self(ptr | usize::from(pointer.kind))
    }

    #[inline(always)]
    fn stolen(&self) -> usize {
        self.0 & Pointer::STOLEN_MASK
    }

    /// SAFETY: this method is only safe if this `Word` is correctly encoded, i.e., the upper bits
    /// must point to a valid memory location if the lower bits are set to `Pointer::GC` or
    /// `Pointer::STATIC`.
    #[inline(always)]
    unsafe fn to_pointer<'gc>(self) -> Option<Pointer<'gc>> {
        let bits = self.stolen();

        if bits == Pointer::NONPTR {
            return None;
        }

        let kind = PointerKind::try_from(bits).unwrap();

        // SAFETY: it doesn't matter whether this is static, boxed, or GC'ed, at the end of the day
        // it's just a pointer, so we can recover it by masking off whatever bits were stolen.
        let ptr = (self.0 & !Pointer::STOLEN_MASK) as *const PackedValue;
        Some(Pointer { ptr: &*ptr, kind })
    }

    #[inline(always)]
    fn from_tag(tag: Tag) -> Self {
        Self((u8::from(tag) as usize) << 8)
    }

    #[inline(always)]
    fn to_tag(self) -> TryFromResult<Tag> {
        Tag::try_from((self.0 >> 8) as u8)
    }

    #[inline(always)]
    fn from_combinator(comb: Combinator) -> Self {
        let tag = u8::from(Tag::Combinator) as usize;
        let comb = u8::from(comb) as usize;
        Self(comb << 16 | tag << 8)
    }

    #[inline(always)]
    fn to_combinator(self) -> TryFromResult<Combinator> {
        Combinator::try_from((self.0 >> 16) as u8)
    }

    #[inline(always)]
    unsafe fn to_str<'gc>(self) -> &'gc str {
        // NOTE: here we transmute `&self.0` to a `&ThinStr`, rather than `self.0` to `ThinStr`,
        // beacuse doing the latter will cause the string to be dropped when this function returns.
        let s: &ThinStr = mem::transmute(&self.0);
        s.as_str()
    }

    #[inline(always)]
    fn from_integer(i: Integer) -> Self {
        // SAFETY: any bit pattern is a valid `Integer`.
        //
        // We use `transmute` rather than an `as` cast to ensure compilation failure in case
        // `Integer` and `self.0` are of a different size.
        Self(unsafe { mem::transmute(i) })
    }

    #[inline(always)]
    fn to_integer(self) -> Integer {
        // SAFETY: any bit pattern is a valid `Integer`.
        //
        // We use `transmute` rather than an `as` cast to ensure compilation failure in case
        // `Integer` and `self.0` are of a different size.
        unsafe { mem::transmute(self.0) }
    }

    #[inline(always)]
    fn from_float(f: Float) -> Self {
        // SAFETY: any bit pattern is some well-defined `Float` (might be `NaN` but we don't care).
        //
        // We use `transmute` rather than an `as` cast to ensure compilation failure in case
        // `Float` and `self.0` are somehow of a different size.
        Self(unsafe { mem::transmute(Float::to_bits(f)) })
    }

    #[inline(always)]
    fn to_float(self) -> Float {
        // SAFETY: any bit pattern is some well-defined `Float` (might be `NaN` but we don't care).
        //
        // We use `transmute` rather than an `as` cast to ensure compilation failure in case
        // `Float` and `self.0` are somehow of a different size.
        Float::from_bits(unsafe { mem::transmute(self.0) })
    }

    /// SAFETY: this word must be a possible bit-stolen pointer, i.e., it cannot be an integer.
    #[inline(always)]
    unsafe fn trace(&self, cc: &gc_arena::Collection) {
        if self.stolen() == Pointer::GC {
            let gc: &Gc<PackedValue> = unsafe { mem::transmute(self.0) };
            gc.trace(cc);
        }
    }

    #[inline(always)]
    fn arbitrary() -> Self {
        Self(0xdeadbeef)
    }
}

#[repr(align(4))]
pub struct PackedValue<'gc>(Word, Word, PhantomData<&'gc ()>);

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    num_enum::IntoPrimitive,
    num_enum::TryFromPrimitive,
)]
#[repr(u8)]
enum Tag {
    Ref = 0 << Pointer::STOLEN_BITS | Pointer::NONPTR as u8,
    Combinator = 1 << Pointer::STOLEN_BITS | Pointer::NONPTR as u8,
    String = 2 << Pointer::STOLEN_BITS | Pointer::NONPTR as u8,
    Integer = 3 << Pointer::STOLEN_BITS | Pointer::NONPTR as u8,
    Float = 4 << Pointer::STOLEN_BITS | Pointer::NONPTR as u8,
}

pub enum Value<'gc> {
    App {
        fun: Pointer<'gc>,
        arg: Pointer<'gc>,
    },
    Ref(Pointer<'gc>),
    Combinator(Combinator),
    String(&'gc str),
    Integer(Integer),
    Float(Float),
}

pub struct Pointer<'gc> {
    ptr: &'gc PackedValue<'gc>,
    kind: PointerKind,
}

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    num_enum::IntoPrimitive,
    num_enum::TryFromPrimitive,
)]
#[repr(usize)]
enum PointerKind {
    Gc = Pointer::GC,
    Box = Pointer::BOX,
    Static = Pointer::STATIC,
}

impl<'gc> Pointer<'gc> {
    const STOLEN_BITS: usize = 2;
    const STOLEN_MASK: usize = 0b11;
    const GC: usize = 0b00;
    const BOX: usize = 0b01;
    const STATIC: usize = 0b10;
    const NONPTR: usize = 0b11;

    #[inline(always)]
    pub fn from_gc(gc: Gc<'gc, PackedValue<'gc>>) -> Self {
        let ptr = unsafe { mem::transmute(gc) };
        Self {
            ptr,
            kind: PointerKind::Gc,
        }
    }

    #[inline(always)]
    pub fn from_box(bx: Box<PackedValue<'gc>>) -> Self {
        let ptr = unsafe { mem::transmute(bx) };
        Self {
            ptr,
            kind: PointerKind::Box,
        }
    }

    #[inline(always)]
    pub fn from_static(bx: &'static PackedValue<'gc>) -> Self {
        let ptr = unsafe { mem::transmute(bx) };
        Self {
            ptr,
            kind: PointerKind::Static,
        }
    }
}

impl<'gc> AsRef<PackedValue<'gc>> for Pointer<'gc> {
    #[inline(always)]
    fn as_ref(&self) -> &PackedValue<'gc> {
        self.ptr
    }
}

impl<'gc> Value<'gc> {
    #[inline(always)]
    pub fn pack(self) -> PackedValue<'gc> {
        match self {
            Value::App { fun, arg } => PackedValue(
                Word::from_pointer(fun),
                Word::from_pointer(arg),
                PhantomData,
            ),
            Value::Ref(rf) => PackedValue(
                Word::from_tag(Tag::Ref),
                Word::from_pointer(rf),
                PhantomData,
            ),
            Value::Combinator(comb) => {
                PackedValue(Word::from_combinator(comb), Word::arbitrary(), PhantomData)
            }
            Value::String(_s) => {
                todo!("I want to prevent user from creating Value::String(), but I don't want to allocate here")
            }
            Value::Integer(i) => PackedValue(
                Word::from_tag(Tag::Integer),
                Word::from_integer(i),
                PhantomData,
            ),
            Value::Float(f) => {
                PackedValue(Word::from_tag(Tag::Float), Word::from_float(f), PhantomData)
            }
        }
    }
}

impl<'gc> PackedValue<'gc> {
    #[inline(always)]
    pub fn unpack(&'gc self) -> Value<'gc> {
        if let Some(fun) = unsafe { self.0.to_pointer() } {
            let arg = unsafe { self.1.to_pointer() }
                .expect("packed application node should have two pointers");
            return Value::App { fun, arg };
        }

        let tag = self
            .0
            .to_tag()
            .expect("non-application node should have valid tag");

        match tag {
            Tag::Ref => {
                let ptr =
                    unsafe { self.1.to_pointer() }.expect("Tag::Ref should have a pointer payload");
                Value::Ref(ptr)
            }
            Tag::Combinator => {
                let comb = self
                    .0
                    .to_combinator()
                    .expect("Tag::Combinator should have valid combinator");
                Value::Combinator(comb)
            }
            Tag::String => Value::String(unsafe { self.1.to_str() }),
            Tag::Integer => Value::Integer(self.1.to_integer()),
            Tag::Float => Value::Float(self.1.to_float()),
        }
    }
}

impl<'gc> Drop for PackedValue<'gc> {
    fn drop(&mut self) {
        // NOTE: we don't need to drop GC or STATIC pointers
        if self.0.stolen() == Pointer::NONPTR {
            match self.0.to_tag().unwrap() {
                Tag::String => {
                    let s: ThinStr = unsafe { mem::transmute(self.0) };
                    mem::drop(s)
                }
                Tag::Ref if self.1.stolen() == Pointer::BOX => {
                    let bx: Box<PackedValue<'gc>> = unsafe { mem::transmute(self.1) };
                    mem::drop(bx)
                }
                _ => (),
            }
        } else {
            // This is an App node
        }
    }
}

unsafe impl<'gc> Collect for PackedValue<'gc> {
    fn trace(&self, cc: &gc_arena::Collection) {
        if self.0.stolen() == Pointer::NONPTR {
            if self.0.to_tag().unwrap() == Tag::Ref {
                unsafe { self.1.trace(cc) };
            }
        } else {
            unsafe { self.0.trace(cc) };
            unsafe { self.1.trace(cc) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "rt")]
    #[test]
    fn non_app_tag_lsb() {
        macro_rules! check_lsb {
            ($tag:ident) => {
                assert_eq!(
                    Tag::$tag as u8 & Pointer::STOLEN_MASK as u8,
                    Pointer::NONPTR as u8,
                    "{:#?}",
                    Tag::$tag
                )
            };
        }
        check_lsb!(Ref);
        check_lsb!(Combinator);
        check_lsb!(String);
        check_lsb!(Integer);
        check_lsb!(Float);
    }
}

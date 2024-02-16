//! Runtime memory layout for combinator graph reducer.

#![allow(non_upper_case_globals)] // Appease rust-analyzer for derive(FromPrimitive)

use crate::combinator::Combinator;
use core::{isize, marker::PhantomData, mem};
use gc_arena::{Collect, Gc};
use thin_str::ThinStr;

/// Typedef for integer values
pub type Integer = isize;

/// Typedef for floating point values
/// TODO: only if target supports floats
#[cfg(target_pointer_width = "32")]
pub type Float = f32;

/// Typedef for floating point values
#[cfg(target_pointer_width = "64")]
pub type Float = f64;

/// Compile-time assertions about size of various kinds of pointers.
mod assertions {
    use super::*;
    const _: () = assert!(
        mem::size_of::<usize>() >= 4,
        "Word should be at least 4 bytes (32-bits)"
    );
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
}

/// A node in the combinator graph.
///
/// The memory layout is based on the 2-word cells in Lennart's runtime:
///
/// ```text
///                                    <--- MSB ----   ---- LSB --->
///                                    Byte3   Byte2   Byte1   Byte0
///  +=======+=======+=======+=======+=======+=======+=======+=======+
///  | these bits are only present   |       | Comb  | Tag   | Ptr   | Word0
///  +-------+-------+-------+-------+-------+-------+-------+-------+
///  | and used on 64-bit platforms  |       |       |       |       | Word1
///  +=======+=======+=======+=======+=======+=======+=======+=======+
/// ```
///
/// When the least significant bits of the `Ptr` field (i.e., `Word0,Byte0`) are NOT set to
/// [`Pointer::NONPTR`] , then this memory cell is to be interpreted as a function application node
/// ("App node", denoted `@`), where `Word0` encodes a pointer to the function node being applied,
/// and `Word1` encodes a pointer to the argument node.
///
/// The word "encodes" is used here for these pointers because this implementation supports three
/// different kinds of pointers: garbage-collected [`Gc`] pointers, unique [`Box`] pointers, and
/// `&'static` references; those are indicated by those least significant bits set to
/// [`Pointer::GC`], [`Pointer::BOX`], and [`Pointer::STATIC`], respectively. These types all have
/// the exact same memory layout (i.e., a single raw pointer), so we can [`mem::transmute()`] the
/// word to the pointer type, _after_ masking off the two lowest bits used as the pointer tag.
///
/// When the least significant bits of the `Ptr` field are set to [`Pointer::NONPTR`]
/// (i.e., `0b11`), then the type (and thus memory layout) of this cell is to be found in the
/// second byte of the first word, i.e., `Word0,Byte1`, which should contain a valid `Tag` value.
/// That `Tag` determines how the rest of the cell should be interpreted.
///
/// In most cases, the rest of the bytes of `Word0` are unused, while `Word1` is interpreted either
/// some kind of value (e.g., [`Integer`] or [`Float`]) or pointer (e.g., [`ThinStr`]). However,
/// when the `Tag` is set to `Tag::Combinator`, the combinator tag value is stored in
/// `Word0,Byte2`, with `Word1` set to some arbitrary value. (TODO: consider packing a pointer into
/// `Word1`?).
///
/// The [`PhantomData`] embedded in the `PackedValue` holds the lifetime of garbage-collected
/// objects, and does not take any space in this node.
///
/// Since this cell is a low-level representation, its word fields are not exposed directly.
/// Instead, this type provides an [`Self::unpack()`] method that expands into a [`Value`] that
/// that represents a type-safe "view" of what is stored in the cell. If the Rust optimizer does
/// its job, that [`Value`] never exist at runtime, and be completely inlined away.
///
/// NOTE: For clarity, consider implementing this as a newtype'd `union`? (Naked `union` doesn't
/// work because we need to implement [`Drop`]).
#[repr(align(4))]
pub struct PackedValue<'gc>(Word, Word, PhantomData<&'gc ()>);

/// A word, the same size as a pointer.
///
/// We newtype the underlying `usize` to get rid of the `Copy` and `Clone` implementations.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
struct Word(usize);

/// What kind of data is stored in a [`PackedValue`].
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
    /// An indirection node
    Ref,
    /// A combinator constant
    Combinator,
    /// A string value
    String,
    /// An integer value
    Integer,
    /// A floating point value
    Float,
}

impl<'gc> PackedValue<'gc> {
    /// Construct an application node cell.
    pub fn new_app(fun: Pointer<'gc>, arg: Pointer<'gc>) -> Self {
        Self(
            Word::from_pointer(fun),
            Word::from_pointer(arg),
            PhantomData,
        )
    }

    /// Construct an indirection node cell.
    pub fn new_ref(ptr: Pointer<'gc>) -> Self {
        Self(
            Word::from_tag(Tag::Ref),
            Word::from_pointer(ptr),
            PhantomData,
        )
    }

    /// Construct a combinator constant node cell.
    pub fn new_combinator(comb: Combinator) -> Self {
        Self(Word::from_combinator(comb), Word::arbitrary(), PhantomData)
    }

    /// Construct a string value node cell.
    pub fn new_str(s: &str) -> Self {
        let s = ThinStr::new(s);
        Self(Word::from_tag(Tag::String), Word::from_str(s), PhantomData)
    }

    /// Construct an integer value node cell.
    pub fn new_integer(i: Integer) -> Self {
        Self(
            Word::from_tag(Tag::Integer),
            Word::from_integer(i),
            PhantomData,
        )
    }

    /// Construct a floating point value node cell.
    pub fn new_float(f: Float) -> Self {
        Self(Word::from_tag(Tag::Float), Word::from_float(f), PhantomData)
    }

    /// Construct a type-safe "view" of what is stored in this cell.
    ///
    /// If the Rust optimizer does its job, the returned [`Value`] should be inlined away and never
    /// need to exist at runtime.
    #[inline(always)]
    pub fn unpack(&'gc self) -> Value<'gc> {
        if let Some(fun) = unsafe { self.0.to_pointer() } {
            let arg = unsafe { self.1.to_pointer() }
                .expect("packed application node should have two pointers");
            return Value::App { fun, arg };
        }

        let tag = self
            .0
            .as_tag()
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
                    .as_combinator()
                    .expect("Tag::Combinator should have valid combinator");
                Value::Combinator(comb)
            }
            Tag::String => Value::String(unsafe { self.1.as_str() }),
            Tag::Integer => Value::Integer(self.1.as_integer()),
            Tag::Float => Value::Float(self.1.as_float()),
        }
    }
}

/// Cells may contain [`Gc`] pointers, which need to be traced by the collector.
///
/// [`Box`] pointers also need to be traced through.
unsafe impl<'gc> Collect for PackedValue<'gc> {
    fn trace(&self, cc: &gc_arena::Collection) {
        if self.0.stolen() == Pointer::NONPTR {
            if self.0.as_tag().unwrap() == Tag::Ref {
                unsafe { self.1.trace_pointer(cc) };
            }
        } else {
            unsafe { self.0.trace_pointer(cc) };
            unsafe { self.1.trace_pointer(cc) };
        }
    }
}

/// Cells may own heap-allocated values, which need to be freed when the cell is dropped.
impl<'gc> Drop for PackedValue<'gc> {
    fn drop(&mut self) {
        if self.0.stolen() == Pointer::NONPTR {
            match self.0.as_tag().unwrap() {
                Tag::String => unsafe { self.1.drop_str() },
                Tag::Ref => unsafe { self.1.drop_pointer() },
                _ => (),
            }
        } else {
            // This is an App node, which contains two pointers
            unsafe {
                self.0.drop_pointer();
                self.1.drop_pointer();
            }
        }
    }
}

impl Word {
    /// Bit-offset of the tag byte
    const TAG_OFFSET: usize = 8;
    /// Bit offset of the combinator byte
    const COMB_OFFSET: usize = 16;

    /// Stolen bits of this word (assuming it is a packed pointer).
    #[inline(always)]
    fn stolen(&self) -> usize {
        self.0 & Pointer::STOLEN_MASK
    }

    /// Construct a word from a pointer.
    #[inline(always)]
    #[must_use = "not using this Word may leak a pointer"]
    fn from_pointer(pointer: Pointer) -> Self {
        // SAFETY: taking the pointer value from a reference
        let ptr: usize = unsafe { mem::transmute(pointer.ptr) };

        debug_assert!(
            ptr & Pointer::STOLEN_MASK == 0,
            "lower bits of *const PackedValue should be zero"
        );
        Self(ptr | usize::from(pointer.kind))
    }

    /// Interpret this word as a possible packed pointer.
    ///
    /// SAFETY: this method is only safe if this `Word` is correctly encoded, i.e., the upper bits
    /// must point to a valid memory location if the lower bits are set to `Pointer::GC` or
    /// `Pointer::STATIC`.
    /// TODO: rephrsae in terms of what this word was created using
    #[inline(always)]
    unsafe fn to_pointer<'gc>(&self) -> Option<Pointer<'gc>> {
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

    /// Trace (through) this pointer during the GC mark phase.
    ///
    /// SAFETY: this word must be a possible bit-stolen pointer, i.e., it cannot be an integer.
    #[inline(always)]
    unsafe fn trace_pointer(&self, cc: &gc_arena::Collection) {
        if self.stolen() == Pointer::GC {
            let gc: &Gc<PackedValue> = unsafe { mem::transmute(self.0) };
            gc.trace(cc);
        } else if self.stolen() == Pointer::BOX {
            todo!("Need to trace through box pointers!")
        }
    }

    /// Drop this pointer, freeing any owned data.
    #[inline(always)]
    unsafe fn drop_pointer(&mut self) {
        if let Some(mut ptr) = self.to_pointer() {
            ptr.drop_box();
        }
    }

    /// Construct a word holding a tag, marked as non-pointer.
    #[inline(always)]
    fn from_tag(tag: Tag) -> Self {
        let tag = u8::from(tag) as usize;
        Self(tag << Self::TAG_OFFSET | Pointer::NONPTR)
    }

    /// Read from the tag byte of this word.
    #[inline(always)]
    fn as_tag(&self) -> TryFromResult<Tag> {
        Tag::try_from((self.0 >> Self::TAG_OFFSET) as u8)
    }

    /// Construct a word holding a combinator and a tag, marked as non-pointer.
    #[inline(always)]
    fn from_combinator(comb: Combinator) -> Self {
        let tag = u8::from(Tag::Combinator) as usize;
        let comb = u8::from(comb) as usize;
        Self(comb << Self::COMB_OFFSET | tag << Self::TAG_OFFSET | Pointer::NONPTR)
    }

    /// Read from the comb byte of this word.
    #[inline(always)]
    fn as_combinator(&self) -> TryFromResult<Combinator> {
        Combinator::try_from((self.0 >> Self::COMB_OFFSET) as u8)
    }

    /// Construct a word holding a pointer to a [`ThinStr`] value.
    #[inline(always)]
    #[must_use = "dropping this word will leak the given ThinStr"]
    fn from_str(s: ThinStr) -> Self {
        // NOTE: we consume `s: ThinStr` here and leak it
        let s = mem::ManuallyDrop::new(s);
        Self(unsafe { mem::transmute(s) })
    }

    /// Obtain a string reference from this word.
    ///
    /// SAFETY: this word must have been constructed using [`Self::from_str`].
    #[inline(always)]
    unsafe fn as_str(&self) -> &str {
        // NOTE: here we transmute `&self.0` rather than `self.0`, because the latter would consume
        // the word from self.
        let s: &mem::ManuallyDrop<ThinStr> = mem::transmute(&self.0);
        s.as_str()
    }

    /// Drop the owned string pointed to by this word.
    ///
    /// SAFETY: this word must have been constructed using [`Self::from_str`].
    #[inline(always)]
    unsafe fn drop_str(&mut self) {
        let s: mem::ManuallyDrop<ThinStr> = mem::transmute(self.0);
        let _: ThinStr = mem::ManuallyDrop::into_inner(s); // drops the ThinStr
    }

    /// Construct a word from an integer value.
    #[inline(always)]
    fn from_integer(i: Integer) -> Self {
        // SAFETY: any bit pattern is a valid `Integer`.
        //
        // We use `transmute` rather than an `as` cast to ensure compilation failure in case
        // `Integer` and `self.0` are of a different size.
        Self(unsafe { mem::transmute(i) })
    }

    /// Interpret this word as an integer value.
    #[inline(always)]
    fn as_integer(&self) -> Integer {
        // SAFETY: any bit pattern is a valid `Integer`.
        //
        // We use `transmute` rather than an `as` cast to ensure compilation failure in case
        // `Integer` and `self.0` are of a different size.
        unsafe { mem::transmute(self.0) }
    }

    /// Construct a word from an floating-point value.
    #[inline(always)]
    fn from_float(f: Float) -> Self {
        // SAFETY: any bit pattern is some well-defined `Float` (might be `NaN` but we don't care).
        //
        // We use `transmute` rather than an `as` cast to ensure compilation failure in case
        // `Float` and `self.0` are somehow of a different size.
        Self(unsafe { mem::transmute(Float::to_bits(f)) })
    }

    /// Interpret this word as a floating-point value.
    #[inline(always)]
    fn as_float(&self) -> Float {
        // SAFETY: any bit pattern is some well-defined `Float` (might be `NaN` but we don't care).
        //
        // We use `transmute` rather than an `as` cast to ensure compilation failure in case
        // `Float` and `self.0` are somehow of a different size.
        Float::from_bits(unsafe { mem::transmute(self.0) })
    }

    /// A word containing an arbitrary value.
    #[inline(always)]
    fn arbitrary() -> Self {
        Self(0xdeadbeef)
    }
}

/// A type-safe "view" of the contents of a [`PackedValue`].
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

/// A managed pointer type.
pub struct Pointer<'gc> {
    ptr: &'gc PackedValue<'gc>,
    kind: PointerKind,
}

/// Enumeration of the support kinds of pointers.
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
    /// Mask to obtain bits stolen from the pointer.
    const STOLEN_MASK: usize = 0b11;
    /// Lower bits of a stolen [`Gc`] pointer.
    pub const GC: usize = 0b00;
    /// Lower bits of a stolen [`Box`] pointer.
    pub const BOX: usize = 0b01;
    /// Lower bits of a stolen `&'static` pointer.
    pub const STATIC: usize = 0b10;
    /// Lower bits of a non-pointer.
    pub const NONPTR: usize = 0b11;

    #[inline(always)]
    pub fn from_gc(gc: Gc<'gc, PackedValue<'gc>>) -> Self {
        let ptr = gc.as_ref();
        Self {
            ptr,
            kind: PointerKind::Gc,
        }
    }

    #[inline(always)]
    pub fn from_box(bx: Box<PackedValue<'gc>>) -> Self {
        let ptr = Box::leak(bx); // Should be the same as a `mem::transmute(bx)`
        Self {
            ptr,
            kind: PointerKind::Box,
        }
    }

    #[inline(always)]
    pub fn from_static(ptr: &'static PackedValue<'gc>) -> Self {
        Self {
            ptr, // No conversion needed: 'static outlives 'gc
            kind: PointerKind::Static,
        }
    }

    #[inline(always)]
    pub fn drop_box(&mut self) {
        if matches!(self.kind, PointerKind::Box) {
            // Inverse of Box::leak(), recovers a Box<T> which is dropped here.
            let _: Box<PackedValue<'gc>> =
                unsafe { Box::from_raw(self.ptr as *const PackedValue as *mut PackedValue) };
        }
    }
}

impl<'gc> AsRef<PackedValue<'gc>> for Pointer<'gc> {
    #[inline(always)]
    fn as_ref(&self) -> &PackedValue<'gc> {
        self.ptr
    }
}

type TryFromResult<T> = Result<T, num_enum::TryFromPrimitiveError<T>>;

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

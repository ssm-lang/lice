//! Runtime memory layout for combinator graph reducer.
//!
//! [`Node`]s in the combinator graph are always at least 4-byte aligned, so we use the two least
//! significant bits of potential pointers to [`Node`]s to determine whether it is a valid pointer,
//! and what kind of pointer it is. Those bit patterns are enumerated by the [`Stolen`] type.
//!
//! The memory layout for [`Node`]s are based on the 2-word memory cells from Lennart's runtime:
//!
//! ```text
//!                                    <--- MSB ----   ---- LSB --->
//!                                    Byte3   Byte2   Byte1   Byte0
//!  +=======+=======+=======+=======+=======+=======+=======+=======+
//!  | these bits are only present   |       | Comb  | Tag   | Ptr   | Word0
//!  +-------+-------+-------+-------+-------+-------+-------+-------+
//!  | and used on 64-bit platforms  |       |       |       |       | Word1
//!  +=======+=======+=======+=======+=======+=======+=======+=======+
//! ```
//!
//! When the least significant bits of a [`Node`]'s `Ptr` field (i.e., `Word0,Byte0`) are NOT set
//! to [`Stolen::NonPtr`], then this node is a function application node ("App node", denoted `@`),
//! where `Word0` encodes a pointer to the function node being applied, and `Word1` encodes
//! a pointer to the argument node.
//!
//! When the least significant bits of a [`Node`]'s `Ptr` field are set to [`Stolen::NonPtr`]
//! (i.e., `0b11`), then the type [`Tag`] (and thus memory layout) of this node is to be found in
//! the second byte of the first word, i.e., `Word0,Byte1`.
//!
//! In most cases, the rest of the bytes of `Word0` are unused, while `Word1` is interpreted either
//! some kind of value (e.g., [`Integer`] or [`Float`]) or pointer (e.g., [`ThinStr`]). However,
//! when the `Tag` is set to `Tag::Combinator`, the combinator tag value is stored in
//! `Word0,Byte2`, with `Word1` set to some arbitrary value.
//!
#![allow(non_upper_case_globals)] // Appease rust-analyzer for derive(FromPrimitive)

use crate::combinator::Combinator;
use core::{
    isize,
    mem::{self, ManuallyDrop},
    ops,
    ptr::NonNull,
};
use gc_arena::{Collect, Gc};
use num_enum::UnsafeFromPrimitive;
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

/// Compile-time checks for size and alignment of low-level data structures.
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
        mem::size_of::<Node>() == mem::size_of::<Word>() * 2,
        "PackedValue is the size of two words (pointers)"
    );
    const _: () = assert!(
        mem::align_of::<Node>() >= 4,
        "PackedValue should have alignment of at least 4 (to allow bit-stealing)"
    );

    const _: () = assert!(
        mem::size_of::<&'static Node>() == mem::size_of::<usize>(),
        "&'a PackedValue should be exactly one word"
    );
    const _: () = assert!(
        mem::size_of::<Box<Node>>() == mem::size_of::<Word>(),
        "Box<PackedValue> should be exactly one word, i.e., it is just a pointer"
    );
    const _: () = assert!(
        mem::size_of::<Gc<Node>>() == mem::size_of::<Word>(),
        "Gc<PackedValue> should be exactly one word, i.e., it is just a pointer"
    );
}

/// A node in the combinator graph.
///
/// Since this type represents the low-level, in-memory representation for nodes, its fields are
/// not exposed directly. It has the same memory footprint and layout as two pointers:
///
/// ```
/// # use lice::memory::Node;
/// # use std::mem;
/// assert_eq!(mem::size_of::<Node>(), mem::size_of::<usize>() * 2);
/// ```
///
/// Instead, this type provides an [`Self::unpack()`] method that expands into a [`Value`] that
/// represents a type-safe "view" of what is stored in the node. If the Rust optimizer does its
/// job, that [`Value`] never exist at runtime, and be completely inlined away.
///
/// For more info, see [module documentation](self).
#[derive(Debug)]
#[repr(align(4))]
pub struct Node<'gc>(Word<'gc>, Word<'gc>);

impl<'gc> Node<'gc> {
    /// Construct an application node from pointers to two other nodes.
    pub fn new_app(fun: Pointer<'gc>, arg: Pointer<'gc>) -> Self {
        Self(Word::from(fun), Word::from(arg))
    }

    /// Construct an indirection node.
    pub fn new_ref(ptr: Pointer<'gc>) -> Self {
        Self(Word::from(Tag::Ref), Word::from(ptr))
    }

    /// Construct a combinator constant node.
    pub fn new_combinator(comb: Combinator) -> Self {
        Self(Word::from(comb), Word::arbitrary())
    }

    /// Construct a string value node.
    pub fn new_str(s: &str) -> Self {
        let s = ThinStr::new(s);
        Self(Word::from(Tag::String), Word::from(s))
    }

    /// Construct an integer value node.
    pub fn new_integer(i: Integer) -> Self {
        Self(Word::from(Tag::Integer), Word::from(i))
    }

    /// Construct a floating point value node.
    pub fn new_float(f: Float) -> Self {
        Self(Word::from(Tag::Float), Word::from(f))
    }

    /// Construct a type-safe "view" of the [`Value`] stored in this node.
    #[inline(always)]
    pub fn unpack(&'gc self) -> Value<'gc> {
        // SAFETY: the words in this node can only be constructed using its `new_*` constructors.
        unsafe {
            if let Some(fun) = self.0.as_ref() {
                let arg = self
                    .1
                    .as_ref()
                    .expect("packed application node should have two pointers");
                return Value::App { fun, arg };
            }

            let tag = self
                .0
                .tag()
                .expect("non-application node should have valid tag");

            match tag {
                Tag::Ref => {
                    let ptr = self
                        .1
                        .as_ref()
                        .expect("Tag::Ref should have a pointer payload");
                    Value::Ref(ptr)
                }
                Tag::Combinator => {
                    let comb = self
                        .0
                        .combinator()
                        .expect("Tag::Combinator should have valid combinator");
                    Value::Combinator(comb)
                }
                Tag::String => Value::String(self.1.as_str()),
                Tag::Integer => Value::Integer(self.1.int),
                Tag::Float => Value::Float(self.1.float),
            }
        }
    }
}

/// Cells may contain [`Gc`] pointers, which need to be traced by the collector.
///
/// [`Box`] pointers also need to be traced through.
unsafe impl<'gc> Collect for Node<'gc> {
    fn trace(&self, cc: &gc_arena::Collection) {
        // SAFETY: the words in this node can only be constructed using its `new_*` constructors.
        unsafe {
            if self.0.stolen() == Stolen::NonPtr {
                if self.0.tag().unwrap() == Tag::Ref {
                    self.1.trace_as_ref(cc);
                }
            } else {
                self.0.trace_as_ref(cc);
                self.1.trace_as_ref(cc);
            }
        }
    }
}

/// Cells may own heap-allocated values, which need to be freed when the cell is dropped.
impl<'gc> Drop for Node<'gc> {
    fn drop(&mut self) {
        // SAFETY: the words in this node can only be constructed using its `new_*` constructors.
        unsafe {
            if self.0.stolen() == Stolen::NonPtr {
                match self.0.tag().unwrap() {
                    Tag::String => self.1.drop_as_string(),
                    Tag::Ref => self.1.drop_as_ref(),
                    _ => (),
                }
            } else {
                // This is an App node, which contains two pointers
                self.0.drop_as_ref();
                self.1.drop_as_ref();
            }
        }
    }
}

/// A word, the same size as a pointer, but representing various kinds of data.
union Word<'gc> {
    word: usize,
    gc_ptr: Gc<'gc, Node<'gc>>,
    bx_ptr: ManuallyDrop<Box<Node<'gc>>>,
    st_ptr: &'static Node<'static>,
    string: ManuallyDrop<ThinStr>,
    int: Integer,
    float: Float,
}

impl<'gc> std::fmt::Debug for Word<'gc> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // SAFETY: self.word is always some valid bit-pattern.
        match mem::size_of::<usize>() {
            8 => f.write_fmt(format_args!("Word({:#066x})", unsafe { self.word })),
            4 => f.write_fmt(format_args!("Word({:#034x})", unsafe { self.word })),
            _ => unreachable!("pointers should only be 8 or 4 bytes"),
        }
    }
}

impl<'gc> From<Pointer<'gc>> for Word<'gc> {
    fn from(value: Pointer<'gc>) -> Self {
        // SAFETY: any `ptr` field in the given `Pointer` should always point to some valid memory.
        unsafe {
            match value {
                Pointer::Gc { ptr, .. } => {
                    let mut word = Word {
                        gc_ptr: Gc::from_ptr(ptr.as_ptr()),
                    };
                    Stolen::Gc.steal_in_place(&mut word.word);
                    word
                }
                Pointer::Box { ptr, .. } => {
                    let mut word = Word {
                        bx_ptr: ManuallyDrop::new(Box::from_raw(ptr.as_ptr())),
                    };
                    Stolen::Box.steal_in_place(&mut word.word);
                    word
                }
                Pointer::Static { ptr, .. } => {
                    let mut word = Word {
                        st_ptr: ptr.as_ref(),
                    };
                    Stolen::Static.steal_in_place(&mut word.word);
                    word
                }
            }
        }
    }
}

impl<'gc> From<Tag> for Word<'gc> {
    fn from(tag: Tag) -> Self {
        let tag = u8::from(tag) as usize;
        Self {
            word: Stolen::NonPtr.steal(tag << Self::TAG_OFFSET),
        }
    }
}

impl<'gc> From<Combinator> for Word<'gc> {
    fn from(comb: Combinator) -> Self {
        let tag = u8::from(Tag::Combinator) as usize;
        let comb = u8::from(comb) as usize;
        Self {
            word: Stolen::NonPtr.steal(comb << Self::COMB_OFFSET | tag << Self::TAG_OFFSET),
        }
    }
}

impl<'gc> From<ThinStr> for Word<'gc> {
    #[must_use = "dropping this word will leak the given ThinStr"]
    fn from(value: ThinStr) -> Self {
        Self {
            string: mem::ManuallyDrop::new(value),
        }
    }
}

impl<'gc> From<Integer> for Word<'gc> {
    fn from(int: Integer) -> Self {
        Self { int }
    }
}

impl<'gc> From<Float> for Word<'gc> {
    fn from(float: Float) -> Self {
        Self { float }
    }
}

impl<'gc> Word<'gc> {
    /// Bit-offset of the tag byte
    const TAG_OFFSET: usize = 8;
    /// Bit offset of the combinator byte
    const COMB_OFFSET: usize = 16;

    /// Stolen bits of this word (assuming it is a packed pointer).
    ///
    /// # Safety
    ///
    /// This method is only safe if and only if this [`Word`] was constructed using its
    /// `From<Pointer>`, `From<Tag>`, or `From<Combinator>` implementations.
    #[inline(always)]
    unsafe fn stolen(&self) -> Stolen {
        Stolen::from(self.word)
    }

    /// Interpret this word as a possible packed pointer.
    ///
    /// # Safety
    ///
    /// This method is only safe if and only if this [`Word`] was constructed using its
    /// `From<Pointer>`, `From<Tag>`, or `From<Combinator>` implementations.
    #[inline(always)]
    unsafe fn as_ref(&self) -> Option<Pointer<'gc>> {
        let ptr = Word {
            word: Stolen::unsteal(self.word),
        };
        Some(match self.stolen() {
            Stolen::NonPtr => return None,
            Stolen::Gc => Pointer::Gc {
                ptr: NonNull::from(&*ptr.gc_ptr),
            },
            Stolen::Box => Pointer::Box {
                ptr: NonNull::from(&*ptr.gc_ptr),
            },
            Stolen::Static => Pointer::Static {
                ptr: NonNull::from(ptr.st_ptr),
            },
        })
    }

    /// Trace possible garbage-collected pointers embedded in this word.
    ///
    /// # Safety
    ///
    /// This method is only safe if and only if this [`Word`] was constructed using its
    /// `From<Pointer>`, `From<Tag>`, or `From<Combinator>` implementations.
    #[inline(always)]
    unsafe fn trace_as_ref(&self, cc: &gc_arena::Collection) {
        match self.stolen() {
            Stolen::Gc => self.gc_ptr.trace(cc),
            Stolen::Box => (*self.bx_ptr).trace(cc),
            // NOTE: no need to trace STATIC pointers because they should not contain GC pointers
            Stolen::Static | Stolen::NonPtr => (),
        }
    }

    /// Perform drop operation for this word.
    ///
    /// # Safety
    ///
    /// This method is only safe if and only if this [`Word`] was constructed using its
    /// `From<Pointer>`, `From<Tag>`, or `From<Combinator>` implementations.
    #[inline(always)]
    unsafe fn drop_as_ref(&mut self) {
        if Stolen::Box == self.stolen() {
            mem::ManuallyDrop::drop(&mut self.bx_ptr)
        }
    }

    /// Read from the type tag of this word.
    ///
    /// # Safety
    ///
    /// This method is only safe if and only if this [`Word`] was constructed using its
    /// `From<Tag>`, or `From<Combinator>` implementations.
    #[inline(always)]
    unsafe fn tag(&self) -> TryFromResult<Tag> {
        Tag::try_from((self.word >> Self::TAG_OFFSET) as u8)
    }

    /// Read from the combinator type of this word.
    ///
    /// # Safety
    ///
    /// This method is only safe if and only if this [`Word`] was constructed using its
    /// `From<Combinator>` implementation.
    #[inline(always)]
    unsafe fn combinator(&self) -> TryFromResult<Combinator> {
        Combinator::try_from((self.word >> Self::COMB_OFFSET) as u8)
    }

    /// Obtain a string reference from this word.
    ///
    /// # Safety
    ///
    /// This method is only safe if and only if this [`Word`] was constructed using its
    /// `From<ThinStr>` implementation.
    #[inline(always)]
    unsafe fn as_str(&self) -> &str {
        self.string.as_str()
    }

    /// Drop the owned string pointed to by this word.
    ///
    /// # Safety
    ///
    /// This method is only safe if and only if this [`Word`] was constructed using its
    /// `From<ThinStr>` implementation.
    #[inline(always)]
    unsafe fn drop_as_string(&mut self) {
        mem::ManuallyDrop::drop(&mut self.string);
    }

    /// A word containing an arbitrary value.
    #[inline(always)]
    fn arbitrary() -> Self {
        Self { word: 0xdeadbebe }
    }
}

/// What kind of data is stored in a [`Node`].
///
/// This type is only made public for documentation purposes.
///
/// For more info, see [module documentation](self).
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
pub enum Tag {
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

/// A type-safe "view" of the contents of a [`Node`].
///
/// The memory layout of this type is not compact, and is not designed to be stored in memory.
/// Instead, it should be used as the temporary return value of [`Node::unpack()`]; if the Rust
/// inliner and optimizer does its job, no memory will be allocated for this type.
#[derive(Debug, PartialEq)]
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

/// A type-safe reference to a [`Node`].
///
/// Encapsulates one of several different types of internal pointers, and can be dererenced to
/// a [`Node`] reference. Users should not create [`Pointer`]s using this type's constructors;
/// instead, use its [`From`] implementations.
#[derive(Debug)]
pub enum Pointer<'gc> {
    Gc { ptr: NonNull<Node<'gc>> },
    Box { ptr: NonNull<Node<'gc>> },
    Static { ptr: NonNull<Node<'static>> },
}

/// Referential pointer equality.
impl<'gc> PartialEq for Pointer<'gc> {
    fn eq(&self, other: &Self) -> bool {
        match (&self, &other) {
            (Pointer::Gc { ptr: this }, Pointer::Gc { ptr: that })
            | (Pointer::Box { ptr: this }, Pointer::Box { ptr: that }) => this.eq(that),
            (Pointer::Static { ptr: ref this }, Pointer::Static { ptr: ref that }) => this.eq(that),
            _ => false,
        }
    }
}

impl<'gc> From<Gc<'gc, Node<'gc>>> for Pointer<'gc> {
    #[inline(always)]
    fn from(value: Gc<'gc, Node<'gc>>) -> Self {
        let ptr = NonNull::new(Gc::as_ptr(value) as *mut Node<'gc>)
            .expect("Gc pointer should be non-null");
        Pointer::Gc { ptr }
    }
}

impl<'gc> From<Box<Node<'gc>>> for Pointer<'gc> {
    #[inline(always)]
    fn from(value: Box<Node<'gc>>) -> Self {
        let ptr = NonNull::new(Box::into_raw(value)).expect("Box pointer should be non-null");
        Pointer::Box { ptr }
    }
}

impl<'gc> From<&'static Node<'static>> for Pointer<'gc> {
    #[inline(always)]
    fn from(value: &'static Node<'static>) -> Self {
        let ptr = NonNull::new(value as *const Node<'static> as *mut Node<'static>)
            .expect("Static pointer should be non-null");
        Pointer::Static { ptr }
    }
}

impl<'gc> ops::Deref for Pointer<'gc> {
    type Target = Node<'gc>;
    fn deref(&self) -> &Self::Target {
        // SAFETY: these are all always valid pointers, so `as_ref()` is fine.
        unsafe {
            match self {
                Pointer::Gc { ptr, .. } | Pointer::Box { ptr, .. } => ptr.as_ref(),
                Pointer::Static { ptr, .. } =>
                // SAFETY: invariance of 'gc comes from Pointer::Gc, which we can transmute away
                {
                    mem::transmute(ptr.as_ref())
                }
            }
        }
    }
}

/// Bits stolen from the two least significant bits of a pointer.
///
/// This type is only made public for documentation purposes.
///
/// Obeys the following identity:
///
/// ```
/// # use lice::memory::Stolen;
/// let word: usize;
/// # word = 0xdeadbe00;
/// let stolen_tag: Stolen;
/// # stolen_tag = Stolen::Static;
/// let stolen_word: usize = stolen_tag.steal(word);
///
/// # unsafe {
/// assert_eq!(word, Stolen::unsteal(stolen_word));
/// assert_eq!(stolen_tag, Stolen::from(stolen_word));
/// # }
/// ```
///
/// For more info, see [module documentation](self).
#[derive(Debug, UnsafeFromPrimitive, PartialEq, Eq, Clone, Copy)]
#[repr(u8)]
pub enum Stolen {
    /// Lower bits of a stolen [`Gc`] pointer.
    Gc = 0b00,
    /// Lower bits of a stolen [`Box`] pointer.
    Box = 0b01,
    /// Lower bits of a stolen `&'static` pointer.
    Static = 0b10,
    /// Lower bits of a non-pointer.
    NonPtr = 0b11,
}

impl Stolen {
    /// Mask to obtain bits stolen from a word.
    const MASK: usize = 0b11;

    /// Obtain stolen bits embedded in a word.
    ///
    /// # Safety
    ///
    /// This word must have been created using [`Self::steal`] or [`Self::steal_in_place`].
    pub unsafe fn from(word: usize) -> Self {
        let word = word & Self::MASK;
        let byte = word as u8;
        Self::unchecked_transmute_from(byte)
    }

    /// Obtain word without stolen bits.
    pub fn unsteal(word: usize) -> usize {
        word & !Self::MASK
    }

    /// Steal bits from a word.
    ///
    pub fn steal(&self, word: usize) -> usize {
        Self::unsteal(word) | *self as u8 as usize
    }

    /// Steal bits from a word by modifying it in place.
    pub fn steal_in_place(&self, word: &mut usize) {
        *word = self.steal(*word);
    }
}

/// Typedef for `enum` conversion results.
type TryFromResult<T> = Result<T, num_enum::TryFromPrimitiveError<T>>;

#[cfg(test)]
mod tests {
    use super::*;
    use gc_arena::{Arena, Rootable};

    #[test]
    fn make_combinator_cell() {
        // We need to create an arena here (despite no heap allocation) because of 'gc invariance.
        type Root = Rootable![Node<'_>];

        let arena: Arena<Root> = Arena::new(|_m| Node::new_combinator(Combinator::BB));
        arena.mutate(|_, comb| {
            assert_eq!(comb.unpack(), Value::Combinator(Combinator::BB));
        });
    }

    #[test]
    fn make_string_cell() {
        // We need to create an arena here (despite no GC allocation) because of 'gc invariance.
        type Root = Rootable![Node<'_>];

        // Create and initialize from a stack string to ensure we're comparing by value.
        let hello = "hello".to_string();

        let arena: Arena<Root> = Arena::new(|_m| Node::new_str(&hello));
        arena.mutate(|_, s| {
            assert_eq!(s.unpack(), Value::String("hello"));
        });
    }

    #[test]
    fn make_number_cell() {
        // We need to create an arena here (despite no GC allocation) because of 'gc invariance.
        type Root = Rootable![Node<'_>];

        let mut arena: Arena<Root> = Arena::new(|_m| Node::new_integer(42));
        arena.mutate(|_, v| assert_eq!(v.unpack(), Value::Integer(42)));

        arena.mutate_root(|_, v| {
            *v = Node::new_float(64.0);
        });
        arena.mutate(|_, v| assert_eq!(v.unpack(), Value::Float(64.0)));
    }

    #[test]
    fn make_app_cell() {
        type Root = Rootable![Vec<Gc<'_, Node<'_>>>];
        let mut arena: Arena<Root> = Arena::new(|_m| vec![]);

        arena.mutate_root(|m, v| {
            v.push(Gc::new(m, Node::new_combinator(Combinator::I)));
        });

        arena.mutate_root(|m, v| {
            v.push(Gc::new(m, Node::new_str("hello")));
        });

        arena.mutate_root(|m, v| {
            let arg = Pointer::from(v[1]);
            let fun = Pointer::from(v[0]);
            v.push(Gc::new(m, Node::new_app(fun, arg)));
        });

        arena.mutate(|_m, v| {
            let arg = Pointer::from(v[1]);
            let fun = Pointer::from(v[0]);
            assert_eq!(v[2].unpack(), Value::App { fun, arg });
        });

        arena.collect_all();

        arena.mutate(|_m, v| {
            let arg = Pointer::from(v[1]);
            let fun = Pointer::from(v[0]);
            assert_eq!(v[2].unpack(), Value::App { fun, arg });
        });

        arena.mutate_root(|_m, v| {
            let app = v.pop().unwrap();
            v.pop().unwrap();
            v.pop().unwrap();
            v.push(app);
        });

        arena.collect_all();

        arena.mutate(|_m, v| {
            let app = v[0].unpack();
            if !matches!(
                app,
                Value::App {
                    fun: _fun,
                    arg: _arg
                }
            ) {
                panic!("expected to unpack application, got: {:?}", v[0].unpack());
            }
        });

        arena.mutate_root(|_m, v| v.clear());

        arena.collect_all();

        assert_eq!(
            arena.metrics().total_allocation(),
            0,
            "arena should be empty after root is cleared"
        );
    }
}

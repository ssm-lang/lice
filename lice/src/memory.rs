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
use crate::string::GcString;
use core::{cell::Cell, isize, ops};
use gc_arena::{barrier::Unlock, lock::Lock, static_collect, Collect, Gc, Mutation};

/// Typedef for integer values
pub type Integer = isize;

/// Typedef for floating point values
/// TODO: only if target supports floats
#[cfg(target_pointer_width = "32")]
pub type Float = f32;

/// Typedef for floating point values
#[cfg(target_pointer_width = "64")]
pub type Float = f64;

/// A node in the combinator graph.
///
/// This type provides an [`Self::unpack()`] method that expands into a [`Value`] that represents
/// a type-safe "view" of what is stored in the node. If the Rust optimizer does its job, that
/// [`Value`] should never exist at runtime, and be completely inlined away.
///
/// For more info, see [module documentation](self).
#[derive(Debug, Collect)]
#[collect(no_drop)]
pub struct Node<'gc>(Lock<Value<'gc>>);

impl<'gc> Unlock for Node<'gc> {
    type Unlocked = Cell<Value<'gc>>;

    unsafe fn unlock_unchecked(&self) -> &Self::Unlocked {
        todo!()
    }
}

/// Inner contents of a node.
#[derive(Debug, Clone, Copy, Collect, PartialEq)]
#[collect(no_drop)]
pub enum Value<'gc> {
    App {
        fun: Pointer<'gc>,
        arg: Pointer<'gc>,
    },
    Ref(Pointer<'gc>),
    Combinator(Combinator),
    String(GcString<'gc>),
    Integer(Integer),
    Float(Float),
}

// Combinators are just constant values.
static_collect!(Combinator);

impl<'gc> Node<'gc> {
    /// View the inner contents.
    pub fn unpack(&self) -> Value<'gc> {
        self.0.get()
    }
}

impl<'gc> From<Value<'gc>> for Node<'gc> {
    fn from(value: Value<'gc>) -> Self {
        Self(Lock::new(value))
    }
}

/// A reference to a [`Node`].
///
/// Because `Pointer`s need to be packed into the [`Node`] union, their size must be the same size
/// as a raw pointer:
///
/// ```
/// # use lice::memory::Pointer;
/// # use std::mem;
/// assert_eq!(mem::size_of::<Pointer>(), mem::size_of::<usize>());
/// ```
///
/// Currently, this is just a wrapper around [`Gc`] pointers, but this may eventually encapsulate
/// several different types of internal pointers, e.g., unique [`Box`]es or `&'static` references.
/// Those pointers will have their lowest bits tagged by the values enumerated in [`Stolen`].
#[derive(Debug, Collect, Clone, Copy)]
#[collect(no_drop)]
pub struct Pointer<'gc> {
    ptr: Gc<'gc, Node<'gc>>,
}

impl <'gc> Pointer<'gc> {
    pub fn new(mc: &Mutation<'gc>, value: Value<'gc>) -> Self {
        Gc::new(mc, Node::from(value)).into()
    }

    pub fn set(&self, mc: &Mutation<'gc>, value: Value<'gc>) {
        self.ptr.unlock(mc).set(value)
    }
}

/// Referential pointer equality.
impl<'gc> PartialEq for Pointer<'gc> {
    fn eq(&self, other: &Self) -> bool {
        Gc::ptr_eq(self.ptr, other.ptr)
    }
}

/// Pointers can be dereferenced like we would expect of regular pointers.
impl<'gc> ops::Deref for Pointer<'gc> {
    type Target = Node<'gc>;
    fn deref(&self) -> &Self::Target {
        self.ptr.as_ref()
    }
}

impl<'gc> From<Gc<'gc, Node<'gc>>> for Pointer<'gc> {
    #[inline(always)]
    fn from(value: Gc<'gc, Node<'gc>>) -> Self {
        Self { ptr: value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gc_arena::{Arena, Rootable};

    #[test]
    fn make_combinator_cell() {
        // We need to create an arena here (despite no heap allocation) because of 'gc invariance.
        type Root = Rootable![Node<'_>];

        let arena: Arena<Root> = Arena::new(|_m| Node::from(Value::Combinator(Combinator::BB)));
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

        let arena: Arena<Root> =
            Arena::new(|m| Node::from(Value::String(GcString::new(m, &hello))));
        arena.mutate(|m, s| {
            assert_eq!(s.unpack(), Value::String(GcString::new(m, "hello")));
        });
    }

    #[test]
    fn make_number_cell() {
        // We need to create an arena here (despite no GC allocation) because of 'gc invariance.
        type Root = Rootable![Node<'_>];

        let mut arena: Arena<Root> = Arena::new(|_m| Node::from(Value::Integer(42)));
        arena.mutate(|_, v| assert_eq!(v.unpack(), Value::Integer(42)));

        arena.mutate_root(|_, v| {
            *v = Node::from(Value::Float(64.0));
        });
        arena.mutate(|_, v| assert_eq!(v.unpack(), Value::Float(64.0)));
    }

    #[test]
    fn make_app_cell() {
        type Root = Rootable![Vec<Gc<'_, Node<'_>>>];
        let mut arena: Arena<Root> = Arena::new(|_m| vec![]);

        arena.mutate_root(|m, v| {
            v.push(Gc::new(m, Node::from(Value::Combinator(Combinator::I))));
        });

        arena.mutate_root(|m, v| {
            v.push(Gc::new(
                m,
                Node::from(Value::String(GcString::new(m, "hello"))),
            ));
        });

        arena.mutate_root(|m, v| {
            let arg = Pointer::from(v[1]);
            let fun = Pointer::from(v[0]);
            v.push(Gc::new(m, Node::from(Value::App { fun, arg })));
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

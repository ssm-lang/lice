//! Runtime memory layout for combinator graph reducer.

use crate::ffi::{self, BadDyn, FfiSymbol, ForeignPtr};
use crate::float::Float;
use crate::string::VString;
use crate::tick::{Tick, TickTable};
use crate::{combinator::Combinator, integer::Integer};
use core::fmt::Debug;
use core::{cell::Cell, ops};
use gc_arena::{barrier::Unlock, lock::Lock, static_collect, Collect, Gc, Mutation};

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

const _: () = assert!(std::mem::size_of::<Node>() == std::mem::size_of::<Value>());

impl<'gc> Unlock for Node<'gc> {
    type Unlocked = Cell<Value<'gc>>;

    unsafe fn unlock_unchecked(&self) -> &Self::Unlocked {
        self.0.unlock_unchecked()
    }
}

/// Inner contents of a node.
#[derive(
    Debug,
    Clone,
    Copy,
    Collect,
    PartialEq,
    derive_more::From,
    derive_more::Unwrap,
    derive_more::TryUnwrap,
    derive_more::IsVariant,
    strum::IntoStaticStr,
)]
#[collect(no_drop)]
pub enum Value<'gc> {
    App(App<'gc>),
    Ref(Pointer<'gc>),
    Combinator(Combinator),
    String(VString<'gc>),
    Integer(Integer),
    Float(Float),
    BadDyn(BadDyn<'gc>),
    Ffi(FfiSymbol),
    Tick(Tick),
    ForeignPtr(ForeignPtr),
}

// Combinators are just constant values.
static_collect!(Combinator);
macro_rules! from_gc_value {
    ($variant:ident) => {
        fn from_value(
            _mc: &gc_arena::Mutation<'gc>,
            value: crate::memory::Value<'gc>,
        ) -> Result<Self, crate::memory::ConversionError> {
            if let crate::memory::Value::$variant(v) = value {
                Ok(v.into())
            } else {
                Err(crate::memory::ConversionError {
                    expected: core::any::type_name::<Self>(),
                    got: value.into(),
                })
            }
        }
    };
}

macro_rules! into_gc_value {
    ($variant:ident) => {
        fn into_value(self, _mc: &gc_arena::Mutation<'gc>) -> crate::memory::Value<'gc> {
            crate::memory::Value::$variant(self.into())
        }
    };
}

impl<'gc> Value<'gc> {
    pub fn whnf(&self) -> bool {
        !matches!(self, Self::App(_) | Self::Ref(_))
    }
}

impl<'gc> Node<'gc> {
    /// View the inner contents.
    pub fn unpack(&self) -> Value<'gc> {
        self.0.get()
    }

    pub fn unpack_fun(&self) -> Pointer<'gc> {
        self.unpack().unwrap_app().fun
    }

    pub fn unpack_arg(&self) -> Pointer<'gc> {
        if let Value::App(_) = self.unpack() {
            self.unpack().unwrap_app().arg
        } else {
            panic!("Tried to unwrap non-app node: {self:?}")
        }
    }
}

impl<'gc> From<Value<'gc>> for Node<'gc> {
    fn from(value: Value<'gc>) -> Self {
        Self(Lock::new(value))
    }
}

impl<'gc> From<bool> for Value<'gc> {
    fn from(value: bool) -> Self {
        // Church-encoded booleans
        if value {
            Self::Combinator(Combinator::A)
        } else {
            Self::Combinator(Combinator::K)
        }
    }
}

impl<'gc> From<()> for Value<'gc> {
    fn from(_: ()) -> Self {
        // Church-encoded unit value
        Self::Combinator(Combinator::I)
    }
}

#[derive(Clone, Copy, Collect, PartialEq)]
#[collect(no_drop)]
pub struct App<'gc> {
    pub fun: Pointer<'gc>,
    pub arg: Pointer<'gc>,
}

impl<'gc> Debug for App<'gc> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("App")
            .field("fun", &Gc::as_ptr(self.fun.ptr))
            .field("arg", &Gc::as_ptr(self.arg.ptr))
            .finish()
    }
}

impl<'gc> FromValue<'gc> for App<'gc> {
    from_gc_value!(App);
}

pub(crate) fn make_pair<'gc>(
    mc: &Mutation<'gc>,
    (x, y): (Pointer<'gc>, Pointer<'gc>),
) -> Pointer<'gc> {
    let pair = Pointer::new(mc, Value::Combinator(Combinator::PAIR));
    Pointer::new(
        mc,
        Value::App(App {
            fun: Pointer::new(mc, Value::App(App { fun: pair, arg: x })),
            arg: y,
        }),
    )
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
#[derive(Collect, Clone, Copy)]
#[collect(no_drop)]
pub struct Pointer<'gc> {
    ptr: Gc<'gc, Node<'gc>>,
}

impl<'gc> Debug for Pointer<'gc> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Pointer")
            .field(&Gc::as_ptr(self.ptr))
            .field(&self.ptr.as_ref().unpack())
            .finish()
    }
}

impl<'gc> Pointer<'gc> {
    #[tracing::instrument(skip(mc, value), fields(T = std::any::type_name::<T>()), ret)]
    pub fn new<T: IntoValue<'gc>>(mc: &Mutation<'gc>, value: T) -> Self {
        Gc::new(mc, Node::from(value.into_value(mc))).into()
    }

    pub fn set<T: IntoValue<'gc>>(&self, mc: &Mutation<'gc>, value: T) {
        self.ptr.unlock(mc).set(value.into_value(mc))
    }

    pub fn set_ref(&self, mc: &Mutation<'gc>, ptr: Pointer<'gc>) -> Pointer<'gc> {
        self.ptr.unlock(mc).set(Value::Ref(ptr));
        ptr
    }

    pub fn follow(&self) -> Self {
        let mut ptr = *self;
        while let Value::Ref(p) = ptr.unpack() {
            ptr = p;
        }
        ptr
    }

    pub fn forward(&mut self, _mc: &Mutation<'gc>) -> Self {
        // TODO: forward all pointers to final destination?
        while let Value::Ref(p) = self.unpack() {
            *self = p;
        }
        *self
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
    fn from(value: Gc<'gc, Node<'gc>>) -> Self {
        Self { ptr: value }
    }
}

#[derive(thiserror::Error, Debug)]
#[error("could not perform conversion, expected {expected}, got {got}")]
pub struct ConversionError {
    pub expected: &'static str,
    pub got: &'static str,
}

pub trait FromValue<'gc>: Sized {
    fn from_value(mc: &Mutation<'gc>, value: Value<'gc>) -> Result<Self, ConversionError>;
}

impl<'gc, T: From<Value<'gc>>> FromValue<'gc> for T {
    fn from_value(_mc: &Mutation<'gc>, value: Value<'gc>) -> Result<Self, ConversionError> {
        Ok(value.into())
    }
}

pub trait IntoValue<'gc> {
    fn into_value(self, mc: &Mutation<'gc>) -> Value<'gc>;
}

impl<'gc, T> IntoValue<'gc> for T
where
    Value<'gc>: From<T>,
{
    fn into_value(self, _mc: &Mutation<'gc>) -> Value<'gc> {
        self.into()
    }
}

#[cfg(feature = "file")]
impl crate::file::Program {
    pub fn deserialize<'gc>(&self, mc: &Mutation<'gc>, tick_table: &mut TickTable) -> Pointer<'gc> {
        // Algorithm: a two-pass scan through the program body
        use crate::file::Expr;

        // First, allocate placeholder memory locations for each node
        let heap: Vec<_> = self
            .body
            .iter()
            .map(|_| Pointer::new(mc, Value::Combinator(Combinator::Y)))
            .collect();

        // Then, modify each node in-place, according to the
        for (i, expr) in self.body.iter().enumerate() {
            heap[i].set(
                mc,
                match expr {
                    Expr::App(f, a) => Value::App(App {
                        fun: heap[*f],
                        arg: heap[*a],
                    }),
                    Expr::Ref(r) => Value::Ref(heap[self.defs[*r]]),
                    Expr::Prim(c) => match *c {
                        Combinator::PNull => ForeignPtr::null().into(),
                        Combinator::StdIn => unsafe { ffi::bfile::mystdin() }.into_value(mc),
                        Combinator::StdOut => unsafe { ffi::bfile::mystdout() }.into_value(mc),
                        Combinator::StdErr => unsafe { ffi::bfile::mystderr() }.into_value(mc),
                        c => Value::Combinator(c),
                    },
                    Expr::Float(f) => Value::Float(Float::from(*f)),
                    Expr::Int(i) => Value::Integer(Integer::from(*i)),
                    Expr::String(s) => Value::String(VString::new(mc, s)),
                    Expr::Tick(name) => Value::Tick(tick_table.add_entry(name)),
                    Expr::Ffi(name) => FfiSymbol::lookup(name).map_or_else(
                        || {
                            log::warn!("Could not find {name}");
                            BadDyn::new(mc, name).into()
                        },
                        Value::Ffi,
                    ),
                    Expr::Array(_, _) => todo!("arrays"),
                    Expr::Unknown(s) => panic!("unable to deserialize {s}"),
                },
            );
            log::trace!("*** Initialized from expr #{i}: {expr}");
            log::trace!("    {:?}", heap[i]);
        }

        heap[self.root]
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

        let arena: Arena<Root> = Arena::new(|m| Node::from(Value::String(VString::new(m, &hello))));
        arena.mutate(|m, s| {
            assert_eq!(s.unpack(), Value::String(VString::new(m, "hello")));
        });
    }

    #[test]
    fn make_number_cell() {
        // We need to create an arena here (despite no GC allocation) because of 'gc invariance.
        type Root = Rootable![Node<'_>];

        let mut arena: Arena<Root> = Arena::new(|_m| Node::from(Value::Integer(42.into())));
        arena.mutate(|_, v| assert_eq!(v.unpack(), Value::Integer(42.into())));

        arena.mutate_root(|_, v| {
            *v = Node::from(Value::Float(64.0.into()));
        });
        arena.mutate(|_, v| assert_eq!(v.unpack(), Value::Float(64.0.into())));
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
                Node::from(Value::String(VString::new(m, "hello"))),
            ));
        });

        arena.mutate_root(|m, v| {
            let arg = Pointer::from(v[1]);
            let fun = Pointer::from(v[0]);
            v.push(Gc::new(m, Node::from(Value::App(App { fun, arg }))));
        });

        arena.mutate(|_m, v| {
            let arg = Pointer::from(v[1]);
            let fun = Pointer::from(v[0]);
            assert_eq!(v[2].unpack(), Value::App(App { fun, arg }));
        });

        arena.collect_all();

        arena.mutate(|_m, v| {
            let arg = Pointer::from(v[1]);
            let fun = Pointer::from(v[0]);
            assert_eq!(v[2].unpack(), Value::App(App { fun, arg }));
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
            if !matches!(app, Value::App(_)) {
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

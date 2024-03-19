use crate::integer::Integer;
use core::{fmt::Display, ops::*};
use gc_arena::Collect;

#[cfg(target_pointer_width = "32")]
type FloatInner = f32;

#[cfg(target_pointer_width = "64")]
type FloatInner = f64;

#[derive(Debug, Clone, Copy, Collect, PartialEq, PartialOrd)]
#[repr(transparent)]
#[collect(require_static)]
pub struct Float(FloatInner);

impl From<FloatInner> for Float {
    fn from(value: FloatInner) -> Self {
        Self(value)
    }
}

impl Display for Float {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

macro_rules! op {
    (fsize::$method:ident as $name:ident(self, $rhs:ident)) => {
        pub fn $name(&self, $rhs: &Self) -> Self {
            self.inner().$method($rhs.inner()).into()
        }
    };
    (fsize::$method:ident as $name:ident(self)) => {
        pub fn $name(&self) -> Self {
            self.inner().$method().into()
        }
    };
    (fsize::$method:ident as $name:ident(self, &$rhs:ident) -> $ret:ty) => {
        pub fn $name(&self, $rhs: &Self) -> $ret {
            self.inner().$method(&$rhs.inner()).into()
        }
    };
}

impl Float {
    fn inner(self) -> FloatInner {
        self.0
    }
    op![fsize::neg as fneg(self)];
    op![fsize::add as fadd(self, rhs)];
    op![fsize::sub as fsub(self, rhs)];
    op![fsize::mul as fmul(self, rhs)];
    op![fsize::div as fdiv(self, rhs)];

    pub fn from_integer(integer: &Integer) -> Self {
        Self(integer.signed() as FloatInner)
    }
}

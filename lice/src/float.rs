use crate::{integer::Integer, string::hstring_from_utf8, memory::Pointer};
use core::{fmt::Display, ops::*};
use gc_arena::{Collect, Mutation};

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
        pub fn $name(self, $rhs: Self) -> Self {
            self.inner().$method($rhs.inner()).into()
        }
    };
    (fsize::$method:ident as $name:ident(self)) => {
        pub fn $name(self) -> Self {
            self.inner().$method().into()
        }
    };
    (fsize::$method:ident as $name:ident(self, &$rhs:ident) -> $ret:ty) => {
        pub fn $name(self, $rhs: Self) -> $ret {
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
    op![fsize::eq as feq(self, &rhs) -> bool];
    op![fsize::ne as fne(self, &rhs) -> bool];
    op![fsize::lt as flt(self, &rhs) -> bool];
    op![fsize::le as fle(self, &rhs) -> bool];
    op![fsize::gt as fgt(self, &rhs) -> bool];
    op![fsize::ge as fge(self, &rhs) -> bool];
    op![fsize::acos as facos(self)];
    op![fsize::asin as fasin(self)];
    op![fsize::atan as fatan(self)];
    op![fsize::atan2 as fatan2(self, rhs)];
    op![fsize::cos as fcos(self)];
    op![fsize::exp as fexp(self)];
    op![fsize::log as flog(self, rhs)];
    op![fsize::sin as fsin(self)];
    op![fsize::sqrt as fsqrt(self)];
    op![fsize::tan as ftan(self)];

    pub fn from_integer(integer: Integer) -> Self {
        Self(integer.signed() as FloatInner)
    }

    pub fn fshow<'gc>(self, mc: &Mutation<'gc>) -> Pointer<'gc> {
        let mut buf = ryu::Buffer::new();
        hstring_from_utf8(mc, buf.format(self.inner()))
    }
}

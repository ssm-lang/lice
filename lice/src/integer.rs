use core::{
    fmt::Display,
    ops::{BitAnd, BitOr, BitXor, Not},
};
use gc_arena::Collect;

#[derive(Debug, Clone, Copy, Collect, PartialEq, Eq)]
#[repr(transparent)]
#[collect(require_static)]
pub struct Integer(usize);

impl Display for Integer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

macro_rules! op {
    (usize::$method:ident as $name:ident(self, $rhs:ident)) => {
        pub fn $name(&self, $rhs: &Self) -> Self {
            self.unsigned().$method($rhs.unsigned()).into()
        }
    };
    (isize::$method:ident as $name:ident(self, $rhs:ident)) => {
        pub fn $name(&self, $rhs: &Self) -> Self {
            self.signed().$method($rhs.signed()).into()
        }
    };
    (usize::$method:ident as $name:ident(self)) => {
        pub fn $name(&self) -> Self {
            self.unsigned().$method().into()
        }
    };
    (isize::$method:ident as $name:ident(self)) => {
        pub fn $name(&self) -> Self {
            self.signed().$method().into()
        }
    };
    (usize::$method:ident as $name:ident(self, &$rhs:ident)) => {
        pub fn $name(&self, $rhs: &Self) -> Self {
            self.unsigned().$method(&$rhs.unsigned()).into()
        }
    };
    (isize::$method:ident as $name:ident(self, &$rhs:ident)) => {
        pub fn $name(&self, $rhs: &Self) -> Self {
            self.signed().$method(&$rhs.signed()).into()
        }
    };
}

impl Integer {
    pub fn signed(self) -> isize {
        self.0 as isize
    }

    pub fn unsigned(self) -> usize {
        self.0
    }

    pub fn subtract(&self, rhs: &Self) -> Self {
        rhs.isub(self)
    }

    op![isize::wrapping_neg as ineg(self)];
    op![isize::wrapping_add as iadd(self, rhs)];
    op![isize::wrapping_sub as isub(self, rhs)];
    op![isize::wrapping_mul as imul(self, rhs)];
    op![isize::wrapping_div as iquot(self, rhs)];
    op![isize::wrapping_rem as irem(self, rhs)];
    op![isize::lt as ilt(self, &rhs)];
    op![isize::le as ile(self, &rhs)];
    op![isize::gt as igt(self, &rhs)];
    op![isize::ge as ige(self, &rhs)];
    op![usize::wrapping_div as uquot(self, rhs)];
    op![usize::wrapping_rem as urem(self, rhs)];
    op![usize::lt as ult(self, &rhs)];
    op![usize::le as ule(self, &rhs)];
    op![usize::gt as ugt(self, &rhs)];
    op![usize::ge as uge(self, &rhs)];
    op![usize::not as binv(self)];
    op![usize::bitand as band(self, rhs)];
    op![usize::bitor as bor(self, rhs)];
    op![usize::bitxor as bxor(self, rhs)];

    pub fn ushl(&self, rhs: &Self) -> Self {
        self.unsigned().wrapping_shl(rhs.unsigned() as u32).into()
    }

    pub fn ushr(&self, rhs: &Self) -> Self {
        self.unsigned().wrapping_shr(rhs.unsigned() as u32).into()
    }

    pub fn ashr(&self, rhs: &Self) -> Self {
        self.signed().wrapping_shr(rhs.unsigned() as u32).into()
    }
}

impl From<usize> for Integer {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

macro_rules! impl_from {
    ($ty:ty as $aty:ty) => {
        impl From<$ty> for Integer {
            fn from(value: $ty) -> Self {
                Self(value as $aty)
            }
        }
    };
    (|$arg:ident: $ty:ty| $expr:expr) => {
        impl From<$ty> for Integer {
            fn from($arg: $ty) -> Self {
                Self($expr)
            }
        }
    };
}

impl_from!(isize as usize);
impl_from!(i8 as usize);
impl_from!(i16 as usize);
impl_from!(i32 as usize);
impl_from!(i64 as usize);
impl_from!(|value: bool| if value { 1 } else { 0 });

use core::{
    char,
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
        pub fn $name(self, $rhs: Self) -> Self {
            self.unsigned().$method($rhs.unsigned()).into()
        }
    };
    (isize::$method:ident as $name:ident(self, $rhs:ident)) => {
        pub fn $name(self, $rhs: Self) -> Self {
            self.signed().$method($rhs.signed()).into()
        }
    };
    (usize::$method:ident as $name:ident(self)) => {
        pub fn $name(self) -> Self {
            self.unsigned().$method().into()
        }
    };
    (isize::$method:ident as $name:ident(self)) => {
        pub fn $name(self) -> Self {
            self.signed().$method().into()
        }
    };
    (usize::$method:ident as $name:ident(self, &$rhs:ident) -> $rty:ty) => {
        pub fn $name(self, $rhs: Self) -> $rty {
            self.unsigned().$method(&$rhs.unsigned()).into()
        }
    };
    (isize::$method:ident as $name:ident(self, &$rhs:ident) -> $rty:ty) => {
        pub fn $name(self, $rhs: Self) -> $rty {
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

    pub fn subtract(self, rhs: Self) -> Self {
        rhs.isub(self)
    }

    op![isize::wrapping_neg as ineg(self)];
    op![isize::wrapping_add as iadd(self, rhs)];
    op![isize::wrapping_sub as isub(self, rhs)];
    op![isize::wrapping_mul as imul(self, rhs)];
    op![isize::wrapping_div as iquot(self, rhs)];
    op![isize::wrapping_rem as irem(self, rhs)];
    op![isize::lt as ilt(self, &rhs) -> bool];
    op![isize::le as ile(self, &rhs) -> bool];
    op![isize::gt as igt(self, &rhs) -> bool];
    op![isize::ge as ige(self, &rhs) -> bool];
    op![usize::wrapping_div as uquot(self, rhs)];
    op![usize::wrapping_rem as urem(self, rhs)];
    op![usize::lt as ult(self, &rhs) -> bool];
    op![usize::le as ule(self, &rhs) -> bool];
    op![usize::gt as ugt(self, &rhs) -> bool];
    op![usize::ge as uge(self, &rhs) -> bool];
    op![usize::not as binv(self)];
    op![usize::bitand as band(self, rhs)];
    op![usize::bitor as bor(self, rhs)];
    op![usize::bitxor as bxor(self, rhs)];

    pub fn ieq(self, rhs: Self) -> bool {
        self.eq(&rhs)
    }

    pub fn ine(self, rhs: Self) -> bool {
        self.ne(&rhs)
    }

    pub fn ushl(self, rhs: Self) -> Self {
        self.unsigned().wrapping_shl(rhs.unsigned() as u32).into()
    }

    pub fn ushr(self, rhs: Self) -> Self {
        self.unsigned().wrapping_shr(rhs.unsigned() as u32).into()
    }

    pub fn ashr(self, rhs: Self) -> Self {
        self.signed().wrapping_shr(rhs.unsigned() as u32).into()
    }
}

macro_rules! impl_convert_value {
    ($ty:ty) => {
        impl TryFrom<crate::memory::Value<'_>> for $ty {
            type Error = &'static str;
            fn try_from(value: crate::memory::Value<'_>) -> Result<Self, Self::Error> {
                Ok(Integer::try_from(value)?.into())
            }
        }
        impl<'gc> From<$ty> for crate::memory::Value<'gc> {
            fn from(value: $ty) -> crate::memory::Value<'gc> {
                Integer::from(value).into()
            }
        }
    };
}

macro_rules! impl_from {
    ($ty:ty) => {
        impl From<$ty> for Integer {
            fn from(value: $ty) -> Self {
                Self(value as usize)
            }
        }
    };
}

macro_rules! impl_into {
    ($ty:ty) => {
        impl From<Integer> for $ty {
            fn from(value: Integer) -> Self {
                value.0 as $ty
            }
        }
    };
}

macro_rules! impl_convert {
    ($ty:ty) => {
        impl_from!($ty);
        impl_into!($ty);
        impl_convert_value!($ty);
    };
}

impl_convert!(isize);
impl_convert!(i8);
impl_convert!(i16);
impl_convert!(i32);
impl_convert!(i64);
impl_convert!(usize);
impl_convert!(u8);
impl_convert!(u16);
impl_convert!(u32);
impl_convert!(u64);

impl_from!(char);
impl From<Integer> for char {
    fn from(value: Integer) -> Self {
        let Ok(value) = u32::try_from(value.0) else {
            return Default::default();
        };
        char::from_u32(value).unwrap_or_default()
    }
}
impl_convert_value!(char);

// impl From<bool> for Integer {
//     fn from(value: bool) -> Self {
//         Self(if value { 1 } else { 0 })
//     }
// }
// impl From<Integer> for bool {
//     fn from(value: Integer) -> Self {
//         value.0 != 0
//     }
// }
// impl_convert_value!(bool);

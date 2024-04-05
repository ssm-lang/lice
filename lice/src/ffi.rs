// for each function, we need to know:
// - how many args (arity); all need to be evaluated strictly
// - types of args and return value; also includes ffi types
use crate::{
    integer::Integer,
    memory::{FromValue, IntoValue},
    string::VString,
};
use core::{ptr::null, str::FromStr};
use gc_arena::{Collect, Mutation};
use lice_macros::Reduce;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForeignPtr(*mut ());

/// Raw pointers should not be traced
unsafe impl Collect for ForeignPtr {
    fn needs_trace() -> bool {
        false
    }
    fn trace(&self, _cc: &gc_arena::Collection) {}
}

impl ForeignPtr {
    pub fn null() -> Self {
        Self::from(null::<()>())
    }

    pub fn as_ptr(self) -> *mut u8 {
        self.0.cast::<u8>()
    }

    pub fn peq(self, rhs: Self) -> bool {
        self.eq(&rhs)
    }

    pub fn padd(self, rhs: Integer) -> Self {
        self.as_ptr().wrapping_offset(rhs.signed()).into()
    }

    pub fn psub(self, rhs: Self) -> Integer {
        unsafe { self.as_ptr().offset_from(rhs.as_ptr()) }.into()
    }
}

impl<T> From<*mut T> for ForeignPtr {
    fn from(value: *mut T) -> Self {
        Self(value.cast::<()>())
    }
}

impl<T> From<*const T> for ForeignPtr {
    fn from(value: *const T) -> Self {
        Self::from(value.cast_mut())
    }
}

impl<T> From<ForeignPtr> for *mut T {
    fn from(value: ForeignPtr) -> Self {
        value.0.cast::<T>()
    }
}

impl<T> From<ForeignPtr> for *const T {
    fn from(value: ForeignPtr) -> Self {
        <*mut T>::from(value).cast_const()
    }
}

impl<'gc> FromValue<'gc> for ForeignPtr {
    from_gc_value!(ForeignPtr);
}

impl<'gc, T> FromValue<'gc> for *const T {
    from_gc_value!(ForeignPtr);
}

impl<'gc, T> FromValue<'gc> for *mut T {
    from_gc_value!(ForeignPtr);
}

impl<'gc, T> IntoValue<'gc> for *const T {
    into_gc_value!(ForeignPtr);
}

impl<'gc, T> IntoValue<'gc> for *mut T {
    into_gc_value!(ForeignPtr);
}

/// Missing symbol; generates an error when reduced (but inert otherwise).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Collect)]
#[collect(no_drop)]
pub struct BadDyn<'gc>(VString<'gc>);

impl<'gc> BadDyn<'gc> {
    pub fn new(mc: &Mutation<'gc>, name: &str) -> Self {
        Self(VString::new(mc, name))
    }
    pub fn name(&self) -> &str {
        self.0.as_ref()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Collect, Reduce, parse_display::Display)]
#[cfg_attr(feature = "std", derive(parse_display::FromStr))]
#[cfg_attr(test, derive(strum::EnumIter))]
#[allow(non_camel_case_types)]
#[collect(require_static)]
pub enum FfiSymbol {
    // GETRAW,
    // GETTIMEMILLI,
    #[reduce(from = "!float", to_io = "float")]
    acos,
    #[reduce(from = "!float", to_io = "float")]
    asin,
    #[reduce(from = "!float", to_io = "float")]
    atan,
    #[reduce(from = "!a:float !b:float", to_io = "float")]
    atan2,
    #[reduce(from = "!float", to_io = "float")]
    cos,
    #[reduce(from = "!float", to_io = "float")]
    exp,
    #[reduce(from = "!x:float", to_io = "float")]
    log,
    #[reduce(from = "!float", to_io = "float")]
    sin,
    #[reduce(from = "!float", to_io = "float")]
    sqrt,

    #[reduce(from = "!float", to_io = "float")]
    tan,

    #[reduce(from = "!path:char* !mode:char*", to_io = "FILE*")]
    fopen,
    #[reduce(from = "!FILE*", to_io = "handle")]
    add_FILE,
    #[reduce(from = "!handle", to_io = "handle")]
    add_utf8,
    #[reduce(from = "!handle", to_io = "void")]
    closeb,
    #[reduce(from = "!handle", to_io = "void")]
    flushb,
    #[reduce(from = "!handle", to_io = "char")]
    getb,
    #[reduce(from = "!char !handle", to_io = "void")]
    putb,
    #[reduce(from = "!char !handle", to_io = "void")]
    ungetb,
    #[reduce(from = "!cmd_str", to_io = "int")]
    system,
    #[reduce(from = "!pre_str !post_str", to_io = "path_str")]
    tmpname,
    #[reduce(from = "!path_str", to_io = "int")]
    unlink,

    #[reduce(from = "!int", to_io = "ptr")]
    malloc,
    #[reduce(from = "!ptr", to_io = "unit")]
    free,
}

impl FfiSymbol {
    pub fn lookup(name: &str) -> Option<Self> {
        Self::from_str(name).ok()
    }
}

#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
pub mod bfile {
    include!(concat!(env!("OUT_DIR"), "/bfile.rs"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::combinator::Reduce;
    #[test]
    fn ffi_is_io() {
        use strum::IntoEnumIterator;
        for sym in FfiSymbol::iter() {
            assert!(sym.io_action(), "{sym} should be an IO action")
        }
    }
}

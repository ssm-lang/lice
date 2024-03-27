// for each function, we need to know:
// - how many args (arity); all need to be evaluated strictly
// - types of args and return value; also includes ffi types
use crate::{memory::Value, string::VString};
use core::str::FromStr;
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

impl<T> From<*mut T> for ForeignPtr {
    fn from(value: *mut T) -> Self {
        Self(value as *mut ())
    }
}

impl<T> From<*const T> for ForeignPtr {
    fn from(value: *const T) -> Self {
        Self(value as *mut ())
    }
}

impl<T> From<ForeignPtr> for *const T {
    fn from(value: ForeignPtr) -> Self {
        value.0 as *const T
    }
}

impl<T> From<ForeignPtr> for *mut T {
    fn from(value: ForeignPtr) -> Self {
        value.0 as *mut T
    }
}

impl<'gc, T> TryFrom<Value<'gc>> for *const T {
    type Error = <ForeignPtr as TryFrom<Value<'gc>>>::Error;
    fn try_from(value: Value) -> Result<Self, Self::Error> {
        Ok(ForeignPtr::try_from(value)?.into())
    }
}

impl<'gc, T> TryFrom<Value<'gc>> for *mut T {
    type Error = <ForeignPtr as TryFrom<Value<'gc>>>::Error;
    fn try_from(value: Value) -> Result<Self, Self::Error> {
        Ok(ForeignPtr::try_from(value)?.into())
    }
}

impl<T> From<*const T> for Value<'_> {
    fn from(value: *const T) -> Self {
        ForeignPtr::from(value).into()
    }
}

impl<T> From<*mut T> for Value<'_> {
    fn from(value: *mut T) -> Self {
        ForeignPtr::from(value).into()
    }
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
#[allow(non_camel_case_types)]
#[collect(require_static)]
pub enum FfiSymbol {
    // GETRAW,
    // GETTIMEMILLI,
    #[reduce(from = "!float", returns = "float")]
    acos,
    #[reduce(from = "!float", returns = "float")]
    asin,
    #[reduce(from = "!float", returns = "float")]
    atan,
    #[reduce(from = "!a_float !b_float", returns = "float")]
    atan2,
    #[reduce(from = "!float", returns = "float")]
    cos,
    #[reduce(from = "!float", returns = "float")]
    exp,
    #[reduce(from = "!float", returns = "float")]
    log,
    #[reduce(from = "!float", returns = "float")]
    sin,
    #[reduce(from = "!float", returns = "float")]
    sqrt,

    #[reduce(from = "!float", returns = "float")]
    tan,

    #[reduce(from = "!path_str !mode_str", returns = "FILE_ptr")]
    fopen,
    #[reduce(from = "!FILE_ptr", returns = "handle")]
    add_FILE,
    #[reduce(from = "!handle", returns = "handle")]
    add_utf8,
    #[reduce(from = "!handle")]
    closeb,
    #[reduce(from = "!handle")]
    flushb,
    #[reduce(from = "!handle", returns = "char")]
    getb,
    #[reduce(from = "!char !handle")]
    putb,
    #[reduce(from = "!char !handle")]
    ungetb,
    #[reduce(from = "!cmd_str", returns = "int")]
    system,
    #[reduce(from = "!pre_str !post_str", returns = "path_str")]
    tmpname,
    #[reduce(from = "!path_str", returns = "int")]
    unlink,
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

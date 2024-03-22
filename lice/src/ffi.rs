use gc_arena::{Collect, Mutation};

use crate::string::VString;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Collect)]
#[collect(no_drop)]
pub struct Ffi<'gc>(VString<'gc>);

impl<'gc> Ffi<'gc> {
    pub fn new(mc: &Mutation<'gc>, name: &str) -> Self {
        Self(VString::new(mc, name))
    }
}

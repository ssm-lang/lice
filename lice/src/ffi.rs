use gc_arena::Collect;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Collect)]
#[collect(require_static)]
pub struct Ffi();

impl Ffi {
    pub fn new(_name: &str) -> Self {
        Self()
    }
}

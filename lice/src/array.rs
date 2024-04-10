use crate::memory::{FromValue, Pointer};
use core::iter::repeat;
use gc_arena::{lock::RefLock, Collect, Gc, Mutation};

#[derive(Debug, Clone, Copy, Collect, PartialEq)]
#[collect(no_drop)]
pub struct Array<'gc>(Gc<'gc, RefLock<Vec<Pointer<'gc>>>>);

impl<'gc> Array<'gc> {
    pub fn new(mc: &Mutation<'gc>, len: usize, init: Pointer<'gc>) -> Self {
        Self(Gc::new(mc, RefLock::new(repeat(init).take(len).collect())))
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.0.borrow().len()
    }

    pub fn read(self, idx: usize) -> Pointer<'gc> {
        *self.0.borrow().get(idx).expect("FIXME: bounds-checking?")
    }

    pub fn write(&self, mc: &Mutation<'gc>, idx: usize, elem: Pointer<'gc>) {
        *self
            .0
            .borrow_mut(mc)
            .get_mut(idx)
            .expect("FIXME: bounds-checking?") = elem;
    }
}

impl<'gc> FromValue<'gc> for Array<'gc> {
    from_gc_value!(Array);
}

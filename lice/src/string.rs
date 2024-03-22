use gc_arena::{Collect, Gc, Mutation};

/// A vector-backed, GC-managed string.
///
/// TODO: this memory layout is really quite inefficient. Consider making this a `Gc<str>`,
/// a `Gc<[u8]>`, or a `Gc`-backed `ThinStr`.
#[derive(Debug, Clone, Copy, Collect, PartialEq, Eq)]
#[collect(no_drop)]
pub struct VString<'gc>(Gc<'gc, String>);

impl<'gc> VString<'gc> {
    pub fn new(m: &Mutation<'gc>, s: &str) -> Self {
        Self(Gc::new(m, s.to_string()))
    }
}

impl<'gc> AsRef<str> for VString<'gc> {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl<'gc> AsRef<[u8]> for VString<'gc> {
    fn as_ref(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

impl<'gc> core::fmt::Display for VString<'gc> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

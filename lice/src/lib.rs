#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod tag;

#[cfg(feature = "file")]
pub mod file;

#[cfg(feature = "graph")]
pub mod graph;

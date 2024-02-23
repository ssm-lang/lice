#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod combinator;

#[cfg(feature = "rt")]
pub mod memory;

#[cfg(feature = "rt")]
pub mod eval;

#[cfg(feature = "rt")]
pub mod string;

#[cfg(feature = "file")]
pub mod file;

#[cfg(feature = "graph")]
pub mod graph;

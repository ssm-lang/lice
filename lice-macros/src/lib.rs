mod reduce;

use proc_macro::TokenStream;

/// Derive the `Reduce` trait implementation for an `enum` definition.
#[proc_macro_derive(Reduce, attributes(reduce))]
pub fn reduce(input: TokenStream) -> TokenStream {
    reduce::reduce(input)
}

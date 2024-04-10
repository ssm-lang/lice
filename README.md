# Lice

## Dependencies

- [Rust toolchain](https://rustup.rs/) (for building the runtime)
- C toolchain (for linking with libc)
- [`just`](https://just.systems) (for running examples and tests)
  - Technically optional but recommended, for convenience.
- [GHC](https://www.haskell.org/ghcup/) (for building MHS)
  - `GHC` is technically optional but recommended, since self-hosted
    MHS is quite slow compared to GHC-hosted MHS.

## Getting started

Install dependencies (see links).

Several convenience targets are defined by the [`justfile`](./justfile).
You can list them by running `just` without arguments.

Working with [examples](./examples/):

```console
$ just eg Example       # Build and run a basic example
Compiling Example to Example.comb... OK
Some factorials
[1,2,6,3628800]
$ just trace StrictData # Build and run an example with verbose logging
Compiling StrictData to StrictData.comb... OK
### OMITTED LOGGING VERBIAGE ###
```

Working with [MicroHs regression tests](https://github.com/augustss/MicroHs/tree/master/tests)
(which are vendored):

```console
$ just compile-test Arith # Build MicroHs regression test
Compiling Arith to Arith.comb... OK
$ just test Arith         # Run MicroHs regression test
Arith        ... [PASS]
```

## TODOs

What is left to achieve language feature-parity with MHS's `eval.c`:

- Garbage collection (we never run `gc_arena`'s `collect()` function lol)
- Arrays
- `StrEqual` and `Compare`
- Node casting
- RNF
- Most FFI stubs (famous last words: _this should be straightforward_)
- Dynamic FFI symbol resolution
- Serialization/deserialization

And optimizations to address some known deficiencies (though it would be
interesting to benchmark without these optimizations):

- Packed memory layout
- String memory layout
- Array memory layout
- Be less safe about error-handling (many are unnecessary)

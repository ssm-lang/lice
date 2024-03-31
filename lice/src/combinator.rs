//! Definitions for combinators that appear in the combinator graph.

#[cfg(feature = "std")]
pub use parse_display::ParseError;

#[cfg(feature = "rt")]
pub use num_enum::TryFromPrimitiveError;

/// Properties of combinators (which may be partially evaluated at compile time).
///
/// The derive macro for `Reduce` uses the `#[reduce(...)]` helper attribute to recieve
/// a specification of the reduction rule, i.e., the redex and reduct. At the moment, the macro
/// doesn't do anything with that information other than compute the arity (number of redexes).
pub trait Reduce {
    fn strictness(&self) -> &'static [bool];

    fn redux(&self) -> Option<&'static [ReduxCode]>;

    /// How many arguments are needed for reduction.
    #[inline]
    fn arity(&self) -> usize {
        self.strictness().len()
    }
}

pub enum ReduxCode {
    Arg(usize),
    Top,
    App,
}

/// Named, primitive values applied to other nodes in a combinator graph.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    lice_macros::Reduce,
    parse_display::Display,
    num_enum::IntoPrimitive,
    num_enum::TryFromPrimitive,
)]
#[cfg_attr(feature = "std", derive(parse_display::FromStr))]
#[cfg_attr(feature = "rt", repr(u8))]
pub enum Combinator {
    /** Turner combinators **/
    #[reduce(from = "f g x", to = "f x (g x)")]
    S,
    #[reduce(from = "x y", to = "x")]
    K,
    #[reduce(from = "x", to = "x")]
    I,
    #[reduce(from = "f g x", to = "f (g x)")]
    B,
    #[reduce(from = "f x y", to = "f y x")]
    C,
    #[reduce(from = "x y", to = "y")]
    A,
    #[reduce(from = "f", to = "f ^")]
    Y,
    #[display("S'")]
    #[reduce(from = "c f g x", to = "c (f x) (g x)")]
    SS,
    #[display("B'")]
    #[reduce(from = "c f g x", to = "c f (g x)")]
    BB,
    #[display("C'")]
    #[reduce(from = "c f x y", to = "c (f y) x")]
    CC,
    #[reduce(from = "x y f", to = "f x y")]
    P,
    #[reduce(from = "y f x", to = "f x y")]
    R,
    #[reduce(from = "x y z f", to = "f x y")]
    O,
    #[reduce(from = "x f", to = "f x")]
    U,
    #[reduce(from = "f x y", to = "f x")]
    Z,
    #[reduce(from = "x y z", to = "x")]
    K2,
    #[reduce(from = "x y z w", to = "x")]
    K3,
    #[reduce(from = "x y z w v", to = "x")]
    K4,
    #[display("C'B")]
    #[reduce(from = "f g x y", to = "f x (g y)")]
    CCB,

    /*** Strictness and errors ***/
    #[display("seq")]
    #[reduce(from = "!to_whnf continuation", to = "continuation")]
    Seq,
    /// Fully reduce argument to normal form, and return `()`.
    ///
    /// TODO: what is the number argument??
    #[display("rnf")]
    #[reduce(from = "!num !term")]
    Rnf,
    /// Bottom value, throws an exception with the Haskell string as the error message.
    #[display("error")]
    #[reduce(from = "!msg")]
    ErrorMsg,
    /// Bottom value, throws an exception with the Haskell string as the error message.
    #[display("noDefault")]
    #[reduce(from = "!msg")]
    NoDefault,
    #[display("noMatch")]
    #[reduce(from = "!msg !line_nr !col_nr")]
    NoMatch,

    /** Comparisons **/
    // TODO: clarify the types. These seem to be (partially?) overloaded.
    #[display("equal")]
    #[reduce(from = "!lhs !rhs")]
    Equal,

    /// String equality; same implementation as Equal
    ///
    /// TODO: figure out if this is for CStrings or Haskell strings.
    #[display("sequal")]
    #[reduce(from = "!lhs !rhs")]
    StrEqual,

    #[display("compare")]
    #[reduce(from = "!lhs !rhs")]
    Compare,

    /// String comparison; same implementation as Compare
    ///
    /// TODO: figure out if this is for CStrings or Haskell strings.
    #[display("scmp")]
    #[reduce(from = "!lhs !rhs")]
    StrCmp,

    /// Integer comparison; same implementation as Compare
    ///
    /// TODO: figure out the types?? What is the diff between this and compare
    #[display("icmp")]
    #[reduce(from = "!lhs !rhs")]
    IntCmp,

    /** Integers **/

    /// Reinterpret cast to integer.
    #[display("toInt")]
    #[reduce(from = "!int")]
    ToInt,

    #[display("neg")]
    #[reduce(from = "!int")]
    Neg,
    #[display("+")]
    #[reduce(from = "!lhs !rhs")]
    Add,
    #[display("-")]
    #[reduce(from = "!minuend !subtrahend")]
    Sub,
    /// The same as `flip (-)`.
    #[display("subtract")]
    #[reduce(from = "!subtrahend !minuend")]
    Subtract,
    #[display("*")]
    #[reduce(from = "!lhs !rhs")]
    Mul,
    #[display("quot")]
    #[reduce(from = "!dividend !divisor")]
    Quot,
    #[display("rem")]
    #[reduce(from = "!dividend !divisor")]
    Rem,

    /// Signed integer equality.
    ///
    /// NOTE: this is not overloaded.
    #[display("==")]
    #[reduce(from = "!lhs !rhs")]
    Eq,
    #[display("/=")]
    #[reduce(from = "!lhs !rhs")]
    Ne,
    #[display("<")]
    #[reduce(from = "!lhs !rhs")]
    Lt,
    #[display("<=")]
    #[reduce(from = "!lhs !rhs")]
    Le,
    #[display(">")]
    #[reduce(from = "!lhs !rhs")]
    Gt,
    #[display(">=")]
    #[reduce(from = "!lhs !rhs")]
    Ge,

    #[display("uquot")]
    #[reduce(from = "!dividend !divisor")]
    UQuot,
    #[display("urem")]
    #[reduce(from = "!dividend !divisor")]
    URem,
    #[display("u<")]
    #[reduce(from = "!lhs !rhs")]
    ULt,
    #[display("u<=")]
    #[reduce(from = "!lhs !rhs")]
    ULe,
    #[display("u>")]
    #[reduce(from = "!lhs !rhs")]
    UGt,
    #[display("u>=")]
    #[reduce(from = "!lhs !rhs")]
    UGe,

    /// Bitwise NOT, aka `~`.
    #[display("inv")]
    #[reduce(from = "!number")]
    Inv,
    /// Bitwise AND, aka `&`.
    #[display("and")]
    #[reduce(from = "!lhs !rhs")]
    And,
    /// Bitwise OR, aka `|`.
    #[display("or")]
    #[reduce(from = "!lhs !rhs")]
    Or,
    /// Bitwise XOR, aka `^`.
    #[display("xor")]
    #[reduce(from = "!lhs !rhs")]
    Xor,

    /// Shift left.
    #[display("shl")]
    #[reduce(from = "!number !bits")]
    Shl,
    /// (Logical) shift right, i.e., without sign extension.
    #[display("shr")]
    #[reduce(from = "!number !bits")]
    Shr,
    /// Arithmetic shift right, i.e., with sign extension.
    #[display("ashr")]
    #[reduce(from = "!number !bits")]
    AShr,

    /** Characters **/

    /// Convert `Char` to `Int`.
    #[display("ord")]
    #[reduce(from = "!char")]
    Ord,
    /// Convert `Int` to `Char`.
    #[display("chr")]
    #[reduce(from = "!int")]
    Chr,

    /** Floats **/

    /// Reinterpret cast to float.
    #[display("toDbl")]
    #[reduce(from = "!value")]
    ToFloat,

    /// Convert integer to float.
    #[display("itof")]
    #[reduce(from = "!int")]
    IToF,

    #[display("fneg")]
    #[reduce(from = "!float")]
    FNeg,
    #[display("f+")]
    #[reduce(from = "!lhs !rhs")]
    FAdd,
    #[display("f-")]
    #[reduce(from = "!lhs !rhs")]
    FSub,
    #[display("f*")]
    #[reduce(from = "!lhs !rhs")]
    FMul,
    #[display("f/")]
    #[reduce(from = "!dividend !divisor")]
    FDiv,
    #[display("f==")]
    #[reduce(from = "!lhs !rhs")]
    FEq,
    #[display("f/=")]
    #[reduce(from = "!lhs !rhs")]
    FNe,
    #[display("f<")]
    #[reduce(from = "!lhs !rhs")]
    FLt,
    #[display("f<=")]
    #[reduce(from = "!lhs !rhs")]
    FLe,
    #[display("f>")]
    #[reduce(from = "!lhs !rhs")]
    FGt,
    #[display("f>=")]
    #[reduce(from = "!lhs !rhs")]
    FGe,

    /// Convert float value into a Haskell string.
    #[display("fshow")]
    #[reduce(from = "!float")]
    FShow,
    /// Parse float value from a C string.
    #[display("fread")]
    #[reduce(from = "!cstr")]
    FRead,

    /** Pointers **/

    #[display("pnull")]
    #[reduce(constant)]
    PNull,
    #[display("toPtr")]
    #[reduce(from = "!value")]
    ToPtr,
    /// Pointer cast (a nop).
    #[display("pcast")]
    #[reduce(from = "!ptr")]
    PCast,
    /// Pointer equality.
    #[display("p==")]
    #[reduce(from = "!lhs !rhs")]
    PEq,
    /// Pointer addition.
    #[display("p+")]
    #[reduce(from = "!base !offset")]
    PAdd,
    /// Pointer subtraction.
    #[display("p-")]
    #[reduce(from = "!lptr !rptr")]
    PSub,

    /*** IO Monad ***/
    /// Construct a pure computation from a value, in the IO monad.
    ///
    /// Has type `Monad m => a -> m a`
    #[display("IO.return")]
    #[reduce(from = "a")]
    Return,
    /// Monadic bind in the IO monad.
    ///
    /// Has type `Monad m => m a -> (a -> m b) -> m b`.
    #[display("IO.>>=")]
    #[reduce(from = "ma a_mb")]
    Bind,
    /// Kleisli-composition (?) in the IO monad.
    ///
    /// Has type `Monad m => (a -> m b) -> (b -> m c) -> (a -> m c)`.
    ///
    /// Sometimes written `>=>`, and is equivalent to `C' (>>=)`.
    ///
    /// This gets generated to avoid deeply nesting the left-hand side of bind operators, via the
    /// following equivalences:
    ///
    /// ```text
    /// (r >>= s) >>= q     ===     r >>= (\x -> s x >>= q)
    ///                     ===     r >>= (C' (>>=) s q)
    ///                     ===     r >>= (s >=> q)
    /// ```
    #[display("IO.C'BIND")]
    #[reduce(from = "a_mb b_mc a")]
    CCBind,
    /// Sequencing in the IO monad.
    ///
    /// Has type `Monad m => m a -> m b -> m b`.
    #[display("IO.>>")]
    #[reduce(from = "ma mb")]
    Then,
    #[display("IO.performIO")]
    #[reduce(from = "io_term")]
    PerformIO,
    /// Execute a term, and catch any thrown exception.
    #[display("IO.catch")]
    #[reduce(from = "!term handler")]
    Catch,
    /// A dynamic FFI lookup from a Haskell string.
    #[display("dynsym")]
    #[reduce(from = "!name")]
    DynSym,

    #[display("IO.serialize")]
    #[reduce(from = "!file_ptr term")]
    Serialize,
    #[display("IO.deserialize")]
    #[reduce(from = "!file_ptr")]
    Deserialize,
    /// A handle to standard input.
    #[display("IO.stdin")]
    #[reduce(constant)]
    StdIn,
    /// A handle to standard output.
    #[display("IO.stdout")]
    #[reduce(constant)]
    StdOut,
    /// A handle to standard error.
    #[display("IO.stderr")]
    #[reduce(constant)]
    StdErr,
    /// Print a term to a file.
    ///
    /// FIXME: Apparently, the semantics are like `IO.serialize`, except without a header??
    #[display("IO.print")]
    #[reduce(from = "!file_ptr !term")]
    Print,
    #[display("IO.getArgRef")]
    #[reduce(constant)]
    GetArgRef,
    #[display("IO.getTimeMilli")]
    #[reduce(from = "FIXME")] // TODO: how is this evaluated?
    GetTimeMilli,

    /*** Arrays ***/
    /// Allocate an array of the specified size, all initialized to the given element.
    #[display("A.alloc")]
    #[reduce(from = "!size elem")]
    ArrAlloc,
    /// Get the size of an array.
    #[display("A.size")]
    #[reduce(from = "!arr")]
    ArrSize,
    /// Read an element from an array.
    #[display("A.read")]
    #[reduce(from = "!arr !idx")]
    ArrRead,
    /// Write an element to an array.
    #[display("A.write")]
    #[reduce(from = "!arr !idx elem")]
    ArrWrite,
    /// Array equality.
    ///
    /// Apparently only checks for referential equality, i.e., not deep equality.
    #[display("A.==")]
    #[reduce(from = "!lhs !rhs")]
    ArrEq,

    /** C strings **/

    /// Read a Haskell string from a UTF-8-encoded C string.
    ///
    /// TODO: Really ??? This doesn't seem right. How is this non-IO? Figure out what is going on.
    #[display("fromUTF8")]
    #[reduce(from = "!cstr")]
    FromUtf8,

    /// Construct a C ASCII string from a Haskell string.
    ///
    /// Related: <https://downloads.haskell.org/~ghc/5.02.3/docs/set/sec-cstring.html>
    #[display("newCAStringLen")]
    #[reduce(from = "hstr")]
    CAStringLenNew,
    /// Read a Haskell string from a C ASCII string.
    ///
    /// The encoding of the string depends on the encoding of the Haskell implementation.
    ///
    /// Related: <https://downloads.haskell.org/~ghc/5.02.3/docs/set/sec-cstring.html>
    #[display("peekCAString")]
    #[reduce(from = "!cstr")]
    CAStringPeek,
    /// Read a Haskell string from a C ASCII string, up to the given length.
    ///
    /// The encoding of the string depends on the encoding of the Haskell implementation.
    ///
    /// Related: <https://downloads.haskell.org/~ghc/5.02.3/docs/set/sec-cstring.html>
    #[display("peekCAStringLen")]
    #[reduce(from = "!cstr !max_len")]
    CAStringPeekLen,
}

/// The runtime needs to know how core data structures are encoded: booleans, pairs, and lists.
impl Combinator {
    /// Scott-encoded unit `()`.
    pub const UNIT: Self = Self::I;

    /// Scott-encoded `True`.
    pub const TRUE: Self = Self::A;
    /// Scott-encoded `False`.
    pub const FALSE: Self = Self::K;

    /// Scott-encoded pair `(,)`
    pub const PAIR: Self = Self::P;

    /// Scott-encoded list cons `(:)`.
    pub const CONS: Self = Self::O;
    /// Scott-encoded list nil `[]`.
    pub const NIL: Self = Self::K;
}

#[cfg(test)]
mod tests {
    use super::*;
    const PRIMS: &[(Combinator, &str)] = &[
        (Combinator::A, "A"),
        (Combinator::SS, "S'"),
        (Combinator::CCB, "C'B"),
        (Combinator::K2, "K2"),
        (Combinator::ErrorMsg, "error"),
        (Combinator::NoDefault, "noDefault"),
        (Combinator::Add, "+"),
        (Combinator::Neg, "neg"),
        (Combinator::ULe, "u<="),
        (Combinator::ToInt, "toInt"),
        (Combinator::Bind, "IO.>>="),
        (Combinator::Return, "IO.return"),
        (Combinator::StdOut, "IO.stdout"),
        (Combinator::PerformIO, "IO.performIO"),
        (Combinator::CAStringLenNew, "newCAStringLen"),
    ];

    #[test]
    fn display_prims() {
        use alloc::string::ToString;
        for (p, s) in PRIMS {
            assert_eq!(p.to_string(), *s)
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn parse_prims() {
        use core::str::FromStr;
        for (p, s) in PRIMS {
            assert_eq!(Ok(*p), s.parse());
        }
        assert!(Combinator::from_str("u<==").is_err());
        assert!(Combinator::from_str("CC'").is_err());
    }
}

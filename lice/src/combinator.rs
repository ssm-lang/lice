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
    /// How many arguments are needed for reduction.
    fn arity(&self) -> usize;
}

/// Named, primitive values applied to other nodes in a combinator graph.
///
/// TODO: the following symbols are still unknown
///   ==
///   >=
///   >
///   <=
///   <
///   /=
///   ==
///   ord
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
    /*** Turner combinators ***/
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
    #[reduce(from = "x", to = "x self")]
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

    /*** Built-ins ***/
    #[display("error")]
    #[reduce(arity = 1)]
    Error,
    #[display("noDefault")]
    #[reduce(arity = 1)]
    NoDefault,
    #[display("noMatch")]
    #[reduce(arity = 3)]
    NoMatch,
    #[display("seq")]
    #[reduce(arity = 2)]
    Seq,
    #[display("equal")]
    #[reduce(arity = 2)]
    Equal,
    /// String equality; same implementation as Equal
    #[display("sequal")]
    #[reduce(arity = 2)]
    SEqual,
    #[display("compare")]
    #[reduce(arity = 2)]
    Compare,
    /// String comparison; same implementation as Compare
    #[display("scmp")]
    #[reduce(arity = 2)]
    SCmp,
    /// Integer comparison; same implementation as Compare
    #[display("icmp")]
    #[reduce(arity = 2)]
    ICmp,
    #[display("rnf")]
    #[reduce(arity = 2)]
    Rnf,
    /// Read file contents as UTF-8 string (?)
    #[display("fromUTF8")]
    #[reduce(arity = 1)]
    FromUtf8,
    /// Convert `Int` to `Char`.
    #[display("chr")]
    #[reduce(arity = 1)]
    Chr,

    /*** Integer arithmetic ***/
    #[display("+")]
    #[reduce(arity = 2)]
    Add,
    #[display("-")]
    #[reduce(arity = 2)]
    Sub,
    #[display("*")]
    #[reduce(arity = 2)]
    Mul,
    #[display("quot")]
    #[reduce(arity = 2)]
    Quot,
    #[display("rem")]
    #[reduce(arity = 2)]
    Rem,
    /// The same as `flip (-)`.
    #[display("subtract")]
    #[reduce(arity = 2)]
    Subtract,
    #[display("uquot")]
    #[reduce(arity = 2)]
    UQuot,
    #[display("urem")]
    #[reduce(arity = 2)]
    URem,
    #[display("neg")]
    #[reduce(arity = 1)]
    Neg,
    #[display("and")]
    #[reduce(arity = 2)]
    And,
    #[display("or")]
    #[reduce(arity = 2)]
    Or,
    #[display("xor")]
    #[reduce(arity = 2)]
    Xor,
    #[display("inv")]
    #[reduce(arity = 1)]
    Inv,
    #[display("shl")]
    #[reduce(arity = 2)]
    Shl,
    #[display("shr")]
    #[reduce(arity = 2)]
    Shr,
    #[display("ashr")]
    #[reduce(arity = 2)]
    AShr,
    #[display("eq")]
    #[reduce(arity = 2)]
    Eq,
    #[display("ne")]
    #[reduce(arity = 2)]
    Ne,
    #[display("lt")]
    #[reduce(arity = 2)]
    Lt,
    #[display("le")]
    #[reduce(arity = 2)]
    Le,
    #[display("gt")]
    #[reduce(arity = 2)]
    Gt,
    #[display("ge")]
    #[reduce(arity = 2)]
    Ge,
    #[display("u<")]
    #[reduce(arity = 2)]
    ULt,
    #[display("u<=")]
    #[reduce(arity = 2)]
    ULe,
    #[display("u>")]
    #[reduce(arity = 2)]
    UGt,
    #[display("u>=")]
    #[reduce(arity = 2)]
    UGe,
    #[display("toInt")]
    #[reduce(arity = 1)]
    ToInt,

    /*** Pointers ***/
    #[display("p==")]
    #[reduce(arity = 2)]
    PEq,
    #[display("pnull")]
    #[reduce(arity = 0)]
    PNull,
    #[display("p+")]
    #[reduce(arity = 2)]
    PAdd,
    #[display("p=")]
    #[reduce(arity = 2)]
    PSub,
    #[display("toPtr")]
    #[reduce(arity = 1)]
    ToPtr,

    /*** IO Monad ***/
    #[display("IO.>>=")]
    #[reduce(arity = 2)]
    Bind,
    #[display("IO.>>")]
    #[reduce(arity = 2)]
    Then,
    #[display("IO.return")]
    #[reduce(arity = 1)]
    Return,
    #[display("IO.serialize")]
    #[reduce(arity = 2)]
    Serialize,
    #[display("IO.deserialize")]
    #[reduce(arity = 1)]
    Deserialize,
    /// A handle to standard input.
    #[display("IO.stdin")]
    #[reduce(arity = 0)]
    StdIn,
    /// A handle to standard output.
    #[display("IO.stdout")]
    #[reduce(arity = 0)]
    StdOut,
    /// A handle to standard error.
    #[display("IO.stderr")]
    #[reduce(arity = 0)]
    StdErr,
    #[display("IO.getArgs")]
    #[reduce(arity = 0)]
    GetArgs,
    #[display("IO.performIO")]
    #[reduce(arity = 1)]
    PerformIO,
    #[display("IO.getTimeMilli")]
    #[reduce(arity = 0)] // FIXME: how is this evaluated?
    GetTimeMilli,
    #[display("IO.print")]
    #[reduce(arity = 2)]
    Print,
    #[display("IO.catch")]
    #[reduce(arity = 2)]
    Catch,
    #[display("dynsym")]
    #[reduce(arity = 1)]
    DynSym,

    /*** Floating point arithmetic ***/
    #[display("f+")]
    #[reduce(arity = 2)]
    FAdd,
    #[display("f-")]
    #[reduce(arity = 2)]
    FSub,
    #[display("f*")]
    #[reduce(arity = 2)]
    FMul,
    #[display("f/")]
    #[reduce(arity = 2)]
    FDiv,
    #[display("fneg")]
    #[reduce(arity = 2)]
    FNeg,
    /// Integer to floating point conversion.
    #[display("itof")]
    #[reduce(arity = 1)]
    IToF,
    #[display("f==")]
    #[reduce(arity = 2)]
    FEq,
    #[display("f/=")]
    #[reduce(arity = 2)]
    FNe,
    #[display("f<")]
    #[reduce(arity = 2)]
    FLt,
    #[display("f<=")]
    #[reduce(arity = 2)]
    FLe,
    #[display("f>")]
    #[reduce(arity = 2)]
    FGt,
    #[display("f>=")]
    #[reduce(arity = 2)]
    FGe,
    #[display("fshow")]
    #[reduce(arity = 1)]
    FShow,
    #[display("fread")]
    #[reduce(arity = 1)]
    FRead,

    /*** Arrays ***/
    #[display("A.alloc")]
    #[reduce(arity = 2)]
    Alloc,
    #[display("A.size")]
    #[reduce(arity = 1)]
    Size,
    #[display("A.read")]
    #[reduce(arity = 2)]
    Read,
    #[display("A.write")]
    #[reduce(arity = 3)]
    Write,
    #[display("A.==")]
    #[reduce(arity = 2)]
    CAEq,
    #[display("newCAStringLen")]
    #[reduce(arity = 1)]
    NewCAStringLen,
    #[display("peekCAString")]
    #[reduce(arity = 1)]
    PeekCAString,
    #[display("peekCAStringLen")]
    #[reduce(arity = 2)]
    PeekCAStringLen,
}

#[cfg(test)]
mod tests {
    use super::*;
    const PRIMS: &[(Combinator, &str)] = &[
        (Combinator::A, "A"),
        (Combinator::SS, "S'"),
        (Combinator::CCB, "C'B"),
        (Combinator::K2, "K2"),
        (Combinator::Error, "error"),
        (Combinator::NoDefault, "noDefault"),
        (Combinator::Add, "+"),
        (Combinator::Neg, "neg"),
        (Combinator::ULe, "u<="),
        (Combinator::ToInt, "toInt"),
        (Combinator::Bind, "IO.>>="),
        (Combinator::Return, "IO.return"),
        (Combinator::StdOut, "IO.stdout"),
        (Combinator::PerformIO, "IO.performIO"),
        (Combinator::NewCAStringLen, "newCAStringLen"),
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

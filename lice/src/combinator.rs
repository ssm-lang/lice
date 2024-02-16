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
    fn arity(&self) -> Option<usize>;
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
    Error,
    #[display("noDefault")]
    NoDefault,
    #[display("noMatch")]
    NoMatch,
    #[display("seq")]
    Seq,
    #[display("equal")]
    Equal,
    #[display("sequal")]
    SEqual,
    #[display("compare")]
    Compare,
    #[display("scmp")]
    SCmp,
    #[display("icmp")]
    ICmp,
    #[display("rnf")]
    Rnf,
    #[display("fromUTF8")]
    FromUtf8,
    #[display("chr")]
    Chr,

    /*** Integer arithmetic ***/
    #[display("+")]
    Add,
    #[display("-")]
    Sub,
    #[display("*")]
    Mul,
    #[display("quot")]
    Quot,
    #[display("rem")]
    Rem,
    #[display("subtract")]
    Subtract,
    #[display("uquot")]
    UQuot,
    #[display("urem")]
    URem,
    #[display("neg")]
    Neg,
    #[display("and")]
    And,
    #[display("or")]
    Or,
    #[display("xor")]
    Xor,
    #[display("inv")]
    Inv,
    #[display("shl")]
    Shl,
    #[display("shr")]
    Shr,
    #[display("ashr")]
    AShr,
    #[display("eq")]
    Eq,
    #[display("ne")]
    Ne,
    #[display("lt")]
    Lt,
    #[display("le")]
    Le,
    #[display("gt")]
    Gt,
    #[display("ge")]
    Ge,
    #[display("u<")]
    ULt,
    #[display("u<=")]
    ULe,
    #[display("u>")]
    UGt,
    #[display("u>=")]
    UGe,
    #[display("toInt")]
    ToInt,

    /*** Pointers ***/
    #[display("p==")]
    PEq,
    #[display("pnull")]
    PNull,
    #[display("p+")]
    PAdd,
    #[display("p=")]
    PSub,
    #[display("toPtr")]
    ToPtr,

    /*** IO Monad ***/
    #[display("IO.>>=")]
    Bind,
    #[display("IO.>>")]
    Then,
    #[display("IO.return")]
    Return,
    #[display("IO.serialize")]
    Serialize,
    #[display("IO.deserialize")]
    Deserialize,
    #[display("IO.stdin")]
    StdIn,
    #[display("IO.stdout")]
    StdOut,
    #[display("IO.stderr")]
    StdErr,
    #[display("IO.getArgs")]
    GetArgs,
    #[display("IO.performIO")]
    PerformIO,
    #[display("IO.getTimeMilli")]
    GetTimeMilli,
    #[display("IO.print")]
    Print,
    #[display("IO.catch")]
    Catch,
    #[display("dynsym")]
    DynSym,

    /*** Floating point arithmetic ***/
    #[display("f+")]
    FAdd,
    #[display("f-")]
    FSub,
    #[display("f*")]
    FMul,
    #[display("f/")]
    FDiv,
    #[display("fneg")]
    FNeg,
    /// Integer to floating point conversion.
    #[display("itof")]
    IToF,
    #[display("f==")]
    Feq,
    #[display("f/=")]
    FNe,
    #[display("f<")]
    FLt,
    #[display("f<=")]
    FLe,
    #[display("f>")]
    FGt,
    #[display("f>=")]
    FGe,
    #[display("fshow")]
    FShow,
    #[display("fread")]
    FRead,

    /*** Arrays ***/
    #[display("A.alloc")]
    Alloc,
    #[display("A.size")]
    Size,
    #[display("A.read")]
    Read,
    #[display("A.write")]
    Write,
    #[display("A.==")]
    CAEq,
    #[display("newCAStringLen")]
    NewCAStringLen,
    #[display("peekCAString")]
    PeekCAString,
    #[display("peekCAStringLen")]
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

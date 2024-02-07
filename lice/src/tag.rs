//! Combinator tags
use lice_macros::Combinator;
use parse_display::Display;
#[cfg(feature = "std")]
use parse_display::FromStr;
pub use parse_display::ParseError;

/// Types of primitive values (leaf nodes) that can appear in the combinator graph.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
#[cfg_attr(feature = "std", derive(FromStr))]
#[display("{0}")]
pub enum Tag {
    Combinator(Turner),
    BuiltIn(BuiltIn),
    Arith(Arith),
    Pointer(Pointer),
    IO(IO),
    FArith(FArith),
    Array(Array),
}

/// Properties of combinators (which may be partially evaluated at compile time).
///
/// The derive macro for `Combinator` uses the `#[rule(...)]` helper attribute to recieve
/// a specification of the reduction rule, i.e., the redex and reduct. At the moment, the macro
/// doesn't do anything with that information other than compute the arity (number of redexes).
pub trait Combinator {
    /// How many arguments are needed for reduction.
    fn arity(&self) -> usize;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, Combinator)]
#[cfg_attr(feature = "std", derive(FromStr))]
pub enum Turner {
    #[rule(from = "f g x", to = "f x (g x)")]
    S,

    #[rule(from = "x y", to = "x")]
    K,

    #[rule(from = "x", to = "x")]
    I,

    #[rule(from = "f g x", to = "f (g x)")]
    B,

    #[rule(from = "f x y", to = "f y x")]
    C,

    #[rule(from = "x y", to = "y")]
    A,

    #[rule(from = "x", to = "x self")]
    Y,

    #[display("S'")]
    #[rule(from = "c f g x", to = "c (f x) (g x)")]
    SS,

    #[display("B'")]
    #[rule(from = "c f g x", to = "c f (g x)")]
    BB,

    #[display("C'")]
    #[rule(from = "c f x y", to = "c (f y) x")]
    CC,

    #[rule(from = "x y f", to = "f x y")]
    P,

    #[rule(from = "y f x", to = "f x y")]
    R,

    #[rule(from = "x y z f", to = "f x y")]
    O,

    #[rule(from = "x f", to = "f x")]
    U,

    #[rule(from = "f x y", to = "f x")]
    Z,

    #[rule(from = "x y z", to = "x")]
    K2,

    #[rule(from = "x y z w", to = "x")]
    K3,

    #[rule(from = "x y z w v", to = "x")]
    K4,

    #[display("C'B")]
    #[rule(from = "f g x y", to = "f x (g y)")]
    CCB,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
#[cfg_attr(feature = "std", derive(FromStr))]
pub enum BuiltIn {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
#[cfg_attr(feature = "std", derive(FromStr))]
pub enum Arith {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
#[cfg_attr(feature = "std", derive(FromStr))]
pub enum Pointer {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
#[cfg_attr(feature = "std", derive(FromStr))]
pub enum IO {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
#[cfg_attr(feature = "std", derive(FromStr))]
pub enum FArith {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
#[cfg_attr(feature = "std", derive(FromStr))]
pub enum Array {
    #[display("A.alloc")]
    Alloc,
    #[display("A.size")]
    Size,
    #[display("A.read")]
    Read,
    #[display("A.write")]
    Write,
    #[display("A.==")]
    Eq,
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

    macro_rules! Prim {
        (Turner::$id:ident) => {
            Tag::Combinator(Turner::$id)
        };
        ($kind:ident::$id:ident) => {
            Tag::$kind($kind::$id)
        };
    }

    const PRIMS: &[(Tag, &str)] = &[
        (Prim!(Turner::A), "A"),
        (Prim!(Turner::SS), "S'"),
        (Prim!(Turner::CCB), "C'B"),
        (Prim!(Turner::K2), "K2"),
        (Prim!(BuiltIn::Error), "error"),
        (Prim!(BuiltIn::NoDefault), "noDefault"),
        (Prim!(Arith::Add), "+"),
        (Prim!(Arith::Neg), "neg"),
        (Prim!(Arith::ULe), "u<="),
        (Prim!(Arith::ToInt), "toInt"),
        (Prim!(IO::Bind), "IO.>>="),
        (Prim!(IO::Return), "IO.return"),
        (Prim!(IO::StdOut), "IO.stdout"),
        (Prim!(IO::PerformIO), "IO.performIO"),
        (Prim!(Array::NewCAStringLen), "newCAStringLen"),
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

        assert!(Turner::from_str("u<=").is_err());
        assert!(Arith::from_str("C'B").is_err());
    }
}

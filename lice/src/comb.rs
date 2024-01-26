//! Combinators
use lice_macros::Combinator;
use parse_display::{Display, FromStr};

pub use parse_display::ParseError;

/// Vector indices in the [`Program`] `body`.
pub type Index = usize;

/// Vector indices in the [`Program`] `defs` list.
pub type Label = usize;

/// Representation of integer literals that appear in the program code.
pub type Int = i64;

/// Representation of float literals that appear in the program code.
pub type Float = f64;

pub(crate) const NIL_INDEX: Index = Index::MAX;

/// Representation for the `.comb` file format.
///
/// Currently we do not do anything to validate the version (though we probably should).
#[derive(Debug, Clone)]
pub struct CombFile {
    // (Major, minor)
    pub version: (usize, usize),

    /// (Maximum) number of definitions.
    ///
    /// This should be the same as `.program.defs.len()`.
    pub size: usize,

    /// Program embedded in the comb file.
    pub program: Program,
}

/// Representation for the program in a `.comb` file.
///
/// Instead of a tree, this representation stores the body of [`Expr`]s in a single [`Vec`], with
/// "pointers" implemented using offsets into that body. Doing so keeps the representation
/// self-contained and easily serializable.
#[derive(Debug, Clone)]
pub struct Program {
    // The root combinator expression in `body`.
    pub root: Index,

    /// Pool of program AST nodes.
    ///
    /// Logically, this vector is a dense map from [`Index`] to [`Expr`].
    pub body: Vec<Expr>,

    /// List of definitions that indirect nodes ([`Expr::Ref`]) can reference.
    ///
    /// Logically, this vector is a dense map from [`Label`] to [`Index`].
    pub defs: Vec<Index>,
}

#[derive(Debug, Clone, Display)]
pub enum Expr {
    /// Application of two expressions, with possible definition label: i.e., `(func [:label] arg)`.
    #[display("@")]
    App(Index, Option<Label>, Index),
    /// Floating point literal, i.e., `&float`.
    #[display("&{0}")]
    Float(Float),
    /// Integer literal, possibly negative, i.e., `#[-]int`.
    #[display("#{0}")]
    Int(Int),
    /// Fixed size array of expressions, i.e., `[size arr]`.
    #[display("[{0}]")]
    Array(usize, Vec<Index>),
    /// Reference to some labeled definition, i.e., `_label`.
    #[display("*")]
    Ref(Label),
    /// String literal, i.e., `"str"`
    #[display("{0:#?}")]
    String(String),
    /// Tick mark, i.e., `!"tick"`.
    #[display("!{0:#?}")]
    Tick(String),
    /// FFI symbol, i.e., `^symbol`.
    #[display("^{0}")]
    Ffi(String),
    /// Combinators and other primitives, e.g., `S` or `IO.>>=`.
    #[display("{0}")]
    Prim(Prim),
    /// Default case. Shouldn't appear, but you know, life happens.
    #[display("?!{0}")]
    Unknown(String),
}

impl Expr {
    pub(crate) fn new_app() -> Self {
        Self::App(NIL_INDEX, None, NIL_INDEX)
    }

    pub(crate) fn new_ref() -> Self {
        Self::Ref(NIL_INDEX)
    }
}

/// Types of primitive values (leaf nodes) that can appear in the combinator graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, FromStr)]
#[display("{0}")]
pub enum Prim {
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

#[derive(Combinator, Debug, Clone, Copy, PartialEq, Eq, Display, FromStr)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, FromStr)]
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, FromStr)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, FromStr)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, FromStr)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, FromStr)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, FromStr)]
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

impl std::fmt::Display for CombFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "v{}.{}\n{}\n{}",
            self.version.0, self.version.1, self.size, self.program
        )
    }
}

impl std::fmt::Display for Program {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.fmt_expr(out, self.root)
    }
}

impl Program {
    /// Recursive helper for [`Program`] to implement [`std::fmt::Display`].
    fn fmt_expr(&self, out: &mut std::fmt::Formatter<'_>, idx: Index) -> std::fmt::Result {
        match self.body.get(idx).ok_or(std::fmt::Error)? {
            Expr::App(f, l, a) => {
                write!(out, "(")?;
                self.fmt_expr(out, *f)?;
                write!(out, " ")?;
                if let Some(l) = l {
                    write!(out, ":{l} ")?;
                }
                self.fmt_expr(out, *a)?;
                write!(out, ")")
            }
            Expr::Array(sz, arr) => {
                // assert!(sz == arr.len());
                write!(out, "[{sz}")?;
                for a in arr {
                    write!(out, " ")?;
                    self.fmt_expr(out, *a)?;
                }
                write!(out, "]")
            }
            Expr::String(s) => {
                write!(out, "\"")?;
                for c in s.chars() {
                    if c.is_ascii_graphic() || c == ' ' {
                        write!(out, "{c}")?;
                    } else {
                        write!(out, "\\{}", c as usize)?;
                    }
                }
                write!(out, "\"")
            }
            Expr::Tick(s) => {
                write!(out, "!\"")?;
                for c in s.chars() {
                    if c.is_ascii_graphic() || c == ' ' {
                        write!(out, "{c}")?;
                    } else {
                        write!(out, "\\{}", c as usize)?;
                    }
                }
                write!(out, "\"")
            }
            expr => {
                // `Expr's derived Display implementation is sufficient
                write!(out, "{}", expr)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;

    #[test]
    fn display_prims() {
        // Spot-check some random primitives
        assert_eq!(Turner::A.to_string(), "A");
        assert_eq!(Turner::SS.to_string(), "S'");
        assert_eq!(Turner::CCB.to_string(), "C'B");
        assert_eq!(Turner::K2.to_string(), "K2");

        assert_eq!(Prim::Combinator(Turner::A).to_string(), "A");
        assert_eq!(Prim::Combinator(Turner::SS).to_string(), "S'");
        assert_eq!(Prim::Combinator(Turner::CCB).to_string(), "C'B");
        assert_eq!(Prim::Combinator(Turner::K2).to_string(), "K2");

        assert_eq!(BuiltIn::Error.to_string(), "error");
        assert_eq!(BuiltIn::NoDefault.to_string(), "noDefault");

        assert_eq!(Arith::Add.to_string(), "+");
        assert_eq!(Arith::Neg.to_string(), "neg");
        assert_eq!(Arith::ULe.to_string(), "u<=");
        assert_eq!(Arith::ToInt.to_string(), "toInt");

        assert_eq!(IO::Bind.to_string(), "IO.>>=");
        assert_eq!(IO::Return.to_string(), "IO.return");
        assert_eq!(IO::StdOut.to_string(), "IO.stdout");
        assert_eq!(IO::PerformIO.to_string(), "IO.performIO");

        assert_eq!(Array::NewCAStringLen.to_string(), "newCAStringLen");
    }

    #[test]
    fn parse_prims() {
        assert_eq!(Ok(Turner::A), "A".parse());
        assert_eq!(Ok(Turner::SS), "S'".parse());
        assert_eq!(Ok(Turner::CCB), "C'B".parse());
        assert_eq!(Ok(Turner::K2), "K2".parse());

        assert_eq!(Ok(Prim::Combinator(Turner::A)), "A".parse());
        assert_eq!(Ok(Prim::Combinator(Turner::SS)), "S'".parse());
        assert_eq!(Ok(Prim::Combinator(Turner::CCB)), "C'B".parse());
        assert_eq!(Ok(Prim::Combinator(Turner::K2)), "K2".parse());

        assert_eq!(Ok(Arith::Add), "+".parse());
        assert_eq!(Ok(Arith::Neg), "neg".parse());
        assert_eq!(Ok(Arith::ULe), "u<=".parse());
        assert_eq!(Ok(Arith::ToInt), "toInt".parse());

        assert!(Turner::from_str("u<=").is_err());
        assert!(Arith::from_str("C'B").is_err());
    }

    #[test]
    fn display_program() {
        // An arbitrarily constructed test case, deliberately featuring:
        // - at least one of each type of expr
        // - a root that doesn't have the last index
        // - negative floating and integer literals
        // - two app exprs that point to the same expr (without indirection)
        // - an otherwise confounding tree structure
        let p = CombFile {
            version: (6, 19),
            size: 1,
            program: Program {
                root: 10,
                body: vec![
                    /* 0 */ Expr::Prim(Prim::Combinator(Turner::K4)),
                    /* 1 */ Expr::Prim(Prim::Combinator(Turner::CCB)),
                    /* 2 */ Expr::Prim(Prim::IO(IO::Bind)),
                    /* 3 */ Expr::Int(-42),
                    /* 4 */ Expr::Float(-4.2),
                    /* 5 */ Expr::String("Hello world!\r\n".to_string()),
                    /* 6 */ Expr::Tick("Lyme's".to_string()),
                    /* 7 */ Expr::Ffi("fork".to_string()),
                    /* 8 */ Expr::Array(5, vec![3, 4, 5, 6, 7]),
                    /* 9 */ Expr::Ffi("UNREACHABLE!".to_string()),
                    /* 10 */ Expr::App(2, Some(0), 13),
                    /* 11 */ Expr::App(10, None, 14),
                    /* 12 */ Expr::App(1, None, 2),
                    /* 13 */ Expr::App(8, None, 0),
                    /* 14 */ Expr::App(12, None, 15),
                    /* 15 */ Expr::Ref(0),
                ],
                defs: vec![],
            },
        };

        assert_eq!(
            p.to_string(),
            r#"v6.19
1
(IO.>>= :0 ([5 #-42 &-4.2 "Hello world!\13\10" !"Lyme's" ^fork] K4))"#
        );
    }
}

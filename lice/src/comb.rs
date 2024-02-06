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
    App(Index, Index),
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
        Self::App(NIL_INDEX, NIL_INDEX)
    }

    pub(crate) fn new_ref() -> Self {
        Self::Ref(NIL_INDEX)
    }
}

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
    #[display("fromUTF8")]
    FromUtf8,
    #[display("chr")]
    Chr,
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

impl Program {
    fn count_refs(&self, refs: &mut Vec<usize>, i: Index) {
        refs[i] += 1;
        if refs[i] > 1 {
            return;
        }
        // TODO: Broken
        match &self.body[i] {
            Expr::App(f, a) => {
                let (f, a) = (*f, *a);
                self.count_refs(refs, f);
                self.count_refs(refs, a);
            }
            Expr::Ref(l) => {
                let r = self.defs[*l];
                self.count_refs(refs, r);
            }
            Expr::Array(_sz, arr) => {
                for &a in arr {
                    self.count_refs(refs, a);
                }
            }
            _ => (),
        }
    }

    fn fmt_rpn(
        &self,
        out: &mut std::fmt::Formatter<'_>,
        refs: &Vec<usize>,
        labels: &mut Vec<Option<Label>>,
        next_label: &mut Label,
        i: Index,
    ) -> std::fmt::Result {
        let mut ii = i;

        // Follow indirections (we will be printing our own anyway)
        while let Expr::Ref(l) = &self.body[ii] {
            ii = self.defs[*l];
            if i == ii {
                // std::fmt::Error doesn't support error messages, so we just log::error!() here
                log::error!("cyclic reference encountered: {i}");
                return Err(std::fmt::Error);
            }
        }

        let i = ii;

        if refs[i] > 1 {
            // The node is shared
            if let Some(label) = labels[i] {
                // This node has already been printed, so just use a reference
                return out.write_fmt(format_args!("_{label}"));
            } else {
                // Not yet printed, so allocate a label
                labels[i] = Some(*next_label);
                *next_label += 1;
            }
        }

        match &self.body[i] {
            Expr::Ref(_) => unreachable!("indirections should have been followed"),
            Expr::App(f, a) => {
                self.fmt_rpn(out, refs, labels, next_label, *f)?;
                out.write_str(" ")?;
                self.fmt_rpn(out, refs, labels, next_label, *a)?;
                out.write_str(" @")?;
            }
            Expr::Array(_sz, arr) => {
                for a in arr {
                    self.fmt_rpn(out, refs, labels, next_label, *a)?;
                    out.write_str(" ")?;
                }
                out.write_fmt(format_args!("[{}]", arr.len()))?;
            }
            e => {
                out.write_fmt(format_args!("{e}"))?;
            }
        }

        if refs[i] > 1 {
            if let Some(label) = labels[i] {
                // We had previously allocated a label for this node; print it out now, post-fix
                out.write_fmt(format_args!(" :{label}"))?;
            }
        }
        if i == self.root {
            out.write_str(" }")?;
        }
        Ok(())
    }
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
        let mut refs = [0].repeat(self.body.len());
        self.count_refs(&mut refs, self.root);
        println!("refs: {:#?}", &refs);
        let mut visited = [None].repeat(self.body.len());
        self.fmt_rpn(out, &refs, &mut visited, &mut 0, self.root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    macro_rules! Prim {
        (Turner::$id:ident) => {
            Prim::Combinator(Turner::$id)
        };
        ($kind:ident::$id:ident) => {
            Prim::$kind($kind::$id)
        };
    }
    const PRIMS: &[(Prim, &str)] = &[
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
        for (p, s) in PRIMS {
            assert_eq!(p.to_string(), *s)
        }
    }

    #[test]
    fn parse_prims() {
        for (p, s) in PRIMS {
            assert_eq!(Ok(*p), s.parse());
        }

        assert!(Turner::from_str("u<=").is_err());
        assert!(Arith::from_str("C'B").is_err());
    }

    /// Ints and floats
    #[test]
    fn display_nums() {
        assert_eq!(Expr::Int(1).to_string(), "#1");
        assert_eq!(Expr::Int(256).to_string(), "#256");
        assert_eq!(Expr::Int(-42).to_string(), "#-42");

        assert_eq!(Expr::Float(3.12).to_string(), "&3.12");
        assert_eq!(Expr::Float(0.01).to_string(), "&0.01");
        assert_eq!(Expr::Float(3.0).to_string(), "&3");
        assert_eq!(Expr::Float(-50.0).to_string(), "&-50");
        assert_eq!(Expr::Float(0.0).to_string(), "&0");
    }

    /// Ints and floats
    #[test]
    fn parse_nums() {}

    /// Strings, ticks, and FFI
    #[test]
    fn display_strings() {}

    // Programs that are trivial trees
    #[test]
    fn display_basic_programs() {
        let mut p = Program {
            root: 0,
            body: vec![
                /* 0 */ Expr::App(1, 2),
                /* 1 */ Expr::App(3, 4),
                /* 2 */ Expr::Int(1),
                /* 3 */ Expr::Prim(Prim::Combinator(Turner::A)),
                /* 4 */ Expr::Prim(Prim::Combinator(Turner::B)),
                /* 5 */ Expr::Prim(Prim::Combinator(Turner::I)), // unreachable
            ],
            defs: vec![],
        };
        assert_eq!(p.to_string(), "A B @ #1 @ }");

        p.body[3] = Expr::Prim(Prim::Combinator(Turner::K4));
        p.body[4] = Expr::Prim(Prim::Combinator(Turner::SS));
        p.body[5] = p.body[0].clone();
        p.root = 5;
        assert_eq!(p.to_string(), "K4 S' @ #1 @ }");
    }

    // Programs that have simple references, but are otherwise well-structured
    #[test]
    fn display_ref_programs() {
        let p = Program {
            root: 0,
            body: vec![
                /* 0 */ Expr::App(2, 1),
                /* 1 */ Expr::Ref(0),
                /* 2 */ Expr::App(3, 4),
                /* 3 */ Expr::Prim(Prim!(Turner::A)),
                /* 4 */ Expr::Prim(Prim!(Turner::B)),
            ],
            defs: vec![4],
        };
        assert_eq!(p.to_string(), "A B :0 @ _0 @ }");

        let p = Program {
            root: 0,
            body: vec![
                /* 0 */ Expr::App(2, 1),
                /* 1 */ Expr::Ref(0),
                /* 2 */ Expr::App(3, 4),
                /* 3 */ Expr::Prim(Prim!(Turner::A)),
                /* 4 */ Expr::Prim(Prim!(Turner::B)),
            ],
            defs: vec![2],
        };
        assert_eq!(p.to_string(), "A B @ :0 _0 @ }");

        let p = Program {
            root: 0,
            body: vec![
                /* 0 */ Expr::App(1, 2),
                /* 1 */ Expr::App(3, 4),
                /* 2 */ Expr::Ref(1),
                /* 3 */ Expr::App(6, 5),
                /* 4 */ Expr::Ref(0),
                /* 5 */ Expr::Prim(Prim!(Turner::B)),
                /* 6 */ Expr::Prim(Prim!(Turner::A)),
            ],
            defs: vec![/* 0 */ 6, /* 1 */ 5],
        };
        assert_eq!(p.to_string(), "A :0 B :1 @ _0 @ _1 @ }");
    }

    // Programs that have multiple references to the same node, via Ref or App.
    #[test]
    fn display_acylic_programs() {
        // Multiple refs to the same node
        let p = Program {
            root: 0,
            body: vec![
                /* 0 */ Expr::App(1, 2),
                /* 1 */ Expr::App(3, 4),
                /* 2 */ Expr::Ref(0),
                /* 3 */ Expr::App(6, 5),
                /* 4 */ Expr::Ref(0),
                /* 5 */ Expr::Prim(Prim!(Turner::B)),
                /* 6 */ Expr::Prim(Prim!(Turner::A)),
            ],
            defs: vec![/* 0 */ 6],
        };
        assert_eq!(p.to_string(), "A :0 B @ _0 @ _0 @ }");

        // Refs and apps all pointing to the same node
        let p = Program {
            root: 0,
            body: vec![
                /* 0 */ Expr::App(1, 5),
                /* 1 */ Expr::App(2, 3),
                /* 2 */ Expr::App(5, 4),
                /* 3 */ Expr::Ref(0),
                /* 4 */ Expr::Prim(Prim!(Turner::B)),
                /* 5 */ Expr::Prim(Prim!(Turner::A)),
            ],
            defs: vec![/* 0 */ 5],
        };
        assert_eq!(p.to_string(), "A :0 B @ _0 @ _0 @ }");

        // Refs with garbage
        let p = Program {
            root: 0,
            body: vec![
                /* 0 */ Expr::App(1, 5),
                /* 1 */ Expr::App(3, 5),
                /* _ */ Expr::App(4, 2), // garbage
                /* 3 */ Expr::Ref(0),
                /* 4 */ Expr::Prim(Prim!(Turner::B)),
                /* 5 */ Expr::Prim(Prim!(Turner::A)),
            ],
            defs: vec![/* 0 */ 5],
        };
        assert_eq!(p.to_string(), "A :0 _0 @ _0 @ }");

        // No refs, just apps pointing to the same node. A def should be generated
        let p = Program {
            root: 0,
            body: vec![
                /* 0 */ Expr::App(1, 4),
                /* 1 */ Expr::App(2, 4),
                /* 2 */ Expr::App(4, 3),
                /* 3 */ Expr::Prim(Prim!(Turner::B)),
                /* 4 */ Expr::Prim(Prim!(Turner::A)),
            ],
            defs: vec![],
        };
        assert_eq!(p.to_string(), "A :0 B @ _0 @ _0 @ }");

        // TODO: ref to ref
    }

    // Programs with cycles
    #[test]
    fn display_cyclic_programs() {
        // Cycle with no refs
        let p = Program {
            root: 0,
            body: vec![Expr::App(0, 0)],
            defs: vec![],
        };
        assert_eq!(p.to_string(), "_0 _0 @ :0 }");

        // Cycle without ref
        let p = Program {
            root: 0,
            body: vec![Expr::App(0, 1), Expr::Prim(Prim!(Turner::I))],
            defs: vec![],
        };
        assert_eq!(p.to_string(), "_0 I @ :0 }");

        // Cycle with ref
        let p = Program {
            root: 0,
            body: vec![Expr::App(1, 2), Expr::Ref(0), Expr::Prim(Prim!(Turner::I))],
            defs: vec![0],
        };
        assert_eq!(p.to_string(), "_0 I @ :0 }");
    }
}

//! Combinator file parser.
use crate::combinator::Combinator;
use anyhow::{anyhow, bail, ensure, Result};
use nom::{
    branch::alt,
    bytes::complete::{is_not, take_till, take_till1},
    character::{
        complete::{char, digit1, multispace0},
        is_space,
    },
    combinator::{map_res, opt, recognize, verify},
    multi::fold_many0,
    number::complete::double,
    sequence::{delimited, preceded, separated_pair},
    IResult, Parser,
};
use parse_display::Display;
use std::str::FromStr;

/// Vector indices in the [`Program`] `body`.
pub type Index = usize;

/// Vector indices in the [`Program`] `defs` list.
pub type Label = usize;

/// Representation of integer literals that appear in the program code.
pub type Int = i64;

/// Representation of float literals that appear in the program code.
pub type Float = f64;

pub(crate) const NIL_INDEX: Index = Index::MAX;
pub(crate) const NIL_LABEL: Label = Label::MAX;

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
    Prim(Combinator),
    /// Default case. Shouldn't appear, but you know, life happens.
    #[display("?!{0}")]
    Unknown(String),
}

impl Program {
    fn count_refs(&self, refs: &mut Vec<usize>, i: Index) {
        refs[i] += 1;
        if refs[i] > 1 {
            return;
        }
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
        let mut visited = [None].repeat(self.body.len());
        self.fmt_rpn(out, &refs, &mut visited, &mut 0, self.root)
    }
}

/// The result of parsing. A parser monad, even. Typedef'd here for convenience.
type Parse<'a, T> = IResult<&'a str, T>;

/// Adapt `nom` errors into `anyhow` errors.
fn to_anyhow(e: nom::Err<nom::error::Error<&str>>) -> anyhow::Error {
    anyhow!("{e}")
}

/// Parse an unsigned integer literal.
fn uinteger(i: &str) -> Parse<usize> {
    map_res(digit1, |s: &str| s.parse::<usize>()).parse(i)
}

/// Parse a possibly signed integer literal.
fn integer(i: &str) -> Parse<i64> {
    map_res(recognize(preceded(opt(char('-')), digit1)), |s: &str| {
        s.parse::<i64>()
    })
    .parse(i)
}

/// Parse a string literal.
fn string_literal(input: &str) -> Parse<String> {
    enum StringFragment<'a> {
        Literal(&'a str),
        EscapedChar(char),
    }

    let literal = verify(is_not("\"\\"), |s: &str| !s.is_empty());
    let escaped_char = preceded(
        char('\\'),
        map_res(digit1, |s: &str| s.parse::<u8>()).map(|n| n as char),
    );

    let build_string = fold_many0(
        alt((
            // The `map` combinator runs a parser, then applies a function to the output
            // of that parser.
            literal.map(StringFragment::Literal),
            escaped_char.map(StringFragment::EscapedChar),
        )),
        // Our init value, an empty string
        String::new,
        // Our folding function. For each fragment, append the fragment to the
        // string.
        |mut string, fragment| {
            match fragment {
                StringFragment::Literal(s) => string.push_str(s),
                StringFragment::EscapedChar(c) => string.push(c),
            }
            string
        },
    );
    delimited(char('"'), build_string, char('"')).parse(input)
}

impl Program {
    /// Parse an expression; this function is recursive.
    fn parse_item<'i>(
        &mut self,
        stk: &mut Vec<Index>,
        i: &'i str,
    ) -> Result<(&'i str, Option<Index>)> {
        let i = multispace0(i).map_err(to_anyhow)?.0;

        if let (i, Some(_)) = opt(char('@'))(i).map_err(to_anyhow)? {
            ensure!(stk.len() >= 2, "unexpected '@', cannot pop 2 from stack");
            let a = stk.pop().unwrap(); // safe due to above bounds check
            let f = stk.pop().unwrap(); // safe due to above bounds check

            log::trace!(
                "encountered item '@', stored at index {}, applying {f} to {a}",
                self.body.len()
            );
            self.body.push(Expr::App(f, a));
            stk.push(self.body.len() - 1);
            return Ok((i, None));
        }

        if let (i, Some(label)) = opt(preceded(char(':'), uinteger))(i).map_err(to_anyhow)? {
            // Labels are postfix (e.g., `e :123`), so we put the index of the last expr item
            self.defs[label] = self.body.len() - 1;

            log::trace!(
                "encountered item ':{label}', which will define item at index {}",
                self.body.len() - 1
            );
            return Ok((i, None));
        }

        if let (i, Some(len)) =
            opt(delimited(char('['), uinteger, char(']')))(i).map_err(to_anyhow)?
        {
            ensure!(
                stk.len() >= len,
                "unexpected '[{len}]', cannot pop {len} from stack"
            );

            let mut arr = Vec::new();
            for _ in 0..len {
                arr.push(stk.pop().unwrap()); // safe due to above bounds check
            }
            // Remove this logging once this is clarified
            log::warn!("TODO: validate correct application order for postfix [{len}]");

            log::trace!(
                "encountered item '[{len}], stored at index {}",
                self.body.len()
            );
            self.body.push(Expr::Array(len, arr));
            stk.push(self.body.len() - 1);
            return Ok((i, None));
        }

        if let (i, Some(_)) = opt(char('}'))(i).map_err(to_anyhow)? {
            if let Some(root) = stk.pop() {
                log::trace!("encountered EOF indicator '}}', root stored at {root}");
                return Ok((i, Some(root)));
            } else {
                bail!("encountered EOF indicator '}}' with empty stack");
            }
        }

        let (i, c) = alt((
            preceded(char('&'), double).map(Expr::Float),
            preceded(char('#'), integer).map(Expr::Int),
            preceded(char('_'), uinteger).map(Expr::Ref),
            string_literal.map(Expr::String),
            preceded(char('!'), string_literal).map(Expr::Tick),
            preceded(char('^'), take_till(|c| is_space(c as u8))) // NOTE: this accepts identifiers like ^1piece
                .map(String::from)
                .map(Expr::Ffi),
            take_till1(|c| is_space(c as u8)).map(|s| {
                if let Ok(p) = Combinator::from_str(s) {
                    Expr::Prim(p)
                } else {
                    log::warn!("encountered unknown symbol: {s}");
                    Expr::Unknown(s.to_string())
                }
            }),
        ))
        .parse(i)
        .map_err(to_anyhow)?;

        log::trace!(
            "encountered item '{c}', stored at index {}",
            self.body.len()
        );
        self.body.push(c);
        stk.push(self.body.len() - 1);
        Ok((i, None))
    }
}

impl FromStr for CombFile {
    type Err = anyhow::Error;
    fn from_str(i: &str) -> Result<Self> {
        let version = preceded(char('v'), separated_pair(uinteger, char('.'), uinteger));
        let mut version = preceded(multispace0, version);
        let mut size = preceded(multispace0, uinteger);

        let (i, version) = version(i).map_err(to_anyhow)?;
        ensure!(
            version.0 >= 7,
            "expected comb file version >= 7.0, got {}.{}",
            version.0,
            version.1
        );

        let (i, size) = size(i).map_err(to_anyhow)?;

        log::debug!(
            "Parsing combinator file version: v{}.{}, with {size} definitions",
            version.0,
            version.1
        );

        let mut program = Program {
            root: NIL_INDEX,
            body: Vec::new(),
            defs: Vec::new(),
        };
        program.defs.resize(size, NIL_INDEX);

        let mut stk = Vec::new();
        let mut ii = i;
        let mut root = None;

        while root.is_none() {
            (ii, root) = program.parse_item(&mut stk, ii)?;
        }

        // Some sanity checks
        ensure!(ii.trim().is_empty(), "trailing characters: {ii}");
        ensure!(
            stk.is_empty(),
            "un-applied expressions in parse stack: {stk:#?}"
        );

        for (label, &def) in program.defs.iter().enumerate() {
            ensure!(def != NIL_INDEX, "label #{label} is not initialized");
            ensure!(
                def < program.body.len(),
                "label #{label} is out of bounds: {def}"
            );
        }

        program.root = root.unwrap(); // Safe due to check in while loop

        Ok(Self {
            version,
            size,
            program,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::combinator::*;

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
                /* 3 */ Expr::Prim(Combinator::A),
                /* 4 */ Expr::Prim(Combinator::B),
                /* 5 */ Expr::Prim(Combinator::I), // unreachable
            ],
            defs: vec![],
        };
        assert_eq!(p.to_string(), "A B @ #1 @ }");

        p.body[3] = Expr::Prim(Combinator::K4);
        p.body[4] = Expr::Prim(Combinator::SS);
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
                /* 3 */ Expr::Prim(Combinator::A),
                /* 4 */ Expr::Prim(Combinator::B),
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
                /* 3 */ Expr::Prim(Combinator::A),
                /* 4 */ Expr::Prim(Combinator::B),
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
                /* 5 */ Expr::Prim(Combinator::B),
                /* 6 */ Expr::Prim(Combinator::A),
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
                /* 5 */ Expr::Prim(Combinator::B),
                /* 6 */ Expr::Prim(Combinator::A),
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
                /* 4 */ Expr::Prim(Combinator::B),
                /* 5 */ Expr::Prim(Combinator::A),
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
                /* 4 */ Expr::Prim(Combinator::B),
                /* 5 */ Expr::Prim(Combinator::A),
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
                /* 3 */ Expr::Prim(Combinator::B),
                /* 4 */ Expr::Prim(Combinator::A),
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
            body: vec![Expr::App(0, 1), Expr::Prim(Combinator::I)],
            defs: vec![],
        };
        assert_eq!(p.to_string(), "_0 I @ :0 }");

        // Cycle with ref
        let p = Program {
            root: 0,
            body: vec![
                Expr::App(1, 2),
                Expr::Ref(0),
                Expr::Prim(Combinator::I),
            ],
            defs: vec![0],
        };
        assert_eq!(p.to_string(), "_0 I @ :0 }");
    }

    // TODO: parse tests
}

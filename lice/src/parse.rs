//! Combinator file parser.
use crate::comb::{CombFile, Expr, Index, Prim, Program, NIL_INDEX};
use anyhow::{anyhow, bail, ensure, Result};
use nom::{
    branch::alt,
    bytes::complete::{is_not, take_till},
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
use std::str::FromStr;

/// The result of parsing. A parser monad, even. Typedef'd here for convenience.
type Parse<'a, T> = IResult<&'a str, T, E<'a>>;

/// The nom error type.
///
/// This is already the default in `IResult`, but I define it here as a single letter to make type
/// annot(tions less cumbersome.)
type E<'a> = nom::error::Error<&'a str>;

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
        let i = multispace0::<_, E<'i>>(i).map_err(to_anyhow)?.0;

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
            take_till(|c| is_space(c as u8)).map(|s| {
                if let Ok(p) = Prim::from_str(s) {
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

use darling::{ast, util::SpannedValue, Error, FromDeriveInput, FromMeta, FromVariant};
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while1},
    character::{
        complete::{char, multispace0, multispace1},
        is_alphanumeric,
    },
    combinator::{cut, eof, opt},
    multi::{many1, separated_list1},
    sequence::{delimited, pair, terminated},
    Finish, IResult, Parser,
};
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

fn identifier(i: &str) -> IResult<&str, &str> {
    take_while1(|c: char| is_alphanumeric(c as u8) || c == '_' || c == '\'' || c == '*' || c == ':')(
        i,
    )
}

#[derive(Debug, Clone)]
struct Arg {
    name: String,
    strict: bool,
}

/// Redex (LHS) of a reduction rule, i.e., a list of the names of the args.
#[derive(Debug, Clone)]
struct Redex {
    args: Vec<Arg>,
}

/// Reduct (RHS) of a reduction rule, an AST.
#[derive(Debug, Clone)]
enum Reduct {
    /// Application, e.g., `f x`.
    App(Box<Self>, Box<Self>),
    /// Var, e.g., `x`.
    Var(String),
    /// Reference to the top-level app node of the redex, parsed from `^`.
    ///
    /// Basically only useful for the `Y` combinator, whose rule is `x => x ^`.
    Top,
}

impl Redex {
    /// Parse arguments from string using `nom`.
    fn nom_str(i: &str) -> IResult<&str, Self> {
        let arg = pair(opt(char('!')), identifier);
        let args = separated_list1(
            multispace1,
            arg.map(|(bang, name): (Option<_>, &str)| Arg {
                name: name.to_string(),
                strict: bang.is_some(),
            }),
        )
        .map(|args| Self { args });
        delimited(multispace0, args, multispace0)(i)
    }

    /// Check for duplicate arguments.
    fn check(&self) -> darling::Result<()> {
        for (i, a) in self.args.iter().enumerate() {
            for j in 0..i {
                if a.name == self.args[j].name {
                    return Err(darling::Error::custom(format!(
                        "duplicate argument: {}",
                        a.name
                    )));
                }
            }
        }
        Ok(())
    }

    /// Find the 0-based index of a given variable name
    fn index_of(&self, name: &str) -> Option<usize> {
        self.args.iter().position(|arg| arg.name == name)
    }
}

impl FromMeta for Redex {
    fn from_string(value: &str) -> darling::Result<Self> {
        terminated(Self::nom_str, eof)(value)
            .finish()
            .map(|(_, v)| v)
            .map_err(darling::Error::custom)
    }
}

impl Reduct {
    /// Parse arguments from string using `nom`.
    fn nom_str(i: &str) -> IResult<&str, Self> {
        let atom = alt((
            delimited(char('('), cut(Self::nom_str), cut(char(')'))),
            tag("^").map(|_| Self::Top),
            identifier.map(|s: &str| Self::Var(s.to_owned())),
        ));
        let atom = delimited(multispace0, atom, multispace0);

        let (i, atoms) = many1(atom)(i)?;

        let mut atoms = atoms.into_iter();
        let mut r = atoms.next().expect("reduct should have at least atom");
        for a in atoms {
            r = Self::App(Box::new(r), Box::new(a));
        }

        Ok((i, r))
    }

    /// Check for undefined variables (i.e., not present in arguments).
    fn check(&self, redex: &Redex) -> darling::Result<()> {
        let _ = self.compile(redex)?;
        Ok(())
    }

    /// Compile redux to symbolic bytecode representation, in bytecode.
    fn compile(&self, redex: &Redex) -> darling::Result<Vec<TokenStream>> {
        Ok(match self {
            Reduct::App(f, a) => {
                let f = f.compile(redex)?;
                let a = a.compile(redex)?;
                [f, a, vec![quote!(ReduxCode::App)]].concat()
            }
            Reduct::Var(a) => {
                let Some(i) = redex.index_of(a) else {
                    return Err(darling::Error::custom(format!("undefined variable: {a}")));
                };
                vec![quote!(ReduxCode::Arg(#i))]
            }
            Reduct::Top => {
                vec![quote!(ReduxCode::Top)]
            }
        })
    }
}

impl FromMeta for Reduct {
    fn from_string(value: &str) -> darling::Result<Self> {
        terminated(Self::nom_str, eof)(value)
            .finish()
            .map(|(_, v)| v)
            .map_err(darling::Error::custom)
    }
}

/// Metadata attached to each combinator.
#[derive(Debug, FromVariant, Clone)]
#[darling(attributes(reduce))]
struct ReduceVariant {
    /// The variant name of the combinator symbol
    ident: Ident,
    /// Specifies the arity and strictness of the combinator's arguments
    ///
    /// The names of the arguments can (and should) be used for documentation purposes.
    from: Option<SpannedValue<Redex>>,
    /// Specifies the concrete returned expression when reducing the combinator
    ///
    /// Used to generate the combinator's redux code.
    to: Option<SpannedValue<Reduct>>,
    /// Specifies the return type when reducing the combinator
    ///
    /// Only used for documentation purposes. Should not overlap with `to`.
    to_io: Option<SpannedValue<String>>,
    /// Specifies that the combinator reduces without any arguments.
    ///
    /// Should not overlap with `from`.
    constant: Option<SpannedValue<()>>,
}

impl ReduceVariant {
    fn arity(&self) -> darling::Result<usize> {
        match (&self.from, self.constant) {
            (None, Some(_)) => Ok(0),
            (Some(from), None) => Ok(from.args.len()),
            (Some(_), Some(_)) => Err(darling::Error::custom(
                "redex ('from') specified for 'constant' (where arity = 0)",
            )
            .with_span(&self.ident.span())),
            (None, None) => Err(darling::Error::custom("no redex ('from') specified!")
                .with_span(&self.ident.span())),
        }
    }

    fn check(&self) -> darling::Result<()> {
        if let Some(from) = &self.from {
            from.check().map_err(|e| e.with_span(&from.span()))?;
            if let Some(to) = &self.to {
                if let Some(to_io) = &self.to_io {
                    return Err(darling::Error::custom(
                        "'to' and 'to_io' cannot both be defined'".to_string(),
                    )
                    .with_span(&to.span())
                    .with_span(&to_io.span()));
                }
                to.check(from).map_err(|e| e.with_span(&to.span()))?;
            }
        } else if let Some(to) = &self.to {
            return Err(
                darling::Error::custom("'to' defined without 'from'".to_string())
                    .with_span(&to.span()),
            );
        }
        let _ = self.arity()?; // check that arity returns Ok()
        Ok(())
    }
}

/// Reduce `enum` definition.
#[derive(Debug, FromDeriveInput)]
#[darling(attributes(reduce), supports(enum_unit))]
struct ReduceEnum {
    ident: Ident,
    data: ast::Data<ReduceVariant, ()>,
}

impl ReduceEnum {
    /// Generate `impl Reduce for TheEnum` definition.
    fn make_impl(&self) -> darling::Result<TokenStream> {
        let mut errors = darling::Error::accumulator();

        let ident = &self.ident;
        let ast::Data::Enum(variants) = &self.data else {
            panic!("{ident} only supports(enum_unit)")
        };

        // First do sanity-checks (and accumulate potential errors)
        for v in variants {
            errors.handle(v.check());
        }

        // No more errors possible from here on out
        errors.finish()?;

        let reduxes = variants.iter().map(|v| {
            let variant = &v.ident;
            if let Some(to) = &v.to {
                let code = to.compile(v.from.as_ref().unwrap()).unwrap();
                quote!(#ident::#variant => Some(&[#(#code),*]))
            } else {
                quote!(#ident::#variant => None)
            }
        });

        let strictness = variants.iter().map(|v| {
            let variant = &v.ident;
            if let Some(from) = &v.from {
                let strict = from.args.iter().map(|arg| arg.strict);
                quote!(#ident::#variant => &[#(#strict),*])
            } else {
                quote!(#ident::#variant => &[])
            }
        });

        let io_action = variants.iter().map(|v| {
            let variant = &v.ident;
            if v.to_io.is_some() {
                quote!(#ident::#variant => true)
            } else {
                quote!(#ident::#variant => false)
            }
        });

        Ok(quote! {
            impl crate::combinator::Reduce for #ident {
                #[inline]
                fn strictness(&self) -> &'static [bool] {
                    match self {
                        #(#strictness),*
                    }
                }

                #[inline]
                fn redux(&self) -> Option<&'static [crate::combinator::ReduxCode]> {
                    match self {
                        #(#reduxes),*
                    }
                }
                #[inline]
                fn io_action(&self) -> bool {
                    match self {
                        #(#io_action),*
                    }
                }
            }
        })
    }
}

/// Derive the `Reduce` trait implementation for an `enum` definition.
pub fn reduce(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    ReduceEnum::from_derive_input(&input)
        .map_or_else(Error::write_errors, |e| {
            e.make_impl().unwrap_or_else(Error::write_errors)
        })
        .into()
}

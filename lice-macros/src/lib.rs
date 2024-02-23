use darling::{ast, util::SpannedValue, Error, FromDeriveInput, FromMeta, FromVariant};
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alphanumeric1, char, multispace0, multispace1},
    combinator::cut,
    multi::{many1, separated_list1},
    sequence::delimited,
    Finish, IResult, Parser,
};
use proc_macro2::{Ident, TokenStream};
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

/// Redex (LHS) of a reduction rule, i.e., a list of the names of the args.
#[derive(Debug, Clone)]
struct Redex {
    args: Vec<String>,
}

/// Reduct (RHS) of a reduction rule, an AST.
#[derive(Debug, Clone)]
enum Reduct {
    /// Application, e.g., `f x`.
    App(Box<Self>, Box<Self>),
    /// Var, e.g., `x`.
    Var(String),
    /// Reference to the top-level node of the redex, parsed from `self`.
    ///
    /// Basically only useful for the `Y` combinator, whose rule is `x => x self`.
    SelfRef,
}

impl Redex {
    /// Parse arguments from string using `nom`.
    fn nom_str(i: &str) -> IResult<&str, Self> {
        let p = separated_list1(multispace1, alphanumeric1.map(|s: &str| s.to_string()))
            .map(|args| Self { args });
        delimited(multispace0, p, multispace0)(i)
    }

    /// Check for duplicate arguments.
    fn check(&self) -> darling::Result<()> {
        for (i, a) in self.args.iter().enumerate() {
            for j in 0..i {
                if a == &self.args[j] {
                    return Err(darling::Error::custom(format!("duplicate argument: {a}")));
                }
            }
        }
        Ok(())
    }
}

impl FromMeta for Redex {
    fn from_string(value: &str) -> darling::Result<Self> {
        Self::nom_str(value)
            // .finish()
            .map(|(_, v)| v)
            .map_err(darling::Error::custom)
    }
}

impl Reduct {
    /// Parse arguments from string using `nom`.
    fn nom_str(i: &str) -> IResult<&str, Self> {
        let atom = alt((
            delimited(char('('), cut(Self::nom_str), cut(char(')'))),
            tag("self").map(|_| Self::SelfRef),
            alphanumeric1.map(|s: &str| Self::Var(s.to_owned())),
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
        match self {
            Reduct::App(f, a) => {
                f.check(redex)?;
                a.check(redex)
            }
            Reduct::Var(a) if !redex.args.contains(a) => {
                Err(darling::Error::custom(format!("undefined variable: {a}")))
            }
            _ => Ok(()),
        }
    }
}

impl FromMeta for Reduct {
    fn from_string(value: &str) -> darling::Result<Self> {
        Self::nom_str(value)
            .finish()
            .map(|(_, v)| v)
            .map_err(darling::Error::custom)
    }
}

/// Metadata attached to each combinator.
#[derive(Debug, FromVariant, Clone)]
#[darling(attributes(reduce))]
struct ReduceVariant {
    ident: Ident,
    from: Option<SpannedValue<Redex>>,
    to: Option<SpannedValue<Reduct>>,
    arity: Option<SpannedValue<usize>>,
}

impl ReduceVariant {
    fn arity(&self) -> darling::Result<usize> {
        match (&self.from, self.arity) {
            (None, Some(arity)) => Ok(*arity),
            (Some(from), None) => Ok(from.args.len()),
            (Some(from), Some(arity)) => {
                if from.args.len() == *arity {
                    Ok(*arity)
                } else {
                    Err(darling::Error::custom(
                        "specified arity does not match number of arguments in 'from'",
                    )
                    .with_span(&self.ident.span()))
                }
            }
            (None, None) => {
                Err(darling::Error::custom("no arity specified!").with_span(&self.ident.span()))
            }
        }
    }

    fn check(&self) -> darling::Result<()> {
        if let Some(from) = &self.from {
            from.check().map_err(|e| e.with_span(&from.span()))?;
            if let Some(to) = &self.to {
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

        // Generate `fn arity()` implementation
        let arities = variants.iter().map(|v| {
            let variant = &v.ident;
            let arity = v.arity().unwrap();
            quote!(#ident::#variant => #arity)
        });

        Ok(quote! {
            impl crate::combinator::Reduce for #ident {
                fn arity(&self) -> usize {
                    match self {
                        #(#arities),*
                    }
                }
            }
        })
    }
}

/// Derive the `Reduce` trait implementation for an `enum` definition.
#[proc_macro_derive(Reduce, attributes(reduce))]
pub fn reduce(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    ReduceEnum::from_derive_input(&input)
        .map_or_else(Error::write_errors, |e| {
            e.make_impl().unwrap_or_else(Error::write_errors)
        })
        .into()
}

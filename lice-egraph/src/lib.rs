use std::collections::{HashMap, HashSet};
use egg::*;
use ordered_float::OrderedFloat;
use lice::combinator::Combinator;
use lice::file::{Expr, Index, Program};

/*
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub enum SKI {
    Prim(Combinator),
    App([Id; 2]),
    Int(i64),
    Float(OrderedFloat<f64>),
    Array(usize, Vec<Id>),
    Ref(usize),
    String(String),
    Tick(String),
    Ffi(String),
    Unknown(String),
    RefPlaceholder(usize),
    Placeholder(usize),
}
impl egg::Language for SKI {
    #[inline(always)]
    fn matches(&self, other: &Self) -> bool {
        ::std::mem::discriminant(self) == ::std::mem::discriminant(other)
            && match (self, other) {
                (SKI::Prim(data1), SKI::Prim(data2)) => data1 == data2,
                (SKI::App(l), SKI::App(r)) => {
                    egg::LanguageChildren::len(l) == egg::LanguageChildren::len(r)
                }
                (SKI::Int(data1), SKI::Int(data2)) => data1 == data2,
                (SKI::Float(data1), SKI::Float(data2)) => data1 == data2,
                (SKI::Array(d1, l), SKI::Array(d2, r)) => {
                    d1 == d2 && egg::LanguageChildren::len(l) == egg::LanguageChildren::len(r)
                }
                (SKI::Ref(data1), SKI::Ref(data2)) => data1 == data2,
                (SKI::String(data1), SKI::String(data2)) => data1 == data2,
                (SKI::Tick(data1), SKI::Tick(data2)) => data1 == data2,
                (SKI::Ffi(data1), SKI::Ffi(data2)) => data1 == data2,
                (SKI::Unknown(data1), SKI::Unknown(data2)) => data1 == data2,
                (SKI::RefPlaceholder(data1), SKI::RefPlaceholder(data2)) => data1 == data2,
                (SKI::Placeholder(data1), SKI::Placeholder(data2)) => data1 == data2,
                _ => false,
            }
    }
    fn children(&self) -> &[Id] {
        match self {
            SKI::Prim(_data) => &[],
            SKI::App(ids) => egg::LanguageChildren::as_slice(ids),
            SKI::Int(_data) => &[],
            SKI::Float(_data) => &[],
            SKI::Array(_, ids) => egg::LanguageChildren::as_slice(ids),
            SKI::Ref(_data) => &[],
            SKI::String(_data) => &[],
            SKI::Tick(_data) => &[],
            SKI::Ffi(_data) => &[],
            SKI::Unknown(_data) => &[],
            SKI::RefPlaceholder(_data) => &[],
            SKI::Placeholder(_data) => &[],
        }
    }
    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            SKI::Prim(_data) => &mut [],
            SKI::App(ids) => egg::LanguageChildren::as_mut_slice(ids),
            SKI::Int(_data) => &mut [],
            SKI::Float(_data) => &mut [],
            SKI::Array(_, ids) => egg::LanguageChildren::as_mut_slice(ids),
            SKI::Ref(_data) => &mut [],
            SKI::String(_data) => &mut [],
            SKI::Tick(_data) => &mut [],
            SKI::Ffi(_data) => &mut [],
            SKI::Unknown(_data) => &mut [],
            SKI::RefPlaceholder(_data) => &mut [],
            SKI::Placeholder(_data) => &mut [],
        }
    }
}
impl ::std::fmt::Display for SKI {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        match (self, f) {
            (SKI::Prim(data), f) => ::std::fmt::Display::fmt(data, f),
            (SKI::App(..), f) => f.write_str("@"),
            (SKI::Int(data), f) => ::std::fmt::Display::fmt(data, f),
            (SKI::Float(data), f) => ::std::fmt::Display::fmt(data, f),
            (SKI::Array(data, _), f) => f.write_fmt(format_args!("Arr<{:#?}>", data)),
            (SKI::Ref(data), f) => f.write_fmt(format_args!("Ref<{:#?}>", data)),
            (SKI::String(data), f) => f.write_fmt(format_args!("Str<{:#?}>", data)),
            (SKI::Tick(data), f) => f.write_fmt(format_args!("Tick<{:#?}>", data)),
            (SKI::Ffi(data), f) => f.write_fmt(format_args!("FFI<{:#?}>", data)),
            (SKI::Unknown(data), f) => f.write_fmt(format_args!("Unknown<{:#?}>", data)),
            (SKI::RefPlaceholder(data), f) => ::std::fmt::Display::fmt(data, f),
            (SKI::Placeholder(data), f) => ::std::fmt::Display::fmt(data, f),
        }
    }
}
impl egg::FromOp for SKI {
    type Error = egg::FromOpError;
    fn from_op(
        op: &str,
        children: ::std::vec::Vec<egg::Id>,
    ) -> ::std::result::Result<Self, Self::Error> {
        match (op, children) {
            (op, children) if op.parse::<Combinator>().is_ok() && children.is_empty() => {
                Ok(SKI::Prim(op.parse().unwrap()))
            }
            (op, children)
                if op == "@"
                    && <[Id; 2] as egg::LanguageChildren>::can_be_length(children.len()) =>
            {
                let children = <[Id; 2] as egg::LanguageChildren>::from_vec(children);
                Ok(SKI::App(children))
            }
            (op, children) if op.parse::<i64>().is_ok() && children.is_empty() => {
                Ok(SKI::Int(op.parse().unwrap()))
            }
            (op, children) if op.parse::<OrderedFloat<f64>>().is_ok() && children.is_empty() => {
                Ok(SKI::Float(op.parse().unwrap()))
            }
            (op, children)
                if op.parse::<usize>().is_ok()
                    && <Vec<Id> as egg::LanguageChildren>::can_be_length(children.len()) =>
            {
                let data = op.parse::<usize>().unwrap();
                let children = <Vec<Id> as egg::LanguageChildren>::from_vec(children);
                Ok(SKI::Array(data, children))
            }
            (op, children) if op.parse::<usize>().is_ok() && children.is_empty() => {
                Ok(SKI::Ref(op.parse().unwrap()))
            }
            (op, children) if op.parse::<String>().is_ok() && children.is_empty() => {
                Ok(SKI::String(op.parse().unwrap()))
            }
            (op, children) if op.parse::<String>().is_ok() && children.is_empty() => {
                Ok(SKI::Tick(op.parse().unwrap()))
            }
            (op, children) if op.parse::<String>().is_ok() && children.is_empty() => {
                Ok(SKI::Ffi(op.parse().unwrap()))
            }
            (op, children) if op.parse::<String>().is_ok() && children.is_empty() => {
                Ok(SKI::Unknown(op.parse().unwrap()))
            }
            (op, children) if op.parse::<usize>().is_ok() && children.is_empty() => {
                Ok(SKI::RefPlaceholder(op.parse().unwrap()))
            }
            (op, children) if op.parse::<usize>().is_ok() && children.is_empty() => {
                Ok(SKI::Placeholder(op.parse().unwrap()))
            }
            (op, children) => Err(egg::FromOpError::new(op, children)),
        }
    }
}
*/

define_language! {
    pub enum SKI {
        Prim(Combinator),
        "@" = App([Id; 2]),
        Int(i64),
        Float(OrderedFloat<f64>),
        Array(usize, Vec<Id>),
        Ref(usize),
        String(String),
        Tick(String),
        Ffi(String),
        Unknown(String),
        RefPlaceholder(usize),
        Placeholder(usize),
    }
}

/*
 * Read Zulip about placeholders, cyclic
 * small test case that exhibits same weird behavior (A's) and figure out
 * make sure placeholders are working, get good grasp of cost function
 */
struct AstSizeHi;
impl CostFunction<SKI> for AstSizeHi{
    type Cost = usize;
    fn cost<C>(&mut self, enode: &SKI, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost
    {
        let node_cost = match enode {
            SKI::Placeholder(_) => 9999999,
            SKI::Ref(_) => 9999999,
            _ => 1
        };
        enode.fold(node_cost, |sum, id| sum.saturating_add(costs(id)))
    }
}


fn ski_reductions() -> Vec<Rewrite<SKI, ()>> {
    vec![
        rewrite!("S"; "(@ (@ (@ S ?f) ?g) ?x)" => "(@ (@ ?f ?x) (@ ?g ?x))"),
        rewrite!("K";"(@ (@ K ?f) ?g)" => "?f"),
        rewrite!("I"; "(@ I ?x)" => "?x"),
        rewrite!("B"; "(@ (@ (@ B ?f) ?g) ?x)" => "(@ ?f (@ ?g ?x))"),
        rewrite!("C"; "(@ (@ (@ C ?f) ?x) ?y)" => "(@ (@ ?f ?y) ?x)"),
        rewrite!("A"; "(@ (@ A ?x) ?y)" => "?y"),
        rewrite!("S'"; "(@ (@ (@ (@ S' ?c) ?f) ?g) ?x)" => "(@ (@ ?c (@ ?f ?x)) (@ ?g ?x))"),
        rewrite!("B'"; "(@ (@ (@ (@ B' ?c) ?f) ?g) ?x)" => "(@ (@ ?c ?f) (@ ?g ?x))"),
        rewrite!("C'"; "(@ (@ (@ (@ C' ?c) ?f) ?x) ?y)" => "(@ (@ ?c (@ ?f ?y)) ?x)"),
        rewrite!("P"; "(@ (@ (@ P ?x) ?y) ?f)" => "(@ (@ ?f ?x) ?y)"),
        rewrite!("R"; "(@ (@ (@ R ?y) ?f) ?x)" => "(@ (@ ?f ?x) ?y)"),
        rewrite!("O"; "(@ (@ (@ (@ O ?x) ?y) ?z) ?f)" => "(@ (@ ?f ?x) ?y)"),
        rewrite!("U"; "(@ (@ U ?x) ?f)" => "(@ ?f ?x)"),
        rewrite!("Z"; "(@ (@ (@ Z ?f) ?x) ?y)" => "(@ ?f ?x)"),
        rewrite!("K2"; "(@ (@ (@ K2 ?x) ?y) ?z)" => "?x"),
        rewrite!("K3"; "(@ (@ (@ (@ K3 ?x) ?y) ?z) ?w)" => "?x"),
        rewrite!("K4"; "(@ (@ (@ (@ (@ K4 ?x) ?y) ?z) ?w) ?v)" => "?x"),
        rewrite!("C'B"; "(@ (@ (@ (@ C'B ?f) ?g) ?x) ?y)" => "(@ (@ ?f ?x) (@ ?g ?y))"),
    ]
}

pub fn program_to_egraph(program: &Program) -> (Id, HashMap<Index, Id>, EGraph<SKI, ()>) {
    let mut egraph = EGraph::<SKI, ()>::default();
    let mut idx_eid_map = HashMap::<Index, Id>::new();
    let mut refs = HashSet::<Id>::new();
    let root = construct_egraph(program, &mut idx_eid_map, &mut refs, &mut egraph, program.root);
    resolve_refs(&idx_eid_map, &refs, &mut egraph);
    (root, idx_eid_map, egraph)
}

pub fn program_to_egraph_hi(program: &Program) -> (Id, HashMap<Index, Id>, EGraph<SKI, ()>) {
    let mut egraph = EGraph::<SKI, ()>::default();
    let mut idx_eid_map = HashMap::<Index, Id>::new();
    construct_egraph_hi(program, &mut idx_eid_map, &mut egraph);
    let root = match idx_eid_map.get(&program.root) {
        Some(eid) => *eid,
        None => panic!("missing root"),
    };
    
    println!("root: {:#?} program root: {:#?}\n", root, &program.root);

    egraph.classes().for_each(|ec| {
        if ec.id == root {
            println!("{:#?}", ec)
        }
    });

    (root, idx_eid_map, egraph)
}

pub fn program_to_egraph_it(program: &Program) -> (Id, HashMap<Index, Id>, EGraph<SKI, ()>) {
    let mut egraph = EGraph::<SKI, ()>::default();
    let mut idx_eid_map = HashMap::<Index, Id>::new();
    construct_egraph_iter(program, &mut idx_eid_map, &mut egraph);

    let root = match idx_eid_map.get(&program.root) {
        Some(eid) => *eid,
        None => panic!("missing root"),
    };
    /* 
    println!("root: {:#?} program root: {:#?}\n", root, &program.root);

    egraph.classes().for_each(|ec| {
        if ec.id == root {
            println!("{:#?}", ec)
        }
    });
    */

    (root, idx_eid_map, egraph)
}

pub fn program_to_egraph_ph(program: &Program) -> (Id, HashMap<Index, Id>, EGraph<SKI, ()>) {
    let mut egraph = EGraph::<SKI, ()>::default();
    let mut idx_eid_map = HashMap::<Index, Id>::new();
    add_placeholders(program, &mut idx_eid_map, &mut egraph);
    fill_placeholders(program, &mut idx_eid_map, &mut egraph);
    
    let root = match idx_eid_map.get(&program.root) {
        Some(eid) => *eid,
        None => panic!("missing root"),
    };
    
    (root, idx_eid_map, egraph)
}

pub fn optimize(egraph: EGraph<SKI, ()>, root: Id, fname: &str) -> String {
    let runner = Runner::<SKI, ()>::default()
        .with_egraph(egraph)
        .run(&ski_reductions());
        // .run(&vec![]);

    runner.egraph.dot().to_svg(fname).unwrap();

    let extractor = Extractor::new(&runner.egraph, AstSizeHi);
    let (_, best) = extractor.find_best(root);
    
    println!("best: {:#?}", best);
    
    best.to_string()
}

pub fn noop(egraph: EGraph<SKI, ()>, root: Id) -> String {
    let runner = Runner::<SKI, ()>::default()
        .with_egraph(egraph)
        .run(&vec![]);

    let extractor = Extractor::new(&runner.egraph, AstSizeHi);
    let (_, best) = extractor.find_best(root);
    
    println!("best: {:#?}", best);
    
    best.to_string()
}

fn const_expr_to_enode(expr: &Expr) -> SKI {
    match expr {
        Expr::Prim(comb) => SKI::Prim(*comb),
        Expr::Int(i) => SKI::Int(*i),
        Expr::Float(flt) => SKI::Float(OrderedFloat(*flt)),
        Expr::Ref(lbl) => SKI::Ref(*lbl),
        Expr::String(s) => SKI::String(s.to_string()),
        Expr::Tick(s) => SKI::Tick(s.to_string()),
        Expr::Ffi(s) => SKI::Ffi(s.to_string()),
        _ => todo!("unreachable, not a const expr {:#?}", expr),
    }
}

fn add_placeholders(
    program: &Program,
    idx_eid_map: &mut HashMap<Index, Id>,
    egraph: &mut EGraph<SKI, ()>
) {
    program.body.iter().enumerate().for_each(|(idx, _)| {
        let eid = egraph.add(SKI::Placeholder(idx));
        idx_eid_map.insert(idx, eid);
    })
}

fn fill_placeholders(
    program: &Program,
    idx_eid_map: &mut HashMap<Index, Id>,
    egraph: &mut EGraph<SKI, ()>
) {
    program.body.iter().enumerate().for_each(|(idx, expr)| {
        let eid = match expr {
            Expr::Prim(comb) => egraph.add(SKI::Prim(*comb)),
            Expr::Int(i) => egraph.add(SKI::Int(*i)),
            Expr::Float(flt) => egraph.add(SKI::Float(OrderedFloat(*flt))),
            Expr::String(s) => egraph.add(SKI::String(s.to_string())),
            Expr::Tick(s) => egraph.add(SKI::Tick(s.to_string())),
            Expr::Ffi(s) => egraph.add(SKI::Ffi(s.to_string())),
            Expr::Ref(lbl) => egraph.add(SKI::Ref(*lbl)),
            Expr::App(f, a) => {
                let func_eid = match idx_eid_map.get(f) {
                    Some(eid) => *eid,
                    None => panic!("missing placeholder: {:#?}", f),
                };
                let arg_eid = match idx_eid_map.get(a) {
                    Some(eid) => *eid,
                    None => panic!("missing placeholder: {:#?}", f),
                };
                egraph.add(SKI::App([func_eid, arg_eid]))
            },
            Expr::Array(u, arr) => {
                let e_arr: Vec<Id> = arr.iter().map(|i| { 
                    let elmt_eid = match idx_eid_map.get(i) {
                        Some(eid) => *eid,
                        None => panic!("missing placeholder: {:#?}", i),
                    };
                    elmt_eid
                }).collect();
                egraph.add(SKI::Array(*u, e_arr))
            },
            _ => panic!("unknown expr: {:#?}", expr),
        };

        let placeholder = match idx_eid_map.get(&idx) {
            Some(x) => *x,
            None => panic!("missing placeholder eclass"),
        };

        egraph.union(placeholder, eid);

        if let Expr::Ref(lbl) = expr {
            let ref_idx = &program.defs[*lbl];
            let ref_eid = match idx_eid_map.get(ref_idx) {
                Some(x) => *x,
                None => panic!("missing ref"),
            };
            egraph.union(eid, ref_eid);
        }
    })
}

fn add_const_exprs_to_egraph(
    program: &Program,
    idx_eid_map: &mut HashMap<Index, Id>,
    egraph: &mut EGraph<SKI, ()>,
) {
    let const_exprs = program.body.iter().enumerate().filter(|(_, expr)| {
        matches!(expr, Expr::Prim(_) | Expr::Int(_) | Expr::Float(_) | Expr::Ref(_) | Expr::String(_) | Expr::Tick(_) | Expr::Ffi(_))
    });

    const_exprs.for_each(|(idx, expr)| {
        let enode = const_expr_to_enode(expr);
        let eid = egraph.add(enode);
        idx_eid_map.insert(idx, eid);
    });
}

fn recursively_add_exprs(
    program: &Program,
    idx_eid_map: &mut HashMap<Index, Id>,
    egraph: &mut EGraph<SKI, ()>,
    idx: Index
) -> Id {
    if let Some(eid) = idx_eid_map.get(&idx) {
        return *eid;
    }

    match &program.body[idx] {
        Expr::App(f, a) => {
            let func_eid = match idx_eid_map.get(f) {
                Some(eid) => *eid,
                None => recursively_add_exprs(program, idx_eid_map, egraph, *f),
            };
            let arg_eid = match idx_eid_map.get(a) {
                Some(eid) => *eid,
                None => recursively_add_exprs(program, idx_eid_map, egraph, *a),
            };
            let app_eid = egraph.add(SKI::App([func_eid, arg_eid]));
            idx_eid_map.insert(idx, app_eid);
            app_eid
        },
        Expr::Array(u, arr) => {
            let e_arr: Vec<Id> = arr.iter().map(|i| { 
                let elmt_eid = match idx_eid_map.get(i) {
                    Some(eid) => *eid,
                    None => recursively_add_exprs(program, idx_eid_map, egraph, *i),
                };
                elmt_eid
            }).collect();
            let arr_eid = egraph.add(SKI::Array(*u, e_arr));
            idx_eid_map.insert(idx, arr_eid);
            arr_eid
        },
        _ => panic!("unreachable"),
    }
}

fn add_rec_expr_to_egraph(
    program: &Program,
    idx_eid_map: &mut HashMap<Index, Id>,
    egraph: &mut EGraph<SKI, ()>,
) {
    program.body.iter().enumerate().filter(|(_, expr)| {
        matches!(expr, Expr::App(_, _) | Expr::Array(_, _))
    }).for_each(|(idx, expr)| { 
        recursively_add_exprs(program, idx_eid_map, egraph, idx);
    });
}

fn construct_egraph_hi(
    program: &Program,
    idx_eid_map: &mut HashMap<Index, Id>,
    egraph: &mut EGraph<SKI, ()>,
) {
    add_placeholders(program, idx_eid_map, egraph);
    fill_placeholders(program, idx_eid_map, egraph);
}

fn unify_refs(
    program: &Program,
    idx_eid_map: &mut HashMap<Index, Id>,
    egraph: &mut EGraph<SKI, ()>,
) {
    program.body.iter().enumerate().filter(|(_, expr)| {
        matches!(expr, Expr::Ref(_))
    }).for_each(|(idx, expr)| {
        match expr {
            Expr::Ref(lbl) => {
                let ref_eid = match idx_eid_map.get(&idx) {
                    Some(x) => *x,
                    None => panic!("missing ref"),
                };
                let lbl_idx = &program.defs[*lbl];
                let lbl_eid = match idx_eid_map.get(lbl_idx) {
                    Some(x) => *x,
                    None => panic!("missing referenced obj"),
                };
                egraph.union(ref_eid, lbl_eid);
            },
            _ => panic!("unreachable"),
        }
    });
}

fn construct_egraph_iter(
    program: &Program,
    idx_eid_map: &mut HashMap<Index, Id>,
    egraph: &mut EGraph<SKI, ()>,
) {
    add_const_exprs_to_egraph(program, idx_eid_map, egraph);
    add_rec_expr_to_egraph(program, idx_eid_map, egraph);
    unify_refs(program, idx_eid_map, egraph);
}

fn construct_egraph(
    program: &Program,
    idx_eid_map: &mut HashMap<Index, Id>,
    refs: &mut HashSet<Id>,
    egraph: &mut EGraph<SKI, ()>,
    idx: Index,
) -> Id {
    match &program.body[idx] {
        Expr::App(f, a) => {
            let func_eid = match idx_eid_map.get(f) {
                Some(eid) => *eid,
                None => construct_egraph(program, idx_eid_map, refs, egraph, *f),
            };
            let arg_eid = match idx_eid_map.get(a) {
                Some(eid) => *eid,
                None => construct_egraph(program, idx_eid_map, refs, egraph, *a),
            };
            let app_eid = egraph.add(SKI::App([func_eid, arg_eid]));
            idx_eid_map.insert(idx, app_eid);
            app_eid
        }
        Expr::Prim(comb) => {
            let comb_eid = egraph.add(SKI::Prim(*comb));
            idx_eid_map.insert(idx, comb_eid);
            comb_eid
        }
        Expr::Int(i) => {
            let int_eid = egraph.add(SKI::Int(*i));
            idx_eid_map.insert(idx, int_eid);
            int_eid
        }
        Expr::Float(flt) => {
            let float_eid = egraph.add(SKI::Float(OrderedFloat(*flt)));
            idx_eid_map.insert(idx, float_eid);
            float_eid
        }
        Expr::Array(u, arr) => {
            let e_arr: Vec<Id> = arr.iter().map(|idx| { 
                let elmt_eid = match idx_eid_map.get(idx) {
                    Some(eid) => *eid,
                    None => construct_egraph(program, idx_eid_map, refs, egraph, *idx),
                };
                elmt_eid
            }).collect();
            let arr_eid = egraph.add(SKI::Array(*u, e_arr));
            idx_eid_map.insert(idx, arr_eid);
            arr_eid
        }
        Expr::Ref(lbl) => {
            let def_idx = &program.defs[*lbl]; // index of referenced expr in program body
            /*
            println!("hi {:#?}", def_idx);
            let ref_obj_eid = match idx_eid_map.get(def_idx) {
                Some(eid) => *eid,
                None => construct_egraph(program, idx_eid_map, refs, egraph, *def_idx),
            };
            let ref_eid = egraph.add(SKI::Ref(usize::from(ref_obj_eid)));
            */
            let ref_eid = egraph.add(SKI::RefPlaceholder(*def_idx));
            refs.insert(ref_eid);
            idx_eid_map.insert(idx, ref_eid);
            ref_eid
        }
        Expr::String(s) => {
            let str_eid = egraph.add(SKI::String(s.to_string()));
            idx_eid_map.insert(idx, str_eid);
            str_eid
        }
        Expr::Tick(s) => {
            let tick_eid = egraph.add(SKI::Tick(s.to_string()));
            idx_eid_map.insert(idx, tick_eid);
            tick_eid
        }
        Expr::Ffi(s) => {
            let ffi_eid = egraph.add(SKI::Ffi(s.to_string()));
            idx_eid_map.insert(idx, ffi_eid);
            ffi_eid
        }
        _ => todo!("Unknown lice Expr"),
    }
}

fn resolve_refs(
    idx_eid_map: &HashMap<Index, Id>,
    refs: &HashSet<Id>,
    egraph: &mut EGraph<SKI, ()>
) {
    let refs_to_create = egraph.classes_mut().filter(|ec| refs.contains(&ec.id));
    let mut refs_to_unify = HashSet::<(Id, Id)>::new();
    let mut hmm = Vec::<Id>::new();

    println!("{:#?}", idx_eid_map);

    // create the refs
    refs_to_create.for_each(|ec| {
        let placeholder = &ec.nodes[0];
        println!("{:#?}", ec);
        let ref_node = match placeholder {
            SKI::RefPlaceholder(lbl_idx) => {
                let lbl_eid = match idx_eid_map.get(lbl_idx) {
                    Some(eid) => *eid,
                    None => panic!("missing ref"),
                };
                refs_to_unify.insert((ec.id, lbl_eid));
                hmm.push(lbl_eid);
                SKI::Ref(usize::from(lbl_eid))
                // SKI::RefPlaceholder(usize::from(lbl_eid))
                // SKI::RefPlaceholder(lbl_idx)
            }
            _ => panic!("unreachable"),
        };
        ec.nodes[0] = ref_node;
        println!("{:#?}\n", ec);
    });
    
    hmm.sort();
    println!("{:#?}", hmm);

    // unify refs
    refs_to_unify.iter().for_each(|(ref_eid, lbl_eid)| {
        egraph.union(*ref_eid, *lbl_eid);
    });

    println!("hi");
}

fn construct_program(
    expr: &RecExpr<SKI>,
    root: Id
) -> Program {
    let expr_vec = expr.as_ref();
    let body = expr_vec.iter().map(|e| match e {
        SKI::App([f, a]) => Expr::App(usize::from(*f), usize::from(*a)),
        SKI::Prim(comb) => Expr::Prim(*comb),
        SKI::Int(i) => Expr:: Int(*i),
        _ => todo!("add more exprs")
    }).collect();
    Program {
        root: usize::from(root),
        body,
        defs: vec![0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_skk_to_id() {
        let mut egraph = EGraph::<SKI, ()>::default();
        let root = egraph.add_expr(&"(@ (@ (@ S K) K) 1)".parse().unwrap());
        let runner = Runner::<SKI, ()>::default()
            .with_egraph(egraph)
            .run(&ski_reductions());

        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (_, best) = extractor.find_best(root);
        assert!(best.to_string() == "1");
    }

    #[test]
    fn program_to_egraph_then_reduce() {
        let p = Program {
            root: 5,
            body: vec![
                /* 0 */ Expr::Prim(Combinator::S),
                /* 1 */ Expr::Prim(Combinator::K),
                /* 2 */ Expr::Int(1),
                /* 3 */ Expr::App(0, 1),
                /* 4 */ Expr::App(3, 1),
                /* 5 */ Expr::App(4, 2),
            ],
            defs: vec![0],
        };
        assert_eq!(p.to_string(), "S K :0 @ _0 @ #1 @ }");
        let (root, _, egraph) = program_to_egraph_it(&p);
        let optimized = optimize(egraph, root, "dots/test1.svg"); 
        // println!("{:#?}\n", optimized);
        assert!(optimized == "1");
    }

    #[test]
    fn ref_handling() {
        let p = Program {
            root: 5,
            body: vec![
                /* 0 */ Expr::Prim(Combinator::S),
                /* 1 */ Expr::Prim(Combinator::K),
                /* 2 */ Expr::Int(1),
                /* 3 */ Expr::App(0, 6),
                /* 4 */ Expr::App(3, 6),
                /* 5 */ Expr::App(4, 2),
                /* 6 */ Expr::Ref(0),
            ],
            defs: vec![1],
        };
        // println!("------- refs\n {:#?}", p.to_string());
        let (root, _m, egraph) = program_to_egraph_it(&p);
        // println!("root: {:#?}", root);
        // egraph.classes().for_each(|ec| { println!("{:#?}", ec)});
        let optimized = optimize(egraph, root, "dots/test2.svg"); 
        // println!("{:#?}\n", optimized);
        assert!(optimized == "1");
        // println!("refs --------")
    }

    #[test]
    fn small() {
        let p = Program {
            root: 7,
            body: vec![
                /* 0 */ Expr::Prim(Combinator::A),
                /* 1 */ Expr::Prim(Combinator::K),
                /* 2 */ Expr::Int(1),
                /* 3 */ Expr::Int(2),
                /* 4 */ Expr::App(0, 1),
                /* 5 */ Expr::App(8, 2),
                /* 6 */ Expr::App(5, 3),
                /* 7 */ Expr::App(4, 6),
                /* 8 */ Expr::Ref(0),
            ],
            defs: vec![1],
        };
        println!("SMALL");
        println!("expr: {:#?}", p.to_string());
        let (root, _m, egraph) = program_to_egraph_ph(&p);
        println!("root: {:#?}", root);
        // egraph.classes().for_each(|ec| { println!("{:#?}", ec)});
        let optimized = optimize(egraph, root, "dots/test3.svg"); 
        println!("optimized: {:#?}\n", optimized);
        println!("SMALL END");
    }

    #[test]
    fn small2() {
        let p = Program {
            root: 6,
            body: vec![
                /* 0 */ Expr::Prim(Combinator::A),
                /* 1 */ Expr::Prim(Combinator::I),
                /* 2 */ Expr::Prim(Combinator::Return),
                /* 3 */ Expr::Ref(0),
                /* 4 */ Expr::App(0, 1),
                /* 5 */ Expr::App(2, 3),
                /* 6 */ Expr::App(4, 5),
            ],
            defs: vec![1],
        };
        println!("SMALL");
        println!("expr: {:#?}", p.to_string());
        let (root, _m, egraph) = program_to_egraph_ph(&p);
        println!("root: {:#?}", root);
        // egraph.classes().for_each(|ec| { println!("{:#?}", ec)});
        let optimized = optimize(egraph, root, "dots/test4.svg"); 
        println!("optimized: {:#?}\n", optimized);
        println!("SMALL END");
    }

    #[test]
    fn small3() {
        let p = Program {
            root: 11,
            body: vec![
                /* 0 */ Expr::Prim(Combinator::A),
                /* 1 */ Expr::Prim(Combinator::K),
                /* 2 */ Expr::Int(1),
                /* 3 */ Expr::Int(2),
                /* 4 */ Expr::App(0, 1),
                /* 5 */ Expr::App(12, 2),
                /* 6 */ Expr::App(5, 3),
                /* 7 */ Expr::App(4, 6),
                
                /* 8 */ Expr::Prim(Combinator::Return),
                /* 9 */ Expr::App(0, 7),
                /* 10 */ Expr::App(8, 13),
                /* 11 */ Expr::App(9, 10),

                /* 12 */ Expr::Ref(0),
                /* 13 */ Expr::Ref(1),
            ],
            defs: vec![1, 7],
        };
        println!("SMALL");
        println!("expr: {:#?}", p.to_string());
        let (root, _m, egraph) = program_to_egraph_ph(&p);
        println!("root: {:#?}", root);
        // egraph.classes().for_each(|ec| { println!("{:#?}", ec)});
        let optimized = optimize(egraph, root, "dots/test5.svg"); 
        println!("optimized: {:#?}\n", optimized);
        println!("SMALL END");
    }
}

use egg::*;
use lice::combinator::Combinator;
use lice::file::{Expr, Index, Program};
use ordered_float::OrderedFloat;
use std::collections::{HashMap};
use std::cmp::Ordering;


pub struct MyAstSize;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MyCost {
    Finite(usize),
    Infinite(isize),
}

fn add(a: MyCost, b: MyCost) -> MyCost {
    use MyCost::*;
    match (a, b) {
        (Infinite(wtv), _) => Infinite(wtv),
        (_, Infinite(wtv)) => Infinite(wtv),
        (Finite(x), Finite(y)) => Finite(x + y),
    }
}

fn add1(a: MyCost, b: MyCost) -> MyCost {
    add(add(a, b), MyCost::Finite(1))
}

impl PartialOrd for MyCost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use MyCost::*;
        match (self, other) {
            (Finite(x), Finite(y)) => x.partial_cmp(y),
            (Infinite(_), Infinite(_)) => Some(Ordering::Equal),
            (Infinite(_), Finite(_)) => Some(Ordering::Greater),
            (Finite(_), Infinite(_)) => Some(Ordering::Less),
        }
    }
}

impl CostFunction<SKI> for MyAstSize {
    type Cost = MyCost;

    fn cost<C>(&mut self, t: &SKI, mut costs: C) -> MyCost where C: FnMut(Id) -> MyCost {
        match t {
            SKI::App([f, a]) => {
                let cf = costs(*f);
                let ca = costs(*a);
                let cost = add1(cf, ca);
                // println!("App ({cost:?}): {f:?} ({cf:?}) @ {a:?} ({ca:?})");
                cost
            }
            SKI::Placeholder(id) => MyCost::Infinite(*id as isize),
            SKI::Ref([id]) => {
                add(MyCost::Finite(1), costs(*id))
            }
            _ => {
                // println!("{t:?}: 1");
                MyCost::Finite(1)
            }
        }
    }
}

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct PlaceHolderNum(usize);

impl core::str::FromStr for PlaceHolderNum {
    type Err = &'static str;

    fn from_str(_s: &str) -> Result<Self, Self::Err> {
        Err("Cannot parse PlaceholderNum")
    }
}

impl core::fmt::Display for PlaceHolderNum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PlaceHolder").field(&self.0).finish()
    }
}

impl From<usize> for PlaceHolderNum {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct MyStr(String);

impl core::str::FromStr for MyStr {
    type Err = &'static str;

    fn from_str(_s: &str) -> Result<Self, Self::Err> {
        Err("Cannot parse MyStr")
    }
}

impl core::fmt::Display for MyStr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = &self.0;
        write!(f, "{}", s.as_str())
    }
}

impl core::fmt::Debug for MyStr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = &self.0;
        write!(f, "{}", s.as_str())
    }
}

impl From<String> for MyStr {
    fn from(value: String) -> Self {
        Self(value)
    }
}

define_language! {
    pub enum SKI {
        Prim(Combinator),
        "@" = App([Id; 2]),
        Int(i64),
        Float(OrderedFloat<f64>),
        Array(usize, Vec<Id>),
        "*" = Ref([Id; 1]), // usize),
        String(MyStr),
        Tick(MyStr),
        Ffi(MyStr),
        Unknown(String),
        Placeholder(usize),
    }
}

struct AstSizeHi;
impl CostFunction<SKI> for AstSizeHi {
    type Cost = usize;
    fn cost<C>(&mut self, enode: &SKI, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let node_cost = match enode {
            SKI::Placeholder(_) => usize::MAX,
            SKI::Ref(_) => usize::MAX,
            _ => 1,
        };

        enode.fold(node_cost, |sum, id| {
            sum.saturating_add(costs(id))
        })
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
    add_placeholders(program, &mut idx_eid_map, &mut egraph);
    fill_placeholders(program, &mut idx_eid_map, &mut egraph);

    let root = match idx_eid_map.get(&program.root) {
        Some(eid) => *eid,
        None => panic!("missing root"),
    };

    eclasses_check(&egraph);
    (root, idx_eid_map, egraph)
}

pub fn optimize(egraph: EGraph<SKI, ()>, root: Id, fname: &str) -> String {
    let runner = Runner::<SKI, ()>::default()
        .with_egraph(egraph)
        .run(&ski_reductions());

    runner.egraph.dot().to_svg(fname).unwrap();
    eclasses_check(&runner.egraph);

    let extractor = Extractor::new(&runner.egraph, MyAstSize);
    let (cost, best) = extractor.find_best(root);
    
    
    println!("best: {best:?}");
    println!("expr: {:#?} \n cost: {:#?}", best.to_string(), cost);
    
    let mut i = true;
    // println!("{{ ");
    for ec in runner.egraph.classes() {
        if !i {
            // print!(", ");
        }
        i = false;
        
        // print!("\"eclass {:04}\": {{ ", ec.id);

        // print!("\"nodes\": [");
        let mut j = true;
        for node in &ec.nodes {
            if !j {
                // print!(", ");
            }
            j = false;
            // print!("\"{node:?}\"");
        }
        // print!(" ], ");
        // print!("\"best cost\": \"{:?}\", ", extractor.find_best_cost(ec.id));
        // print!("\"Best node\": \"{:?}\" ", extractor.find_best_node(ec.id));
        // println!("}}");
    }
    // println!("\n\n");

    best.to_string()
}

pub fn noop(egraph: EGraph<SKI, ()>, root: Id) -> String {
    let runner = Runner::<SKI, ()>::default()
        .with_egraph(egraph)
        .run(&vec![]);

    let extractor = Extractor::new(&runner.egraph, AstSizeHi);
    let (_, best) = extractor.find_best(root);

    runner.egraph.dot().to_svg("dots/foo.svg").unwrap();
    println!("best: {:#?}", best);

    best.to_string()
}

fn add_placeholders(
    program: &Program,
    idx_eid_map: &mut HashMap<Index, Id>,
    egraph: &mut EGraph<SKI, ()>,
) {
    program.body.iter().enumerate().for_each(|(idx, _)| {
        let eid = egraph.add(SKI::Placeholder(idx.into()));
        idx_eid_map.insert(idx, eid);
    })
}

fn fill_placeholders(
    program: &Program,
    idx_eid_map: &mut HashMap<Index, Id>,
    egraph: &mut EGraph<SKI, ()>,
) {
    program.body.iter().enumerate().for_each(|(idx, expr)| {
        let eid = match expr {
            Expr::Prim(comb) => egraph.add(SKI::Prim(*comb)),
            Expr::Int(i) => egraph.add(SKI::Int(*i)),
            Expr::Float(flt) => egraph.add(SKI::Float(OrderedFloat(*flt))),
            Expr::String(s) => egraph.add(SKI::String(MyStr(s.to_string()))),
            Expr::Tick(s) => egraph.add(SKI::Tick(MyStr(s.to_string()))),
            Expr::Ffi(s) => egraph.add(SKI::Ffi(MyStr(s.to_string()))),
            Expr::Ref(lbl) => {
                let ref_idx = program.defs[*lbl];
                let ref_eid = match idx_eid_map.get(&ref_idx) {
                    Some(x) => *x,
                    None => panic!("missing ref"),
                };
                let new_eid = egraph.add(SKI::Ref([ref_eid]));
                egraph.union(new_eid, ref_eid);
                new_eid
            }
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
            }
            Expr::Array(u, arr) => {
                let e_arr: Vec<Id> = arr
                    .iter()
                    .map(|i| {
                        let elmt_eid = match idx_eid_map.get(i) {
                            Some(eid) => *eid,
                            None => panic!("missing placeholder: {:#?}", i),
                        };
                        elmt_eid
                    })
                    .collect();
                egraph.add(SKI::Array(*u, e_arr))
            }
            _ => panic!("unknown expr: {:#?}", expr),
        };

        let placeholder = match idx_eid_map.get(&idx) {
            Some(x) => *x,
            None => panic!("missing placeholder eclass"),
        };

        egraph.union(placeholder, eid);
    })
}

fn eclasses_check(egraph: &EGraph<SKI, ()>) {
    egraph.classes().for_each(|ec| {
        let not_ref_or_placeholder: Vec<&SKI> = ec
            .nodes
            .iter()
            .filter(|en| !matches!(en, SKI::Placeholder(_) | SKI::Ref(_)))
            .collect();
        assert!(!not_ref_or_placeholder.is_empty())
    })
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
        println!("expr: {:#?}\n", p.to_string());
        let (root, _, egraph) = program_to_egraph(&p);
        let optimized = optimize(egraph, root, "dots/test1.svg");
        println!("optimized: {:#?}\n", optimized);
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
        let (root, _m, egraph) = program_to_egraph(&p);
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
        println!("expr: {:#?}", p.to_string());
        let (root, _m, egraph) = program_to_egraph(&p);
        // println!("root: {:#?}", root);
        // egraph.classes().for_each(|ec| { println!("{:#?}", ec)});
        let optimized = optimize(egraph, root, "dots/test3.svg");
        println!("optimized: {:#?}\n", optimized);
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
        println!("expr: {:#?}", p.to_string());
        let (root, _m, egraph) = program_to_egraph(&p);
        // println!("root: {:#?}", root);
        // egraph.classes().for_each(|ec| { println!("{:#?}", ec)});
        let optimized = optimize(egraph, root, "dots/test4.svg");
        println!("optimized: {:#?}\n", optimized);
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
        println!("expr: {:#?}", p.to_string());
        let (root, _m, egraph) = program_to_egraph(&p);
        println!("root: {:#?}", root);
        // egraph.classes().for_each(|ec| { println!("{:#?}", ec)});
        let optimized = optimize(egraph, root, "dots/test5.svg");
        println!("optimized: {:#?}\n", optimized);
    }

    #[test]
    fn cyclic() {
        let p = Program {
            root: 3,
            body: vec![
                /* 0 */ Expr::Prim(Combinator::K),
                /* 1 */ Expr::String("hi".to_string()),
                /* 2 */ Expr::Ref(0),
                /* 3 */ Expr::App(4, 1),
                /* 4 */ Expr::App(0, 2),
            ],
            defs: vec![3],
        };

        println!("expr: {:#?}", p.to_string());
        let (root, _m, egraph) = program_to_egraph(&p);
        println!("root: {:#?}", root);
        // egraph.classes().for_each(|ec| { println!("{:#?}", ec)});
        let optimized = optimize(egraph, root, "dots/test6.svg");
        println!("optimized: {:#?}\n", optimized);
    }
}

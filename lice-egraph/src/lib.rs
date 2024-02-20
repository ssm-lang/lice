use std::collections::HashMap;

use egg;
use egg::*;
use lice::combinator::Combinator;
use lice::file::{Expr, Index, Program};

define_language! {
    enum SKI {
        Comb(Combinator),
        "@" = App([Id; 2]),
        Int(i64),
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

fn program_to_egraph(program: &Program) -> (Id, HashMap<Index, Id>, EGraph<SKI, ()>) {
    let mut egraph = EGraph::<SKI, ()>::default();
    let mut idx_eid_map = HashMap::<Index, Id>::new();
    let root = construct_egraph(program, &mut idx_eid_map, &mut egraph, program.root);
    (root, idx_eid_map, egraph)
}

fn construct_egraph(
    program: &Program,
    idx_eid_map: &mut HashMap<Index, Id>,
    egraph: &mut EGraph<SKI, ()>,
    idx: Index,
) -> Id {
    match &program.body[idx] {
        Expr::App(f, a) => {
            let func_eid = match idx_eid_map.get(f) {
                Some(eid) => *eid,
                None => construct_egraph(program, idx_eid_map, egraph, *f),
            };
            let arg_eid = match idx_eid_map.get(a) {
                Some(eid) => *eid,
                None => construct_egraph(program, idx_eid_map, egraph, *a),
            };
            let app_eid = egraph.add(SKI::App([func_eid, arg_eid]));
            idx_eid_map.insert(idx, app_eid);
            app_eid
        }
        Expr::Prim(comb) => {
            let comb_eid = egraph.add(SKI::Comb(*comb));
            idx_eid_map.insert(idx, comb_eid);
            comb_eid
        }
        Expr::Int(i) => {
            let int_eid = egraph.add(SKI::Int(*i));
            idx_eid_map.insert(idx, int_eid);
            int_eid
        }
        _ => todo!("add more exprs"),
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
    fn program_conversion() {
        let p = Program {
            root: 6,
            body: vec![
                /* 0 */ Expr::Prim(Combinator::S),
                /* 1 */ Expr::Prim(Combinator::K),
                /* 2 */ Expr::Prim(Combinator::K),
                /* 3 */ Expr::Int(1),
                /* 4 */ Expr::App(0, 1),
                /* 5 */ Expr::App(4, 2),
                /* 6 */ Expr::App(5, 3),
            ],
            defs: vec![0],
        };
        assert_eq!(p.to_string(), "S K @ K @ #1 @ }");
        let (root, idx_eid_map_, egraph) = program_to_egraph(&p);
        let runner = Runner::<SKI, ()>::default()
            .with_egraph(egraph)
            .run(&ski_reductions());

        let extractor = Extractor::new(&runner.egraph, AstSize);
        let (_, best) = extractor.find_best(root);
        assert!(best.to_string() == "1");
    }
}

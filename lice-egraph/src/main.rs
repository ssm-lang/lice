use egg::*;
use lice::tag::Turner;

define_language! {
    enum SKI {
        Combinator(Turner),
        "@" = App([Id; 2]),
        Symbol(Symbol),
    }
}

fn ski_reductions() -> Vec<Rewrite<SKI, ()>> {
    vec![
        rewrite!("S"; "(@ (@ (@ S ?f) ?g) ?x)" => "(@ (@ ?f ?x) (@ ?g ?x))"),
        rewrite!("K"; "(@ (@ K ?f) ?g)" => "?f"),
        rewrite!("I"; "(@ I ?x)" => "?x"),
        rewrite!("B"; "(@ (@ (@ B ?f) ?g) ?x)" => "(@ ?f (@ ?g ?x))"),
        rewrite!("C"; "(@ (@ (@ C ?f) ?x) ?y)" => "(@ (@ ?f ?y) ?x)"),
        rewrite!("A"; "(@ (@ A ?x) ?y)" => "?y"),
        // rewrite!("Y"; "(@ I ?x)" => "?x"), TODO: how to handle recursion?
        rewrite!("S'"; "(@ (@ (@ (@ SS ?c) ?f) ?g) ?x)" => "(@ (@ ?c (@ ?f ?x)) (@ ?g ?x))"),
        rewrite!("B'"; "(@ (@ (@ (@ BB ?c) ?f) ?g) ?x)" => "(@ (@ ?c ?f) (@ ?g ?x))"),
        rewrite!("C'"; "(@ (@ (@ (@ CC ?c) ?f) ?x) ?y)" => "(@ (@ ?c (@ ?f ?y)) ?x)"),
        rewrite!("P"; "(@ (@ (@ P ?x) ?y) ?f)" => "(@ (@ ?f ?x) ?y)"),
        rewrite!("R"; "(@ (@ (@ R ?y) ?f) ?x)" => "(@ (@ ?f ?x) ?y)"),
        rewrite!("O"; "(@ (@ (@ (@ O ?x) ?y) ?z) ?f)" => "(@ (@ ?f ?x) ?y)"),
        rewrite!("U"; "(@ (@ U ?x) ?f)" => "(@ ?f ?x)"),
        rewrite!("Z"; "(@ (@ (@ Z ?f) ?x) ?y)" => "(@ ?f ?x)"),
        rewrite!("K2"; "(@ (@ (@ K2 ?x) ?y) ?z)" => "?x"),
        rewrite!("K3"; "(@ (@ (@ (@ K3 ?x) ?y) ?z) ?w)" => "?x"),
        rewrite!("K4"; "(@ (@ (@ (@ (@ K4 ?x) ?y) ?z) ?w) ?v)" => "?x"),
        rewrite!("CCB"; "(@ (@ (@ (@ CCB ?f) ?g) ?x) ?y)" => "(@ (@ ?f ?x) (@ ?g ?y))"),
    ]
}

fn main() {
    let mut egraph = EGraph::<SKI, ()>::default();
    let en1 = egraph.add(SKI::Combinator(Turner::S));
    let en2 = egraph.add(SKI::Combinator(Turner::K));
    let en3 = egraph.add(SKI::Combinator(Turner::K));
    let en4 = egraph.add(SKI::Symbol("x".into()));
    let app1 = egraph.add(SKI::App([en1, en2]));
    let app2 = egraph.add(SKI::App([app1, en3]));
    let app3 = egraph.add(SKI::App([app2, en4]));

    let runner = Runner::<SKI, ()>::default()
        .with_egraph(egraph)
        .with_hook(|runner| {
            println!("Egraph is this big: {}", runner.egraph.total_size());
            Ok(())
        })
        .run(&ski_reductions());

    // use an Extractor to pick the best element of the root eclass
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (best_cost, best) = extractor.find_best(app3);
    println!("{}", &runner.egraph.number_of_classes());
    println!("Simplified to {} with cost {}", best, best_cost);
    best.to_string();
}

#[test]
fn reduce_skk_to_i() {
    let mut egraph = EGraph::<SKI, ()>::default();
    let en1 = egraph.add(SKI::Combinator(Turner::S));
    let en2 = egraph.add(SKI::Combinator(Turner::K));
    let en3 = egraph.add(SKI::Combinator(Turner::K));
    let en4 = egraph.add(SKI::Symbol("x".into()));
    let app1 = egraph.add(SKI::App([en1, en2]));
    let app2 = egraph.add(SKI::App([app1, en3]));
    let app3 = egraph.add(SKI::App([app2, en4]));

    let runner = Runner::<SKI, ()>::default()
        .with_egraph(egraph)
        .run(&ski_reductions());

    // use an Extractor to pick the best element of the root eclass
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (_, best) = extractor.find_best(app3);
    assert!(best.to_string() == "x");
}

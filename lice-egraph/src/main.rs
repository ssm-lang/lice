use clap::Parser;
use std::{fs::File, io::Read, path::PathBuf, process, str::FromStr};
use lice::file::CombFile;
use lice_egraph::{program_to_egraph_hi, program_to_egraph_it, program_to_egraph_ph, optimize, noop};
use log::error;

#[derive(Parser, Debug)]
struct Cli {
    filename: PathBuf,
}

fn main() {
    let args = Cli::parse();

    let Ok(mut f) = File::open(&args.filename) else {
        error!("No such file or directory: {}", args.filename.to_string_lossy());
        process::exit(1);
    };
    let mut buf = String::new();
    f.read_to_string(&mut buf).unwrap_or_else(|e| {
        error!("{e}");
        process::exit(1);
    });
    let c = CombFile::from_str(&buf).unwrap_or_else(|e| {
        error!("{e}");
        process::exit(1);
    });

    // c.program.
    // println!("{:#?}\n", &c.program);
    
    let (root, mp, egraph) = program_to_egraph_ph(&c.program);
    let optimized = optimize(egraph, root, "dots/main.svg"); 
    println!("{:#?}\n len: {:#?}", optimized, optimized.len());
    
    let (root2, _, egraph2) = program_to_egraph_ph(&c.program);
    let noop = noop(egraph2, root2); 
    println!("{:#?}\n len: {:#?}", noop, noop.len());

    // println!("{:#?}", mp[&c.program.defs[42]]);
}



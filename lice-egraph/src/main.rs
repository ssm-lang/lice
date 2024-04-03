use clap::Parser;
use std::{fs::File, io::Read, path::PathBuf, process, str::FromStr};
use lice::file::CombFile;
use lice_egraph::{program_to_egraph, optimize, noop};
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
    
    // println!("{:#?}\n", &c.program);

    let (root, _, egraph) = program_to_egraph(&c.program);
    let optimized = optimize(egraph, root); 

    let (root2, _, egraph2) = program_to_egraph(&c.program);
    let noop = noop(egraph2, root2); 
    // println!("{:#?}\n", optimized);
    println!("{:#?}", noop);
}


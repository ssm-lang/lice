use clap::Parser;
use lice::file::CombFile;
use lice::combinator::Combinator;
use lice::file::{Expr, Index, Program};
use lice_egraph::{noop, optimize, program_to_egraph};
use log::error;
use std::{fs::File, io::Read, path::PathBuf, process, str::FromStr};

#[derive(Parser, Debug)]
struct Cli {
    filename: PathBuf,
}

fn main() {
    let args = Cli::parse();

    let Ok(mut f) = File::open(&args.filename) else {
        error!(
            "No such file or directory: {}",
            args.filename.to_string_lossy()
        );
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

    let (root, mp, egraph) = program_to_egraph(&c.program);
    let optimized = optimize(egraph, root, "dots/main.svg");
}

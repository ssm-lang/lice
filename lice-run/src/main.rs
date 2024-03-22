use clap::Parser;
use lice::{eval::VM, file::CombFile};
use log::error;
use std::{fs::File, io::Read, path::PathBuf, process};

#[derive(Parser, Debug)]
struct Cli {
    filename: PathBuf,
}

fn main() {
    env_logger::init();
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
    let c: CombFile = buf.parse().unwrap_or_else(|e| {
        error!("{e}");
        process::exit(1);
    });

    let mut vm = VM::from(c.program);
    println!("constructed VM");

    loop {
        vm.step();
        println!("stepped");
    }
}

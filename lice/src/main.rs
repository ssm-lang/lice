use clap::Parser;
use lice::{
    combinator::Combinator,
    eval::{VMError, VM},
    file::CombFile,
};
use std::{fs::File, io::Read, path::PathBuf, process};
use tracing_subscriber::{fmt::format::FmtSpan, EnvFilter};

#[derive(Parser, Debug)]
struct Cli {
    filename: PathBuf,
}

fn main() {
    tracing_subscriber::fmt()
        .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let args = Cli::parse();
    let Ok(mut f) = File::open(&args.filename) else {
        tracing::error!(
            "No such file or directory: {}",
            args.filename.to_string_lossy()
        );
        process::exit(1);
    };
    let mut buf = String::new();
    f.read_to_string(&mut buf).unwrap_or_else(|e| {
        tracing::error!("{e}");
        process::exit(1);
    });
    let c: CombFile = buf.parse().unwrap_or_else(|e| {
        tracing::error!("{e}");
        process::exit(1);
    });

    let mut vm = VM::from(c.program);
    tracing::info!("VM constructed");

    let mut i = 0;
    loop {
        tracing::info!("VM step {i}");
        match vm.step() {
            Ok(_) => i += 1,
            Err(VMError::IOTerminated(Combinator::UNIT)) => break,
            Err(e) => panic!("{e}"),
        }
    }
}

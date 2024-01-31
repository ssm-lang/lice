mod app;
mod gui;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use crate::app::CombApp;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    cmd: Commands,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Subcommand, Debug, Clone)]
enum Commands {
    /// View combinator graph in graphical UI
    Graph {
        #[arg(value_name = "input.comb")]
        /// Combinator file path
        filename: PathBuf,
    },
    // Other commands: strings, redexes, dot
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::init();
    let args = Cli::parse();
    match args.cmd {
        Commands::Graph { filename } => CombApp::run(filename),
    }
}

// When compiling to web using trunk:
#[cfg(target_arch = "wasm32")]
fn main() {
    // Redirect `log` message to `console.log` and friends:
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();
    CombApp::run();
}

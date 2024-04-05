use std::env;
use std::path::PathBuf;

fn main() {
    cc::Build::new().file("bfile.c").compile("bfile");
    println!("cargo:rustc-link-lib=bfile");

    bindgen::Builder::default()
        .header("bfile.c") // it's not a header, but it's skibidi C, so it's ok
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new())) // invalidate the crate whenever included header files changed
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(PathBuf::from(env::var("OUT_DIR").unwrap()).join("bfile.rs"))
        .expect("Couldn't write bfile bindings!");
}

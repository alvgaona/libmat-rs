use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let libmat_dir = Path::new("vendor/libmat");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mat_c = out_dir.join("mat.c");

    fs::write(&mat_c, "#define MAT_IMPLEMENTATION\n#include \"mat.h\"\n")
        .expect("Failed to create mat.c");

    println!("cargo:rerun-if-changed=vendor/libmat/mat.h");

    cc::Build::new()
        .file(&mat_c)
        .include(libmat_dir)
        .compile("mat");
}

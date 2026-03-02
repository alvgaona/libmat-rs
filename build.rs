use std::fs;
use std::path::Path;

fn main() {
    let libmat_dir = Path::new("vendor/libmat");

    if !libmat_dir.join("mat.h").exists() {
        panic!(
            "libmat not found. Clone it first:\n\
             git clone https://github.com/alvgaona/libmat vendor/libmat"
        );
    }

    // Generate mat.c if it doesn't exist (libmat is header-only)
    let mat_c = libmat_dir.join("mat.c");
    if !mat_c.exists() {
        fs::write(&mat_c, "#define MAT_IMPLEMENTATION\n#include \"mat.h\"\n")
            .expect("Failed to create mat.c");
    }

    println!("cargo:rerun-if-changed=vendor/libmat/mat.h");

    cc::Build::new()
        .file(&mat_c)
        .include(libmat_dir)
        .compile("mat");
}

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    println!("cargo:rerun-if-changed=cuda/kernels.cu");

    cc::Build::new()
        .cuda(true)
        .file("cuda/kernels.cu")
        .flag("-ccbin=/usr/bin/g++-12")
        .flag("-allow-unsupported-compiler")
        .flag("-O3")
        .flag("-std=c++17")
        .compile("wavelet_cuda");

    println!("cargo:rustc-link-lib=dylib=cudart");
}

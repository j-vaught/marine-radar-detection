use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RADAR_CUDA_ARCH");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    println!("cargo:rerun-if-changed=cuda/kernels.cu");

    let mut build = cc::Build::new();
    build
        .cuda(true)
        .file("cuda/kernels.cu")
        .flag("-ccbin=/usr/bin/g++-12")
        .flag("-allow-unsupported-compiler")
        .flag("-O3")
        .flag("-std=c++17");
    if let Ok(arch) = env::var("RADAR_CUDA_ARCH") {
        build.flag(&format!("-arch={arch}"));
    }
    build.compile("wavelet_cuda");

    println!("cargo:rustc-link-lib=dylib=cudart");
}

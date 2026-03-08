# libmat-rs

[![crates.io](https://img.shields.io/crates/v/libmat-rs)](https://crates.io/crates/libmat-rs)
[![docs.rs](https://docs.rs/libmat-rs/badge.svg)](https://docs.rs/libmat-rs)
[![ci](https://github.com/alvgaona/libmat-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/alvgaona/libmat-rs/actions/workflows/ci.yml)
[![license](https://img.shields.io/crates/l/libmat-rs)](LICENSE)

Rust bindings for [libmat](https://github.com/alvgaona/libmat), an stb-style single-header
linear algebra library in pure C.

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
libmat-rs = "0.1"
```

```rust
use libmat_rs::Mat;

let a = Mat::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
let b = Mat::eye(2);
let c = a.mul(&b);

let eig = a.eigen_sym();
```

Matrices use **column-major** storage (BLAS-compatible).

## API

| Function | Description |
|----------|-------------|
| `Mat::new(rows, cols)` | Zero-initialized matrix |
| `Mat::from_slice(rows, cols, data)` | Matrix from column-major slice |
| `Mat::eye(dim)` | Identity matrix |
| `mat.mul(other)` | Matrix multiplication |
| `mat.add(other)` | Element-wise addition |
| `mat.at(row, col)` | Element access |
| `mat.eigvals()` | Eigenvalues (general) |
| `mat.eigvals_sym()` | Eigenvalues (symmetric) |
| `mat.eigen()` | Eigendecomposition (general) |
| `mat.eigen_sym()` | Eigendecomposition (symmetric) |

## Coverage

This crate exposes a subset of libmat's functionality. The underlying C library supports many
more operations (SVD, Cholesky, QR, LU, solvers, norms, SIMD kernels, etc.) that are not yet
wrapped. Contributions are welcome.

## License

Apache-2.0

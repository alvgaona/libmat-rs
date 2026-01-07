# libmat-rs

Rust bindings for [libmat](https://github.com/alvgaona/libmat).

## Development

```bash
git clone https://github.com/alvgaona/libmat vendor/libmat
cargo build
cargo test
```

## Usage

```rust
use libmat::Mat;

let a = Mat::from_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
let b = Mat::eye(2);
let c = a.mul(&b);
c.print();
```

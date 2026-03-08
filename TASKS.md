# Tasks

## Implement `Send` and `Sync` for `Mat`

`Mat` wraps `*mut ffi::Mat`, and raw pointers are not `Send`/`Sync` by default. The underlying C
memory is a plain heap allocation with no thread-local state, no global mutexes, and no thread
affinity — just a `(rows, cols, data*)` triple. Each `Mat` exclusively owns its allocation and
`Drop` frees it exactly once.

- [ ] Add `unsafe impl Send for Mat {}` and `unsafe impl Sync for Mat {}` in `src/lib.rs`
- [ ] Document the safety invariant in a comment above the impls
- [ ] Add a test that moves `Mat` across threads (`std::thread::spawn`)

## Automate `mat.h` updates from libmat releases

Currently `vendor/libmat/mat.h` is manually downloaded from the libmat GitHub release and committed.
Add a GitHub workflow that watches for new libmat releases and opens a PR to update `mat.h`.

- [ ] Create a scheduled workflow (e.g. weekly) or use `repository_dispatch`
- [ ] Download `mat.h` from the latest libmat GitHub release
- [ ] Open a PR if the file changed

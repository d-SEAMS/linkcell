linkcell depends on minimage with `default-features = false` and exposes
optional `minimage-capi` when callers need minimage's C ABI; the default
build no longer pulls minimage's C ABI (`minimage_*`) into `liblinkcell`.

linkcell depends on minimage with `default-features = false` and exposes
optional `minimage-capi` when callers need minimage's C ABI; the default
build no longer pulls `mi_*` symbols into `liblinkcell`.
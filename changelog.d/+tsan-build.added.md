`LINKCELL_SANITIZE=thread` makes the Meson build rebuild the crate and std with nightly `-Zsanitizer=thread`, so a C++ ThreadSanitizer link sees the Rayon joins.

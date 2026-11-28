Add an ABI stamp and a capability query to the C ABI, for a host that loads the library at run time.
`linkcell.h` defines `LC_ABI_MAJOR`, `LC_ABI_MINOR`, and `LC_ABI_LAYOUT_REVISION`; `lc_abi_stamp` returns the library's, and `lc_abi_compatible` says whether the library serves a host compiled against a given stamp.
`lc_capabilities` fills a size-prefixed `lc_caps` with what the library supports.
`lc_row_ops_get` fills a size-prefixed `lc_row_ops` with the row-grid entry points as function pointers, a null member being an operation the library lacks, so such a host resolves three symbols by name and takes the rest from the table.
The C++ header adds `header_abi`, `library_abi`, `abi_compatible`, and `capabilities`.
Meson provides `linkcell-headers`, the include directory with nothing to link, and `packaging/linkcell-headers/` is the overlay for a wrap that needs no Rust toolchain; an installed CMake package provides `linkcell::headers`.

# A host with a C compiler and no Rust toolchain. It takes the linkcell
# declarations from the headers-only subproject and loads the library at
# run time. scripts/check-headers-only.sh assembles the subproject the
# way the wrap does and configures this with no cargo on PATH.
project('linkcell-headers-consumer', 'c', default_options: ['c_std=c11'])

linkcell_headers = dependency('linkcell-headers')
cc = meson.get_compiler('c')
executable(
  'row_grid_host',
  'row_grid.c',
  c_args: ['-D_DEFAULT_SOURCE'],
  dependencies: [linkcell_headers, cc.find_library('dl', required: false)],
)

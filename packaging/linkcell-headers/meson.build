# The linkcell headers as a Meson dependency, with no Rust toolchain.
#
# A host that loads liblinkcell at run time needs the declarations and
# nothing to link. Its wrap fetches the linkcell sources and lays this
# file over their meson.build (patch_directory), so configuring the
# subproject never looks for cargo. See docs/how-to/load-at-run-time.md.
project(
  'linkcell-headers',
  'c',
  version: '0.6.0',
  license: 'MIT',
  meson_version: '>=1.2.0',
)

linkcell_headers_dep = declare_dependency(
  include_directories: include_directories('include'),
  version: meson.project_version(),
)
meson.override_dependency('linkcell-headers', linkcell_headers_dep)

if not meson.is_subproject()
  install_headers('include/linkcell.h', 'include/linkcell.hpp')
endif

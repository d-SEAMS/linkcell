project(
  'linkcell',
  ['c', 'cpp'],
  # Keep in lockstep with Cargo.toml [package].version (source of truth).
  version: '0.4.2',
  meson_version: '>=1.2.0',
  license: 'MIT',
  default_options: [
    'buildtype=debugoptimized',
    'warning_level=2',
    'c_std=c11',
    'cpp_std=c++17',
  ],
)

cc = meson.get_compiler('c')
cargo_prog = find_program('cargo', required: true)
py = find_program('python3', 'python')
_install = not meson.is_subproject()
_msproot = meson.project_source_root()
_mbproot = meson.project_build_root()

if host_machine.system() == 'windows'
  cargo_shared = 'linkcell.dll'
  cargo_static = 'linkcell.lib'
  cargo_implib = 'linkcell.dll.lib'
else
  _so = host_machine.system() == 'darwin' ? '.dylib' : '.so'
  cargo_shared = 'liblinkcell' + _so
  cargo_static = 'liblinkcell.a'
endif

_profile = get_option('buildtype') == 'debug' ? 'debug' : 'release'

_cargo_outputs = [cargo_shared, cargo_static]
_cargo_install_dirs = [get_option('libdir'), get_option('libdir')]
_cargo_command = [
  py,
  files('scripts/copy-cargo-artifacts.py'),
  '--cargo', cargo_prog.full_path(),
  '--source', _msproot,
  '--target', _mbproot / 'cargo-target',
  '--profile', _profile,
  '--shared', cargo_shared,
  '--static', cargo_static,
  '--out-shared', '@OUTPUT0@',
  '--out-static', '@OUTPUT1@',
  '--depfile', '@DEPFILE@',
  # The cc crate compiles the OpenMP region with this compiler, so its
  # runtime matches the one the openmp dependency below links.
  '--cc', ' '.join(cc.cmd_array()),
]
if host_machine.system() == 'windows'
  _cargo_outputs += [cargo_implib]
  _cargo_install_dirs = [get_option('bindir'), get_option('libdir'),
                         get_option('libdir')]
  _cargo_command += [
    '--implib', cargo_implib,
    '--out-implib', '@OUTPUT2@',
  ]
endif

# cargo rustc emits both crate types; the target lists both so Meson
# does not pretend the static archive is the only artifact.
# The crate sources come from Cargo's dep-info through the depfile.
linkcell_cargo = custom_target(
  'linkcell-cargo',
  input: files(
    'Cargo.toml',
    'Cargo.lock',
    'scripts/copy-cargo-artifacts.py',
  ),
  output: _cargo_outputs,
  depfile: 'linkcell-cargo.d',
  command: _cargo_command,
  build_by_default: true,
  install: _install,
  install_dir: _cargo_install_dirs,
)

rust_sys = [dependency('threads')]
if host_machine.system() == 'linux'
  # libgomp under GCC, libomp under Clang: the runtime of the compiler
  # that builds the k-nearest region. It is linked by name, as in
  # build.rs and CMake, so linkcell.pc names it and a consumer's own
  # sources are not compiled with -fopenmp.
  _omp_lib = cc.get_id() == 'clang' ? 'omp' : 'gomp'
  if not cc.has_header('omp.h', args: ['-fopenmp'])
    error('the k-nearest region needs omp.h from ' + cc.get_id() + ' -fopenmp')
  endif
  rust_sys += [
    cc.find_library('dl', required: true),
    cc.find_library('m', required: true),
    cc.find_library(_omp_lib, required: true),
  ]
elif host_machine.system() == 'darwin'
  rust_sys += [cc.find_library('m', required: true)]
elif host_machine.system() == 'windows'
  rust_sys += [
    cc.find_library('ws2_32', required: true),
    cc.find_library('userenv', required: true),
    cc.find_library('bcrypt', required: true),
    cc.find_library('ntdll', required: true),
  ]
endif

# link_args rather than link_with: Meson promotes installed static
# consumers to link_whole, and link_whole rejects custom_targets.
# Do not attach the cargo outputs as sources: a static consumer then
# passes liblinkcell.so to `ar` ("file format not recognized").
linkcell_built_h = custom_target(
  'linkcell-built-h',
  input: linkcell_cargo,
  output: 'linkcell_built.h',
  command: [
    py,
    '-c',
    'from pathlib import Path; Path("@OUTPUT@").write_text("#pragma once\\n")',
  ],
  build_by_default: true,
)
# A wrap consumer builds yodaLib static and then a Python/Lua module.
# That consumer's default_library does not change this option, so a
# shared default here would leave the module needing liblinkcell.so
# at import time. Subprojects always take the static archive.
_use_static = meson.is_subproject() or get_option('default_library') == 'static'
if _use_static
  _link_idx = 1
elif host_machine.system() == 'windows'
  _link_idx = 2
else
  _link_idx = 0
endif

_gpu_opt = get_option('with_gpulite')
gpulite_dep = dependency('gpulite', required: false,
                         fallback: ['gpulite', 'gpulite_dep'])
if _gpu_opt.enabled() and not gpulite_dep.found()
  error('with_gpulite=enabled but gpulite was not found')
endif
_gpu_on = gpulite_dep.found() and not _gpu_opt.disabled()
_gpu_deps = []
_gpu_args = []
_gpu_src = []
if _gpu_on
  _gpu_args += ['-DLINKCELL_HAS_GPULITE=1']
  if host_machine.system() == 'windows'
    linkcell_gpu = static_library(
      'linkcell_gpu',
      ['src/gpu/knearest.cpp', 'src/gpu/capi.cpp', 'src/gpu/partition.cpp'],
      include_directories: include_directories('include'),
      dependencies: [gpulite_dep],
      cpp_args: _gpu_args,
      name_prefix: '',
      name_suffix: 'lib',
      install: _install,
    )
  else
    linkcell_gpu = static_library(
      'linkcell_gpu',
      ['src/gpu/knearest.cpp', 'src/gpu/capi.cpp', 'src/gpu/partition.cpp'],
      include_directories: include_directories('include'),
      dependencies: [gpulite_dep],
      cpp_args: _gpu_args,
      install: _install,
    )
  endif
  linkcell_gpu_dep = declare_dependency(
    link_with: linkcell_gpu,
    link_args: [linkcell_cargo[_link_idx].full_path()],
    sources: linkcell_built_h,
    include_directories: include_directories('include'),
    compile_args: _gpu_args,
    dependencies: [gpulite_dep] + rust_sys)
  meson.override_dependency('linkcell-gpu', linkcell_gpu_dep)
  _gpu_deps += [linkcell_gpu_dep]
endif

linkcell_dep = declare_dependency(
  link_args: [linkcell_cargo[_link_idx].full_path()],
  sources: linkcell_built_h,
  include_directories: include_directories('include'),
  dependencies: rust_sys + _gpu_deps,
  compile_args: _gpu_args,
  version: meson.project_version(),
)
meson.override_dependency('linkcell', linkcell_dep)

if _install
  install_headers('include/linkcell.h', 'include/linkcell.hpp')
  if _gpu_on
    install_headers('include/linkcell_gpu.h', 'include/linkcell_gpu.hpp')
  endif
  pkg = import('pkgconfig')
  pkg.generate(
    name: 'linkcell',
    filebase: 'linkcell',
    description: 'Periodic linked-cell k-nearest neighbour search',
    version: meson.project_version(),
    url: 'https://github.com/d-SEAMS/linkcell',
    # Raw `-lfoo` does not get `-L${libdir}`; wrap consumers then fail
    # at link with "cannot find -llinkcell".
    libraries: ['-L${libdir}', '-llinkcell'],
    libraries_private: rust_sys,
    subdirs: '.',
  )
  if _gpu_on
    pkg.generate(
      name: 'linkcell-gpu',
      filebase: 'linkcell-gpu',
      description: 'Device periodic k-nearest neighbour search',
      version: meson.project_version(),
      url: 'https://github.com/d-SEAMS/linkcell',
      requires: ['linkcell'],
      libraries: ['-L${libdir}', '-llinkcell_gpu'] + rust_sys,
      extra_cflags: ['-DLINKCELL_HAS_GPULITE=1'],
      subdirs: '.',
    )
  endif
endif

summary({
  'Device k-nearest (gpulite)': _gpu_on,
}, section: 'Configuration')

if get_option('with_tests')
  test(
    'cargo-tests',
    cargo_prog,
    args: ['test', '--manifest-path', _msproot / 'Cargo.toml'],
    timeout: 300,
  )
endif

if get_option('with_examples')
  subdir('examples')
endif

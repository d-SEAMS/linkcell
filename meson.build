project(
  'linkcell',
  ['c', 'cpp'],
  # Keep in lockstep with Cargo.toml [package].version (source of truth).
  version: '0.6.0',
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
_omp_supported = host_machine.system() == 'linux' or host_machine.system() == 'darwin'
_use_openmp = get_option('use_openmp') and _omp_supported
if get_option('use_openmp') and not _omp_supported
  warning('use_openmp=true is not supported on ' + host_machine.system() + '; building without OpenMP')
endif

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
  '--openmp', _use_openmp ? 'true' : 'false',
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

rust_sys = [dependency('threads')]
_omp_c_args = []
_omp_dirs = []
if host_machine.system() == 'linux'
  rust_sys += [
    cc.find_library('dl', required: true),
    cc.find_library('m', required: true),
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

if _use_openmp
  # Linux uses libgomp under GCC and libomp under Clang. Darwin uses libomp.
  # Keep the runtime as a link dependency without compiling consumer sources
  # with -fopenmp.
  if host_machine.system() == 'darwin'
    _omp_lib = 'omp'
  else
    _omp_lib = cc.get_id() == 'clang' ? 'omp' : 'gomp'
  endif
  _fs = import('fs')
  _omp_suffix = host_machine.system() == 'darwin' ? '.dylib' : '.so'
  _omp_file = 'lib' + _omp_lib + _omp_suffix
  _cc0 = cc.cmd_array()[0]

  # Homebrew's libomp is keg-only. Check LIBOMP_PREFIX first, then ask brew,
  # then check the standard Apple Silicon and Intel prefixes in case brew is
  # not on PATH.
  if host_machine.system() == 'darwin' and cc.get_id() == 'clang'
    _brew_prefixes = []
    _libomp_env = run_command(
      py, '-c', 'import os; print(os.environ.get("LIBOMP_PREFIX", ""))',
      check: false,
    ).stdout().strip()
    if _libomp_env != ''
      _brew_prefixes += _libomp_env
    endif
    _brew = find_program('brew', required: false)
    if _brew.found()
      _brew_libomp = run_command(_brew, '--prefix', 'libomp', check: false)
      if _brew_libomp.returncode() == 0
        _brew_prefixes += _brew_libomp.stdout().strip()
      endif
    endif
    _brew_prefixes += [
      '/opt/homebrew/opt/libomp',
      '/usr/local/opt/libomp',
    ]
    foreach _prefix : _brew_prefixes
      _libdir = _prefix / 'lib'
      _includedir = _prefix / 'include'
      if _fs.is_file(_libdir / _omp_file)
        _omp_dirs += _libdir
      endif
      if _fs.is_file(_includedir / 'omp.h')
        _omp_c_args += '-I' + _includedir
      endif
    endforeach
  endif

  # Clang on Debian/Ubuntu reports `-print-file-name=libomp.so` as the bare
  # name. The linker name is next to the resource dir; only the SONAME is
  # on the default path. LIBRARY_PATH stays ahead of that fallback.
  _printed = run_command(_cc0, '-print-file-name=' + _omp_file, check: false)
  if _printed.returncode() == 0 and _fs.is_file(_printed.stdout().strip())
    _omp_dirs += _fs.parent(_printed.stdout().strip())
  endif
  _lp = run_command(
    py, '-c', 'import os; print(os.environ.get("LIBRARY_PATH", ""))',
    check: false,
  ).stdout().strip()
  foreach _dir : _lp.split(':')
    if _dir != '' and _fs.is_file(_dir / _omp_file)
      _omp_dirs += _dir
    endif
  endforeach
  _res = run_command(_cc0, '-print-resource-dir', check: false)
  if _res.returncode() == 0 and _res.stdout().strip() != ''
    _resource_dir = _res.stdout().strip()
    _resource_include = _resource_dir / 'include'
    if _fs.is_file(_resource_include / 'omp.h')
      _omp_c_args += '-I' + _resource_include
    endif
    _reslib = _fs.parent(_fs.parent(_resource_dir))
    if _fs.is_file(_reslib / _omp_file)
      _omp_dirs += _reslib
    endif
  endif
  # A separately installed libomp, including Homebrew's keg-only formula,
  # keeps omp.h in include next to its lib directory. Discover that before
  # the mandatory header probe and pass the same path to Cargo's cc crate.
  foreach _dir : _omp_dirs
    _include = _fs.parent(_dir) / 'include'
    if _fs.is_file(_include / 'omp.h')
      _omp_c_args += '-I' + _include
    endif
  endforeach
  _omp_probe_args = _omp_c_args
  if host_machine.system() == 'darwin' and cc.get_id() == 'clang'
    _omp_probe_args = ['-Xpreprocessor', '-fopenmp'] + _omp_probe_args
  else
    _omp_probe_args = ['-fopenmp'] + _omp_probe_args
  endif
  if not cc.has_header('omp.h', args: _omp_probe_args)
    error('the threaded build needs omp.h from ' + cc.get_id() + ' -fopenmp')
  endif
  _omp_dep = cc.find_library(_omp_lib, dirs: _omp_dirs, required: true)
  if host_machine.system() == 'linux'
    # find_library with dirs answers with the file's full path. Link the
    # runtime by name, with -L for the directory that holds it, so
    # linkcell.pc names -lomp or -lgomp and a consumer picks up the
    # runtime's own SONAME.
    _omp_link = ['-l' + _omp_lib]
    if _omp_dirs.length() > 0
      _omp_link = ['-L' + _omp_dirs[0]] + _omp_link
    endif
    rust_sys += [declare_dependency(link_args: _omp_link)]
  else
    # Homebrew's libomp is keg-only and its install name is the full path.
    rust_sys += [_omp_dep]
  endif
endif

# Give Cargo's cc crate Meson's compiler and the proven OpenMP paths, keeping
# the compiled region's runtime aligned with rust_sys.
_cargo_command += ['--cc', ' '.join(cc.cmd_array() + _omp_c_args)]
foreach _dir : _omp_dirs
  _cargo_command += ['--library-path', _dir]
endforeach
if get_option('sanitize') != 'none'
  _cargo_command += ['--sanitize', get_option('sanitize')]
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
  'OpenMP k-nearest region': _use_openmp,
  'Device k-nearest (gpulite)': _gpu_on,
}, section: 'Configuration')

if get_option('with_tests')
  test(
    'cargo-tests',
    cargo_prog,
    args: ['test', '--manifest-path', _msproot / 'Cargo.toml'],
    env: {'LINKCELL_OPENMP': _use_openmp ? '1' : '0'},
    timeout: 300,
  )
endif

if get_option('with_examples')
  subdir('examples')
endif

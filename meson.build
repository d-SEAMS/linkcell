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

# Meson 1.3 hands a subproject its parent's buildtype and drops the
# buildtype=debugoptimized above; Meson 1.12 keeps it. So the same wrap
# consumer got an unoptimized crate on one Meson and an optimized one on
# another. auto builds the release profile in a subproject on every
# Meson, and follows the buildtype at the top level.
_profile = get_option('cargo_profile')
if _profile == 'auto'
  _profile = (get_option('buildtype') == 'debug' and not meson.is_subproject()) ? 'debug' : 'release'
endif
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
#
# A sanitized build carries its own rebuilt std. Linked static, those std
# objects clash with the std of any other Rust staticlib in the same link
# (rust_eh_personality, __rdl_alloc_error_handler); the shared library
# exports only the C API and keeps its std private.
#
# The Rust objects are built with -Zexternal-clangrt, so the library leaves
# every __tsan_* symbol to the final link; each consumer has to link the
# sanitizer runtime, whether or not its own sources are instrumented.
_sanitized = get_option('sanitize') != 'none'
_use_static = (meson.is_subproject() or get_option('default_library') == 'static') and not _sanitized
_sanitize_link = []
if _sanitized
  _sanitize_flag = '-fsanitize=' + get_option('sanitize')
  if not cc.has_multi_link_arguments(_sanitize_flag)
    error('sanitize=@0@ needs a C compiler that links with @1@'.format(get_option('sanitize'), _sanitize_flag))
  endif
  _sanitize_link = [_sanitize_flag, '-Wl,-rpath,' + meson.current_build_dir()]
endif
if _use_static
  _link_idx = 1
elif host_machine.system() == 'windows'
  _link_idx = 2
else
  _link_idx = 0
endif

# linkcell::Error's destructor is the key function. It has to live in one
# shared library even when this project is a subproject or default_library
# is static: a parent such as seams-core builds hidden-visibility DSOs and
# libc++ catches by the typeinfo address. install: true from a subproject
# so the parent prefix ships that library. eh_shared=false links the object
# into each final image; that copy is unique only inside one final link.
_eh_shared = get_option('eh_shared')
if _eh_shared
  _eh_cargs = []
  linkcell_eh = shared_library(
    'linkcell_eh',
    'src/error.cpp',
    include_directories: include_directories('include'),
    cpp_args: ['-DLINKCELL_EH_BUILDING'],
    install: true,
  )
else
  _eh_cargs = ['-DLINKCELL_EH_STATIC']
  linkcell_eh = static_library(
    'linkcell_eh',
    'src/error.cpp',
    include_directories: include_directories('include'),
    cpp_args: ['-DLINKCELL_EH_STATIC'],
    pic: true,
    install: true,
  )
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
    link_with: [linkcell_gpu, linkcell_eh],
    link_args: [linkcell_cargo[_link_idx].full_path()] + _sanitize_link,
    sources: linkcell_built_h,
    include_directories: include_directories('include'),
    compile_args: _gpu_args + _eh_cargs,
    dependencies: [gpulite_dep] + rust_sys)
  meson.override_dependency('linkcell-gpu', linkcell_gpu_dep)
  _gpu_deps += [linkcell_gpu_dep]
endif

linkcell_dep = declare_dependency(
  link_with: linkcell_eh,
  link_args: [linkcell_cargo[_link_idx].full_path()] + _sanitize_link,
  sources: linkcell_built_h,
  include_directories: include_directories('include'),
  dependencies: rust_sys + _gpu_deps,
  compile_args: _gpu_args + _eh_cargs,
  version: meson.project_version(),
)
meson.override_dependency('linkcell', linkcell_dep)

# The headers alone, for a host that loads the library at run time and
# links nothing. packaging/linkcell-headers/meson.build declares the same
# dependency without cargo, for a host that builds no Rust.
linkcell_headers_dep = declare_dependency(
  include_directories: include_directories('include'),
  version: meson.project_version(),
)
meson.override_dependency('linkcell-headers', linkcell_headers_dep)

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
    libraries: ['-L${libdir}', '-llinkcell', '-llinkcell_eh'] + (_sanitized ? [_sanitize_flag] : []),
    extra_cflags: _eh_cargs,
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
  'linkcell::Error typeinfo': _eh_shared ? 'shared linkcell_eh' : 'static, one copy per final link',
}, section: 'Configuration')

if get_option('with_tests')
  nd_capi_smoke = executable(
    'nd_capi_smoke',
    meson.project_source_root() / 'tests/nd_capi_smoke.c',
    dependencies: [linkcell_dep, cc.find_library('m', required: false)],
    build_rpath: meson.project_build_root(),
  )
  _meson_test_env = environment()
  if host_machine.system() == 'linux'
    _meson_test_env.prepend('LD_LIBRARY_PATH', meson.project_build_root())
  elif host_machine.system() == 'darwin'
    _meson_test_env.prepend('DYLD_LIBRARY_PATH', meson.project_build_root())
  elif host_machine.system() == 'windows'
    _meson_test_env.prepend('PATH', meson.project_build_root())
  endif
  test('nd-capi-smoke-c', nd_capi_smoke, env: _meson_test_env)
  # Cargo's test profile compiles unoptimized. The neighbour checks call
  # the dyadic metric often enough that an unoptimized run misses the
  # meson budget on macOS. Opt level 3 keeps debug assertions, and this
  # run does not use the release profile's LTO or single codegen unit.
  test(
    'cargo-tests',
    cargo_prog,
    args: ['test', '--manifest-path', _msproot / 'Cargo.toml'],
    env: {
      'LINKCELL_OPENMP': _use_openmp ? '1' : '0',
      'CARGO_PROFILE_TEST_OPT_LEVEL': '3',
    },
    timeout: 300,
  )
  _cpp = meson.get_compiler('cpp')
  # The row grid from C++, through the wrapper of linkcell.hpp.
  _row_grid_cpp = executable(
    'row_grid_cpp',
    'tests/row_grid.cpp',
    dependencies: [linkcell_dep, dependency('threads')],
    build_rpath: meson.project_build_root(),
    install: false,
  )
  test('row-grid-cpp', _row_grid_cpp, env: _meson_test_env, timeout: 120)
  # The row grid from a C host that has the headers and no library on its
  # link line, and loads the shared library by path.
  if host_machine.system() == 'linux'
    _row_grid_c = executable(
      'row_grid_c',
      'tests/row_grid.c',
      c_args: ['-D_DEFAULT_SOURCE'],
      dependencies: [linkcell_headers_dep, cc.find_library('dl', required: false)],
      install: false,
    )
    test(
      'row-grid-c-headers-only',
      _row_grid_c,
      args: [linkcell_cargo[0].full_path()],
      depends: linkcell_cargo,
      timeout: 120,
    )
  endif
  _eh_static_probe = executable(
    'exception_typeinfo_static',
    files(
      'tests/exception_typeinfo_static.cpp',
      'tests/exception_typeinfo_static_other.cpp',
      'src/error.cpp',
    ),
    include_directories: include_directories('include'),
    cpp_args: ['-DLINKCELL_EH_STATIC'],
    install: false,
  )
  test('exception-typeinfo-static', _eh_static_probe)
  if _eh_shared
    # The owner is the installed linkcell_eh target, not a second copy.
    # MSVC has no -fvisibility; dllimport keeps the typeinfo in the DLL.
    _hidden_vis = ['-fvisibility=hidden', '-fvisibility-inlines-hidden']
    _hidden_args = []
    if _cpp.has_multi_arguments(_hidden_vis)
      _hidden_args = _hidden_vis
    endif
    _eh_dep = declare_dependency(
      link_with: linkcell_eh,
      include_directories: include_directories('include'),
    )
    _eh_hidden_a = shared_library(
      'exception_typeinfo_hidden_a',
      'tests/exception_typeinfo_hidden.cpp',
      dependencies: _eh_dep,
      cpp_args: _hidden_args,
      install: false,
    )
    _eh_hidden_b = shared_library(
      'exception_typeinfo_hidden_b',
      'tests/exception_typeinfo_hidden.cpp',
      dependencies: _eh_dep,
      cpp_args: _hidden_args + ['-DLC_PROBE_B'],
      install: false,
    )
    _eh_probe = executable(
      'exception_typeinfo',
      'tests/exception_typeinfo.cpp',
      dependencies: _eh_dep,
      link_with: [_eh_hidden_a, _eh_hidden_b],
      install: false,
    )
    test('exception-typeinfo', _eh_probe, env: _meson_test_env)
  endif
endif

if get_option('with_examples')
  subdir('examples')
endif

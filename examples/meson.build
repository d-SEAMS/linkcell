two_points_c = executable(
  'two_points',
  'two_points.c',
  dependencies: linkcell_dep,
  build_rpath: meson.project_build_root(),
)

pairs_csr_c = executable(
  'pairs_csr',
  'pairs_csr.c',
  dependencies: [linkcell_dep, cc.find_library('m', required: false)],
  build_rpath: meson.project_build_root(),
)

two_points_cpp = executable(
  'two_points_cpp',
  'two_points.cpp',
  dependencies: linkcell_dep,
  build_rpath: meson.project_build_root(),
)

_test_env = environment()
if host_machine.system() == 'linux'
  _test_env.prepend('LD_LIBRARY_PATH', meson.project_build_root())
elif host_machine.system() == 'darwin'
  _test_env.prepend('DYLD_LIBRARY_PATH', meson.project_build_root())
endif

test('two-points-c', two_points_c, env: _test_env)
test('two-points-cpp', two_points_cpp, env: _test_env)
test('pairs-csr-c', pairs_csr_c, env: _test_env)

if _gpu_on
  gpu_agree = executable(
    'gpu_agree',
    meson.project_source_root() / 'tests/gpu_agree.cpp',
    dependencies: linkcell_gpu_dep,
    include_directories: include_directories('../src/gpu'),
    build_rpath: meson.project_build_root(),
  )
  test('gpu-agree', gpu_agree, env: _test_env)
endif

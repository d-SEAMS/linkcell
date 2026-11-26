cmake_minimum_required(VERSION 3.22)

# Version source of truth: Cargo.toml [package].version.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml" _linkcell_cargo_toml)
set(LINKCELL_VERSION "")
foreach(_line ${_linkcell_cargo_toml})
  if(_line MATCHES "^version = \"([0-9]+\\.[0-9]+\\.[0-9]+.*)\"")
    set(LINKCELL_VERSION "${CMAKE_MATCH_1}")
    break()
  endif()
endforeach()
if(LINKCELL_VERSION STREQUAL "")
  message(FATAL_ERROR "could not parse version from ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml")
endif()
unset(_linkcell_cargo_toml)
unset(_line)

string(REGEX REPLACE "([0-9]*)\\.([0-9]*)\\.([0-9]*).*" "\\1.\\2.\\3"
  LINKCELL_CMAKE_VERSION "${LINKCELL_VERSION}")
project(linkcell VERSION ${LINKCELL_CMAKE_VERSION} LANGUAGES C CXX)
set(PROJECT_VERSION ${LINKCELL_VERSION})

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(Threads REQUIRED)
find_program(CARGO_EXECUTABLE cargo REQUIRED)

option(LINKCELL_BUILD_EXAMPLES "Build C and C++ example programs" ON)
option(LINKCELL_INSTALL "Install headers, libraries, pkg-config, and CMake package" ON)
option(LINKCELL_EH_STATIC
  "Link linkcell::Error's key function into each final image instead of the shared linkcell_eh library"
  OFF)
option(LINKCELL_WITH_GPULITE "Build the device k-nearest C and C++ APIs" ON)
option(LINKCELL_CARGO_VERBOSE "Show Cargo and build-script commands" OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR APPLE)
  set(_linkcell_openmp_supported TRUE)
else()
  set(_linkcell_openmp_supported FALSE)
endif()
option(LINKCELL_USE_OPENMP
  "Compile the fused k-nearest OpenMP region and link the compiler OpenMP runtime"
  ${_linkcell_openmp_supported})

if(LINKCELL_USE_OPENMP AND NOT _linkcell_openmp_supported)
  message(WARNING
    "LINKCELL_USE_OPENMP=ON is not supported on ${CMAKE_SYSTEM_NAME}; "
    "building without OpenMP")
  set(_linkcell_use_openmp FALSE)
elseif(LINKCELL_USE_OPENMP)
  set(_linkcell_use_openmp TRUE)
else()
  set(_linkcell_use_openmp FALSE)
endif()

if(_linkcell_use_openmp)
  set(_linkcell_openmp_env "LINKCELL_OPENMP=1")
else()
  set(_linkcell_openmp_env "LINKCELL_OPENMP=0")
endif()

set(LINKCELL_CARGO_VERBOSITY)
if(LINKCELL_CARGO_VERBOSE)
  set(LINKCELL_CARGO_VERBOSITY -vv)
endif()

if(WIN32)
  set(LINKCELL_SHARED_NAME linkcell.dll)
  set(LINKCELL_IMPLIB_NAME linkcell.dll.lib)
  set(LINKCELL_STATIC_NAME linkcell.lib)
else()
  set(LINKCELL_SHARED_NAME "${CMAKE_SHARED_LIBRARY_PREFIX}linkcell${CMAKE_SHARED_LIBRARY_SUFFIX}")
  set(LINKCELL_STATIC_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}linkcell${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif()
set(LINKCELL_GPU_STATIC_NAME
  "${CMAKE_STATIC_LIBRARY_PREFIX}linkcell_gpu${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(LINKCELL_EH_SHARED_NAME
  "${CMAKE_SHARED_LIBRARY_PREFIX}linkcell_eh${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(LINKCELL_EH_STATIC_NAME
  "${CMAKE_STATIC_LIBRARY_PREFIX}linkcell_eh${CMAKE_STATIC_LIBRARY_SUFFIX}")
if(WIN32)
  set(LINKCELL_EH_IMPLIB_NAME
    "${CMAKE_IMPORT_LIBRARY_PREFIX}linkcell_eh${CMAKE_IMPORT_LIBRARY_SUFFIX}")
else()
  set(LINKCELL_EH_IMPLIB_NAME "")
endif()

set(LINKCELL_CARGO_DIR "${CMAKE_CURRENT_BINARY_DIR}/cargo-target")
string(TOLOWER "${CMAKE_BUILD_TYPE}" _linkcell_build_type)
if(_linkcell_build_type STREQUAL "debug")
  set(LINKCELL_CARGO_PROFILE debug)
  set(LINKCELL_CARGO_ARGS)
else()
  set(LINKCELL_CARGO_PROFILE release)
  set(LINKCELL_CARGO_ARGS --release)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.lock")
  list(APPEND LINKCELL_CARGO_ARGS --locked)
endif()
set(LINKCELL_CARGO_OUT "${LINKCELL_CARGO_DIR}/${LINKCELL_CARGO_PROFILE}")
set(LINKCELL_SHARED "${CMAKE_CURRENT_BINARY_DIR}/${LINKCELL_SHARED_NAME}")
set(LINKCELL_STATIC "${CMAKE_CURRENT_BINARY_DIR}/${LINKCELL_STATIC_NAME}")
if(WIN32)
  set(LINKCELL_IMPLIB "${CMAKE_CURRENT_BINARY_DIR}/${LINKCELL_IMPLIB_NAME}")
endif()

file(GLOB_RECURSE LINKCELL_RUST_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.rs"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)
list(APPEND LINKCELL_RUST_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/build.rs"
  "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
  "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.lock"
)

set(LINKCELL_RUSTC_LINK)
if(APPLE)
  set(LINKCELL_RUSTC_LINK "-Clink-arg=-Wl,-install_name,@rpath/${LINKCELL_SHARED_NAME}")
elseif(UNIX)
  set(LINKCELL_RUSTC_LINK "-Clink-arg=-Wl,-soname,${LINKCELL_SHARED_NAME}")
endif()

set(LINKCELL_CARGO_OUTPUTS "${LINKCELL_SHARED}" "${LINKCELL_STATIC}")
set(LINKCELL_COPY_IMPLIB)
if(WIN32)
  list(APPEND LINKCELL_CARGO_OUTPUTS "${LINKCELL_IMPLIB}")
  set(LINKCELL_COPY_IMPLIB
    COMMAND "${CMAKE_COMMAND}" -E copy
            "${LINKCELL_CARGO_OUT}/${LINKCELL_IMPLIB_NAME}"
            "${LINKCELL_IMPLIB}")
endif()

# The cc crate compiles the OpenMP region with the C compiler CMake uses,
# so build.rs and CMake's OpenMP probe select the same runtime
# (LINKCELL_OMP_LIB below). LINKCELL_OPENMP=0 skips the region so the
# archive carries no OpenMP symbols or runtime dependency.
set(LINKCELL_CARGO_ENV "${CMAKE_COMMAND}" -E env
  "CC=${CMAKE_C_COMPILER}"
  "${_linkcell_openmp_env}")

add_custom_command(
  OUTPUT ${LINKCELL_CARGO_OUTPUTS}
  COMMAND ${LINKCELL_CARGO_ENV} "${CARGO_EXECUTABLE}" ${LINKCELL_CARGO_VERBOSITY}
          rustc --lib ${LINKCELL_CARGO_ARGS}
          --features capi
          --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
          --target-dir "${LINKCELL_CARGO_DIR}"
          -- --crate-type=cdylib --crate-type=staticlib ${LINKCELL_RUSTC_LINK}
  COMMAND "${CMAKE_COMMAND}" -E copy
          "${LINKCELL_CARGO_OUT}/${LINKCELL_SHARED_NAME}"
          "${LINKCELL_SHARED}"
  COMMAND "${CMAKE_COMMAND}" -E copy
          "${LINKCELL_CARGO_OUT}/${LINKCELL_STATIC_NAME}"
          "${LINKCELL_STATIC}"
  ${LINKCELL_COPY_IMPLIB}
  DEPENDS ${LINKCELL_RUST_SOURCES}
  COMMENT "Building liblinkcell via cargo rustc (cdylib + staticlib)"
  VERBATIM
)
add_custom_target(linkcell_cargo ALL DEPENDS ${LINKCELL_CARGO_OUTPUTS})

add_library(linkcell SHARED IMPORTED GLOBAL)
add_library(linkcell::linkcell ALIAS linkcell)
add_dependencies(linkcell linkcell_cargo)
set_target_properties(linkcell PROPERTIES
  IMPORTED_LOCATION "${LINKCELL_SHARED}"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
if(WIN32)
  set_target_properties(linkcell PROPERTIES IMPORTED_IMPLIB "${LINKCELL_IMPLIB}")
endif()

set(LINKCELL_OMP_DIR "")
set(LINKCELL_OMP_LIB "")
if(_linkcell_use_openmp)
  if(APPLE)
    set(LINKCELL_OMP_LIB omp)
    set(_linkcell_omp_prefixes)
    if(DEFINED ENV{LIBOMP_PREFIX} AND NOT "$ENV{LIBOMP_PREFIX}" STREQUAL "")
      list(APPEND _linkcell_omp_prefixes "$ENV{LIBOMP_PREFIX}")
    endif()
    find_program(_linkcell_brew brew)
    if(_linkcell_brew)
      execute_process(
        COMMAND "${_linkcell_brew}" --prefix libomp
        RESULT_VARIABLE _linkcell_brew_result
        OUTPUT_VARIABLE _linkcell_brew_prefix
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      if(_linkcell_brew_result EQUAL 0)
        list(APPEND _linkcell_omp_prefixes "${_linkcell_brew_prefix}")
      endif()
    endif()
    list(APPEND _linkcell_omp_prefixes
      /opt/homebrew/opt/libomp
      /usr/local/opt/libomp
    )
    foreach(_prefix IN LISTS _linkcell_omp_prefixes)
      if(EXISTS "${_prefix}/include/omp.h"
         AND EXISTS "${_prefix}/lib/libomp.dylib")
        set(OpenMP_ROOT "${_prefix}")
        set(LINKCELL_OMP_DIR "${_prefix}/lib")
        set(OpenMP_C_FLAGS
          "-Xpreprocessor -fopenmp -I${_prefix}/include")
        set(OpenMP_CXX_FLAGS
          "-Xpreprocessor -fopenmp -I${_prefix}/include")
        set(OpenMP_C_LIB_NAMES omp)
        set(OpenMP_CXX_LIB_NAMES omp)
        set(OpenMP_omp_LIBRARY "${_prefix}/lib/libomp.dylib")
        break()
      endif()
    endforeach()
    if(NOT LINKCELL_OMP_DIR)
      message(FATAL_ERROR
        "the threaded build needs Homebrew libomp; install it with brew install libomp")
    endif()
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Clang's -fopenmp calls libomp, GCC's calls libgomp. Because Cargo runs
    # with CC=${CMAKE_C_COMPILER}, this matches the compiler build.rs uses
    # for src/knn_region.c.
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
      set(LINKCELL_OMP_LIB omp)
    else()
      set(LINKCELL_OMP_LIB gomp)
    endif()
    # Directory of lib<runtime>.so when it is not on the default linker path.
    # Ubuntu's libomp-dev keeps the linker name beside the Clang resource dir.
    set(_linkcell_omp_file "lib${LINKCELL_OMP_LIB}.so")
    execute_process(
      COMMAND "${CMAKE_C_COMPILER}" "-print-file-name=${_linkcell_omp_file}"
      OUTPUT_VARIABLE _linkcell_omp_printed
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(IS_ABSOLUTE "${_linkcell_omp_printed}" AND EXISTS "${_linkcell_omp_printed}")
      get_filename_component(LINKCELL_OMP_DIR "${_linkcell_omp_printed}" DIRECTORY)
    endif()
    if(NOT LINKCELL_OMP_DIR AND DEFINED ENV{LIBRARY_PATH})
      string(REPLACE ":" ";" _linkcell_lp "$ENV{LIBRARY_PATH}")
      foreach(_dir IN LISTS _linkcell_lp)
        if(EXISTS "${_dir}/${_linkcell_omp_file}")
          set(LINKCELL_OMP_DIR "${_dir}")
          break()
        endif()
      endforeach()
    endif()
    if(NOT LINKCELL_OMP_DIR)
      execute_process(
        COMMAND "${CMAKE_C_COMPILER}" -print-resource-dir
        OUTPUT_VARIABLE _linkcell_resdir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      if(_linkcell_resdir)
        get_filename_component(_linkcell_res_parent "${_linkcell_resdir}" DIRECTORY)
        get_filename_component(_linkcell_res_lib "${_linkcell_res_parent}" DIRECTORY)
        if(EXISTS "${_linkcell_res_lib}/${_linkcell_omp_file}")
          set(LINKCELL_OMP_DIR "${_linkcell_res_lib}")
        endif()
      endif()
    endif()
  endif()
  find_package(OpenMP REQUIRED COMPONENTS C CXX)
  target_link_libraries(linkcell INTERFACE
    OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
endif()

set(_linkcell_sys Threads::Threads)
if(UNIX AND NOT APPLE)
  list(APPEND _linkcell_sys ${CMAKE_DL_LIBS} m)
  if(_linkcell_use_openmp)
    list(APPEND _linkcell_sys OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
  endif()
elseif(APPLE)
  list(APPEND _linkcell_sys m)
  if(_linkcell_use_openmp)
    list(APPEND _linkcell_sys OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
  endif()
elseif(WIN32)
  list(APPEND _linkcell_sys ws2_32 userenv bcrypt ntdll)
endif()

add_library(linkcell_static STATIC IMPORTED GLOBAL)
add_library(linkcell::static ALIAS linkcell_static)
add_dependencies(linkcell_static linkcell_cargo)
set_target_properties(linkcell_static PROPERTIES
  IMPORTED_LOCATION "${LINKCELL_STATIC}"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
  INTERFACE_LINK_LIBRARIES "${_linkcell_sys}"
)
if(LINKCELL_OMP_DIR)
  set_property(TARGET linkcell_static APPEND PROPERTY
    INTERFACE_LINK_DIRECTORIES "${LINKCELL_OMP_DIR}")
endif()

# Key function for linkcell::Error. Shared, so two consumer DSOs import one
# typeinfo. LINKCELL_EH_STATIC keeps the fully static link: one copy in
# each final image, not across DSOs.
if(LINKCELL_EH_STATIC)
  add_library(linkcell_eh STATIC src/error.cpp)
  target_compile_definitions(linkcell_eh PUBLIC LINKCELL_EH_STATIC)
else()
  add_library(linkcell_eh SHARED src/error.cpp)
  target_compile_definitions(linkcell_eh PRIVATE LINKCELL_EH_BUILDING)
endif()
add_library(linkcell::eh ALIAS linkcell_eh)
target_include_directories(linkcell_eh PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_compile_features(linkcell_eh PUBLIC cxx_std_17)
set_target_properties(linkcell_eh PROPERTIES POSITION_INDEPENDENT_CODE ON)
foreach(_linkcell_eh_kind RUNTIME LIBRARY ARCHIVE)
  set_target_properties(linkcell_eh PROPERTIES
    "${_linkcell_eh_kind}_OUTPUT_DIRECTORY" "${CMAKE_CURRENT_BINARY_DIR}")
  foreach(_linkcell_eh_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(linkcell_eh PROPERTIES
      "${_linkcell_eh_kind}_OUTPUT_DIRECTORY_${_linkcell_eh_cfg}"
      "${CMAKE_CURRENT_BINARY_DIR}")
  endforeach()
endforeach()

add_library(linkcell_cpp INTERFACE)
add_library(linkcell::cpp ALIAS linkcell_cpp)
target_link_libraries(linkcell_cpp INTERFACE linkcell::linkcell linkcell::eh)
target_compile_features(linkcell_cpp INTERFACE cxx_std_17)

set(LINKCELL_GPU_SYSTEM_LIBS Threads::Threads)
if(CMAKE_DL_LIBS)
  list(APPEND LINKCELL_GPU_SYSTEM_LIBS ${CMAKE_DL_LIBS})
endif()

if(LINKCELL_WITH_GPULITE)
  add_library(linkcell_gpu STATIC
    src/gpu/knearest.cpp
    src/gpu/capi.cpp
    src/gpu/partition.cpp
  )
  add_library(linkcell::gpu ALIAS linkcell_gpu)
  target_include_directories(linkcell_gpu
    PUBLIC
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
      "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/third_party/gpulite"
  )
  target_compile_definitions(linkcell_gpu PUBLIC LINKCELL_HAS_GPULITE=1)
  target_compile_features(linkcell_gpu PUBLIC cxx_std_17)
  target_link_libraries(linkcell_gpu PUBLIC
    linkcell::linkcell
    linkcell::eh
    ${LINKCELL_GPU_SYSTEM_LIBS}
  )
endif()

if(LINKCELL_BUILD_EXAMPLES)
  add_executable(two_points examples/two_points.c)
  target_link_libraries(two_points PRIVATE linkcell::linkcell)
  set_target_properties(two_points PROPERTIES BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}")
  add_executable(two_points_cpp examples/two_points.cpp)
  target_link_libraries(two_points_cpp PRIVATE linkcell::cpp)
  set_target_properties(two_points_cpp PROPERTIES BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}")
  add_executable(knearest_non_finite tests/knearest_non_finite.c)
  target_link_libraries(knearest_non_finite PRIVATE linkcell::linkcell)
  if(UNIX)
    target_link_libraries(knearest_non_finite PRIVATE m)
  endif()
  set_target_properties(knearest_non_finite PROPERTIES BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}")
  add_executable(knearest_non_finite_cpp tests/knearest_non_finite.cpp)
  target_link_libraries(knearest_non_finite_cpp PRIVATE linkcell::cpp)
  set_target_properties(knearest_non_finite_cpp PROPERTIES BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}")
  enable_testing()
  add_test(NAME two-points-c COMMAND two_points)
  add_test(NAME two-points-cpp COMMAND two_points_cpp)
  add_test(NAME knearest-non-finite-c COMMAND knearest_non_finite)
  add_test(NAME knearest-non-finite-cpp COMMAND knearest_non_finite_cpp)
  set(_linkcell_example_tests
    two-points-c two-points-cpp knearest-non-finite-c knearest-non-finite-cpp)
  add_executable(exception_typeinfo_static
    tests/exception_typeinfo_static.cpp
    tests/exception_typeinfo_static_other.cpp
    src/error.cpp)
  target_include_directories(exception_typeinfo_static PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_compile_features(exception_typeinfo_static PRIVATE cxx_std_17)
  target_compile_definitions(exception_typeinfo_static PRIVATE LINKCELL_EH_STATIC)
  add_test(NAME exception-typeinfo-static COMMAND exception_typeinfo_static)
  list(APPEND _linkcell_example_tests exception-typeinfo-static)
  if(NOT LINKCELL_EH_STATIC)
    add_library(exception_typeinfo_hidden_a SHARED tests/exception_typeinfo_hidden.cpp)
    add_library(exception_typeinfo_hidden_b SHARED tests/exception_typeinfo_hidden.cpp)
    target_compile_definitions(exception_typeinfo_hidden_b PRIVATE LC_PROBE_B)
    foreach(_linkcell_hidden exception_typeinfo_hidden_a exception_typeinfo_hidden_b)
      target_link_libraries(${_linkcell_hidden} PRIVATE linkcell::eh)
      target_compile_features(${_linkcell_hidden} PRIVATE cxx_std_17)
      set_target_properties(${_linkcell_hidden} PROPERTIES
        BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}")
      if(NOT MSVC)
        target_compile_options(${_linkcell_hidden} PRIVATE
          -fvisibility=hidden -fvisibility-inlines-hidden)
      endif()
    endforeach()
    add_executable(exception_typeinfo tests/exception_typeinfo.cpp)
    target_compile_features(exception_typeinfo PRIVATE cxx_std_17)
    target_link_libraries(exception_typeinfo PRIVATE
      linkcell::eh exception_typeinfo_hidden_a exception_typeinfo_hidden_b)
    set_target_properties(exception_typeinfo PROPERTIES
      BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR}")
    add_test(NAME exception-typeinfo COMMAND exception_typeinfo)
    list(APPEND _linkcell_example_tests exception-typeinfo)
  endif()
  if(UNIX AND NOT APPLE)
    set_tests_properties(${_linkcell_example_tests} PROPERTIES
      ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}")
  elseif(APPLE)
    set_tests_properties(${_linkcell_example_tests} PROPERTIES
      ENVIRONMENT "DYLD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}")
  elseif(WIN32)
    set_tests_properties(${_linkcell_example_tests} PROPERTIES
      ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${CMAKE_CURRENT_BINARY_DIR}")
  endif()
endif()

if(LINKCELL_INSTALL)
  if(UNIX AND NOT APPLE)
    if(_linkcell_use_openmp)
      if(LINKCELL_OMP_DIR)
        set(LINKCELL_PKG_LIBS_PRIVATE
          "-ldl -lpthread -lm -L${LINKCELL_OMP_DIR} -l${LINKCELL_OMP_LIB}")
      else()
        set(LINKCELL_PKG_LIBS_PRIVATE "-ldl -lpthread -lm -l${LINKCELL_OMP_LIB}")
      endif()
    else()
      set(LINKCELL_PKG_LIBS_PRIVATE "-ldl -lpthread -lm")
    endif()
    set(LINKCELL_GPU_PKG_LIBS "-ldl -lpthread")
  elseif(APPLE)
    if(_linkcell_use_openmp)
      set(LINKCELL_PKG_LIBS_PRIVATE
        "-lpthread -lm -L${LINKCELL_OMP_DIR} -l${LINKCELL_OMP_LIB}")
    else()
      set(LINKCELL_PKG_LIBS_PRIVATE "-lpthread -lm")
    endif()
    set(LINKCELL_GPU_PKG_LIBS "-lpthread")
  else()
    set(LINKCELL_PKG_LIBS_PRIVATE "")
    set(LINKCELL_GPU_PKG_LIBS "")
  endif()

  if(LINKCELL_EH_STATIC)
    set(LINKCELL_EH_PC_CFLAGS " -DLINKCELL_EH_STATIC")
  else()
    set(LINKCELL_EH_PC_CFLAGS "")
  endif()
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/linkcell.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/linkcell.pc"
    @ONLY
  )
  if(LINKCELL_WITH_GPULITE)
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/linkcell-gpu.pc.in"
      "${CMAKE_CURRENT_BINARY_DIR}/linkcell-gpu.pc"
      @ONLY
    )
  endif()
  configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/linkcellConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/linkcellConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/linkcell"
  )
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/linkcellConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
  )

  if(WIN32)
    install(FILES "${LINKCELL_SHARED}" DESTINATION "${CMAKE_INSTALL_BINDIR}")
    install(FILES "${LINKCELL_IMPLIB}" DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  else()
    install(FILES "${LINKCELL_SHARED}" DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  endif()
  install(FILES "${LINKCELL_STATIC}" DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  install(TARGETS linkcell_eh
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  if(LINKCELL_WITH_GPULITE)
    install(TARGETS linkcell_gpu ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
  endif()
  install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/include/linkcell.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/linkcell.hpp"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  )
  if(LINKCELL_WITH_GPULITE)
    install(FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/include/linkcell_gpu.h"
      "${CMAKE_CURRENT_SOURCE_DIR}/include/linkcell_gpu.hpp"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )
  endif()
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/linkcell.pc"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
  )
  if(LINKCELL_WITH_GPULITE)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/linkcell-gpu.pc"
      DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
    )
  endif()
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/linkcellConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/linkcellConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/linkcell"
  )
endif()

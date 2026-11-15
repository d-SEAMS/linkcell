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
option(LINKCELL_WITH_GPULITE "Build the device k-nearest C and C++ APIs" ON)

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

file(GLOB_RECURSE LINKCELL_RUST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.rs")
list(APPEND LINKCELL_RUST_SOURCES
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

add_custom_command(
  OUTPUT ${LINKCELL_CARGO_OUTPUTS}
  COMMAND "${CARGO_EXECUTABLE}" rustc --lib ${LINKCELL_CARGO_ARGS}
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

set(_linkcell_sys Threads::Threads)
if(UNIX AND NOT APPLE)
  list(APPEND _linkcell_sys ${CMAKE_DL_LIBS} m)
elseif(APPLE)
  list(APPEND _linkcell_sys m)
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

add_library(linkcell_cpp INTERFACE)
add_library(linkcell::cpp ALIAS linkcell_cpp)
target_link_libraries(linkcell_cpp INTERFACE linkcell::linkcell)
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
  enable_testing()
  add_test(NAME two-points-c COMMAND two_points)
  add_test(NAME two-points-cpp COMMAND two_points_cpp)
  if(UNIX AND NOT APPLE)
    set_tests_properties(two-points-c two-points-cpp PROPERTIES
      ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}")
  elseif(APPLE)
    set_tests_properties(two-points-c two-points-cpp PROPERTIES
      ENVIRONMENT "DYLD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}")
  endif()
endif()

if(LINKCELL_INSTALL)
  if(UNIX AND NOT APPLE)
    set(LINKCELL_PKG_LIBS_PRIVATE "-ldl -lpthread -lm")
    set(LINKCELL_GPU_PKG_LIBS "-ldl -lpthread")
  elseif(APPLE)
    set(LINKCELL_PKG_LIBS_PRIVATE "-lpthread -lm")
    set(LINKCELL_GPU_PKG_LIBS "-lpthread")
  else()
    set(LINKCELL_PKG_LIBS_PRIVATE "")
    set(LINKCELL_GPU_PKG_LIBS "")
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

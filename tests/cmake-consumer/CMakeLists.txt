cmake_minimum_required(VERSION 3.22)
project(linkcell-consumer LANGUAGES C CXX)

# Installed-prefix contract: find_package(linkcell) yields CPU and GPU targets
# with their include, language, and transitive link requirements.
find_package(linkcell 0.6 REQUIRED CONFIG)
if(NOT TARGET linkcell::linkcell)
  message(FATAL_ERROR "find_package(linkcell) did not define linkcell::linkcell")
endif()
if(NOT TARGET linkcell::cpp)
  message(FATAL_ERROR "find_package(linkcell) did not define linkcell::cpp")
endif()
if(NOT TARGET linkcell::gpu)
  message(FATAL_ERROR "find_package(linkcell) did not define linkcell::gpu")
endif()

add_executable(find_lc_c "${LINKCELL_C_EXAMPLE}")
target_link_libraries(find_lc_c PRIVATE linkcell::linkcell)

add_executable(find_lc_cpp "${LINKCELL_CPP_EXAMPLE}")
target_link_libraries(find_lc_cpp PRIVATE linkcell::cpp)

add_executable(find_lc_gpu gpu.cpp)
target_link_libraries(find_lc_gpu PRIVATE linkcell::gpu)

enable_testing()
add_test(NAME installed-linkcell-c-consumer COMMAND find_lc_c)
add_test(NAME installed-linkcell-cpp-consumer COMMAND find_lc_cpp)
add_test(NAME installed-linkcell-gpu-consumer COMMAND find_lc_gpu)

# Installed linkcell::eh is the one shared owner. Two hidden DSOs import it
# and the probe catches linkcell::Error by type.
get_target_property(_linkcell_eh_loc linkcell::eh IMPORTED_LOCATION)
get_filename_component(_linkcell_eh_dir "${_linkcell_eh_loc}" DIRECTORY)
add_library(installed_hidden_a SHARED
  "${CMAKE_CURRENT_SOURCE_DIR}/../exception_typeinfo_hidden.cpp")
add_library(installed_hidden_b SHARED
  "${CMAKE_CURRENT_SOURCE_DIR}/../exception_typeinfo_hidden.cpp")
target_compile_definitions(installed_hidden_b PRIVATE LC_PROBE_B)
foreach(_linkcell_hidden installed_hidden_a installed_hidden_b)
  target_link_libraries(${_linkcell_hidden} PRIVATE linkcell::eh)
  set_target_properties(${_linkcell_hidden} PROPERTIES
    BUILD_RPATH "${_linkcell_eh_dir}")
  if(NOT MSVC)
    target_compile_options(${_linkcell_hidden} PRIVATE
      -fvisibility=hidden -fvisibility-inlines-hidden)
  endif()
endforeach()
add_executable(installed_exception_typeinfo
  "${CMAKE_CURRENT_SOURCE_DIR}/../exception_typeinfo.cpp")
target_link_libraries(installed_exception_typeinfo PRIVATE
  linkcell::eh installed_hidden_a installed_hidden_b)
set_target_properties(installed_exception_typeinfo PROPERTIES
  BUILD_RPATH "${_linkcell_eh_dir};${CMAKE_CURRENT_BINARY_DIR}")
add_test(NAME installed-exception-typeinfo COMMAND installed_exception_typeinfo)
if(WIN32)
  set_tests_properties(installed-exception-typeinfo PROPERTIES
    ENVIRONMENT "PATH=${_linkcell_eh_dir};$ENV{PATH}")
elseif(APPLE)
  set_tests_properties(installed-exception-typeinfo PROPERTIES
    ENVIRONMENT "DYLD_LIBRARY_PATH=${_linkcell_eh_dir}:${CMAKE_CURRENT_BINARY_DIR}")
else()
  set_tests_properties(installed-exception-typeinfo PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${_linkcell_eh_dir}:${CMAKE_CURRENT_BINARY_DIR}")
endif()

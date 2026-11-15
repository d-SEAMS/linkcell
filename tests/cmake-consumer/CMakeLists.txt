cmake_minimum_required(VERSION 3.22)
project(linkcell-consumer LANGUAGES C CXX)

# Installed-prefix contract: find_package(linkcell) yields CPU and GPU targets
# with their include, language, and transitive link requirements.
find_package(linkcell 0.4 REQUIRED CONFIG)
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

A Clang build finds `libomp` beside the compiler resource directory, so Meson, CMake, and the crate link it when the linker name is not on the default search path.

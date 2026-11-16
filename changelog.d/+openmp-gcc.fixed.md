A native Linux build with no C compiler chosen compiles the OpenMP k-nearest region with gcc, so the `parallel` build no longer stops on a missing `omp.h` when `cc` is clang.
`CC`, `TARGET_CC` and `CC_<target>` still choose the compiler.

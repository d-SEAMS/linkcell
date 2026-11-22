`LINKCELL_USE_OPENMP` (CMake), `use_openmp` (Meson), and `LINKCELL_OPENMP=0` (Cargo) disable the fused OpenMP k-nearest region so the archive and `linkcell.pc` omit OpenMP symbols and runtime links.

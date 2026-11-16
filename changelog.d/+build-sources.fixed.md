Meson and CMake rebuild the library when any crate source changes. Meson reads the sources from the Cargo dep-info, and CMake also tracks `build.rs` and the C sources.

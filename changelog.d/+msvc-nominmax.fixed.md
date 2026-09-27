The MSVC build defines `NOMINMAX` before `windows.h` and calls `(std::min)` / `(std::max)`, so those names are not macros.

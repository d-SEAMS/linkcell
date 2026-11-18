Meson links libomp or libgomp by name, with `-L` for the directory it found the runtime in, so `linkcell.pc` keeps `-lomp` where it had started to name the file's full path on Linux.

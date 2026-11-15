The NEON fold truncates like minimage `floor_fast`, so an input of -0 keeps a
fraction of -0 on aarch64 and matches the scalar path.

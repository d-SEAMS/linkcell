The `scotch` and `ptscotch` sources include `stdio.h` before the Scotch headers. Scotch 7.0.11 uses `FILE` in those headers and does not include `stdio.h` itself.

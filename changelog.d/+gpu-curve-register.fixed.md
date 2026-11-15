The device partition finds the GPU curve split through `lc_set_gpu_curve_split`, which the GPU library calls when it loads.
The crate no longer carries a weak stub, so it links on macOS and with MSVC.

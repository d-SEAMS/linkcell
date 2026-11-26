#![cfg(feature = "capi")]
//! C ABI buffer contract. Feature `capi` is on by default.

use std::os::raw::c_int;

use linkcell::{knearest_into, Cell, Error, Metric, PeriodicSpace};

#[repr(C)]
struct LcCell {
    ax: f64,
    ay: f64,
    az: f64,
    bx: f64,
    by: f64,
    bz: f64,
    cx: f64,
    cy: f64,
    cz: f64,
    ox: f64,
    oy: f64,
    oz: f64,
}

#[repr(C)]
struct LcNdIndex {
    _private: u8,
}

extern "C" {
    #[cfg(any(target_os = "linux", target_os = "macos", windows))]
    fn lc_set_threads(n: c_int);
    fn lc_knearest(
        xyz: *const f64,
        n: usize,
        simbox: *const LcCell,
        k: usize,
        mask: *const c_int,
        cell_hint: f64,
        out_nn: *mut c_int,
    ) -> c_int;
    fn lc_knearest_d2(
        xyz: *const f64,
        n: usize,
        simbox: *const LcCell,
        k: usize,
        mask: *const c_int,
        cell_hint: f64,
        out_nn: *mut c_int,
        out_d2: *mut f64,
    ) -> c_int;
    fn lc_knearest_many(
        xyz: *const f64,
        n: usize,
        n_frames: usize,
        simbox: *const LcCell,
        k: usize,
        mask: *const c_int,
        cell_hint: f64,
        out_nn: *mut c_int,
        out_d2: *mut f64,
    ) -> c_int;
    fn lc_pairs_within(
        xyz: *const f64,
        n: usize,
        simbox: *const LcCell,
        cutoff: f64,
        mask: *const c_int,
        cell_hint: f64,
        half: c_int,
        out_i: *mut c_int,
        out_j: *mut c_int,
        out_shift: *mut c_int,
        out_d2: *mut f64,
        cap: usize,
        out_count: *mut usize,
    ) -> c_int;
    fn lc_pairs_within_rows(
        xyz: *const f64,
        n: usize,
        simbox: *const LcCell,
        cutoff: f64,
        mask: *const c_int,
        cell_hint: f64,
        half: c_int,
        out: *mut linkcell::lc_pair,
        cap: usize,
        out_count: *mut usize,
    ) -> c_int;
    fn lc_nd_build(
        data: *const f64,
        n: usize,
        d: usize,
        periods: *const f64,
        out: *mut *mut LcNdIndex,
    ) -> c_int;
    fn lc_nd_free(index: *mut LcNdIndex);
    fn lc_nd_len(index: *const LcNdIndex) -> usize;
    fn lc_nd_dims(index: *const LcNdIndex) -> usize;
    fn lc_nd_knn(
        index: *const LcNdIndex,
        query: *const f64,
        k: usize,
        out_idx: *mut c_int,
        out_d2: *mut f64,
    ) -> c_int;
    fn lc_nd_within(
        index: *const LcNdIndex,
        query: *const f64,
        radius: f64,
        out_idx: *mut c_int,
        out_d2: *mut f64,
        cap: usize,
        out_count: *mut usize,
    ) -> c_int;
    fn lc_nd_pairs_within(
        index: *const LcNdIndex,
        radius: f64,
        out_i: *mut c_int,
        out_j: *mut c_int,
        out_d2: *mut f64,
        cap: usize,
        out_count: *mut usize,
    ) -> c_int;
    fn lc_nd_dbscan(
        index: *const LcNdIndex,
        eps: f64,
        min_pts: usize,
        out_labels: *mut c_int,
        out_core: *mut c_int,
        out_clusters: *mut usize,
    ) -> c_int;
}

/// Kernel identifiers of this process's live threads.
#[cfg(target_os = "linux")]
fn thread_ids() -> std::collections::BTreeSet<u64> {
    std::fs::read_dir("/proc/self/task")
        .unwrap()
        .map(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_str()
                .unwrap()
                .parse()
                .unwrap()
        })
        .collect()
}

/// Kernel thread ids of this process's live threads.
///
/// These are the ids `pthread_threadid_np` returns. `PROC_PIDLISTTHREADS`
/// (6) lists each thread's user-space cthread handle instead, and a joined
/// thread still in the task list then compares unequal to the id the probe
/// recorded. `PROC_PIDLISTTHREADIDS` (28) is the XNU flavor that lists the
/// stable ids (`thuniqueid` in `proc_pidlistthreads`). It is not in the
/// public SDK header; `proc_pidinfo` forwards the flavor.
#[cfg(target_os = "macos")]
fn thread_ids() -> std::collections::BTreeSet<u64> {
    extern "C" {
        fn getpid() -> c_int;
        fn proc_pidinfo(
            pid: c_int,
            flavor: c_int,
            arg: u64,
            buffer: *mut std::ffi::c_void,
            size: c_int,
        ) -> c_int;
    }
    const PROC_PIDLISTTHREADIDS: c_int = 28;
    let mut ids = vec![0u64; 64];
    loop {
        let bytes = c_int::try_from(ids.len() * 8).unwrap();
        let got = unsafe {
            proc_pidinfo(
                getpid(),
                PROC_PIDLISTTHREADIDS,
                0,
                ids.as_mut_ptr().cast(),
                bytes,
            )
        };
        assert!(
            got > 0,
            "proc_pidinfo(PROC_PIDLISTTHREADIDS) returned {got}"
        );
        if got < bytes {
            return ids[..got as usize / 8].iter().copied().collect();
        }
        ids.resize(ids.len() * 2, 0);
    }
}

/// Threads this executable created with `CreateThread` that are still alive.
///
/// Toolhelp lists every thread in the process. Loader and thread-pool
/// threads that `ntdll` starts without this executable's `CreateThread` are
/// not linkcell's, and the Linux and macOS snapshots do not include an
/// equivalent. The raw walk must still see this thread; the returned set is
/// the recorded ids that walk still contains.
#[cfg(windows)]
fn thread_ids() -> std::collections::BTreeSet<u64> {
    // THREADENTRY32: dwSize, cntUsage, th32ThreadID, th32OwnerProcessID,
    // tpBasePri, tpDeltaPri, dwFlags. Every field is 32 bits.
    type ThreadEntry32 = [u32; 7];
    const SIZE: usize = 0;
    const THREAD_ID: usize = 2;
    const OWNER_PROCESS_ID: usize = 3;
    #[link(name = "kernel32")]
    extern "system" {
        fn CreateToolhelp32Snapshot(flags: u32, process_id: u32) -> *mut std::ffi::c_void;
        fn Thread32First(snapshot: *mut std::ffi::c_void, entry: *mut ThreadEntry32) -> i32;
        fn Thread32Next(snapshot: *mut std::ffi::c_void, entry: *mut ThreadEntry32) -> i32;
        fn CloseHandle(handle: *mut std::ffi::c_void) -> i32;
        fn GetCurrentProcessId() -> u32;
    }
    const TH32CS_SNAPTHREAD: u32 = 0x4;
    let entry_size = std::mem::size_of::<ThreadEntry32>() as u32;
    let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
    assert_ne!(snapshot as isize, -1, "CreateToolhelp32Snapshot failed");
    let pid = unsafe { GetCurrentProcessId() };
    let mut entry: ThreadEntry32 = [0; 7];
    entry[SIZE] = entry_size;
    let mut ids = std::collections::BTreeSet::new();
    let mut more = unsafe { Thread32First(snapshot, &mut entry) } != 0;
    while more {
        if entry[OWNER_PROCESS_ID] == pid {
            ids.insert(u64::from(entry[THREAD_ID]));
        }
        entry[SIZE] = entry_size;
        more = unsafe { Thread32Next(snapshot, &mut entry) } != 0;
    }
    assert_ne!(unsafe { CloseHandle(snapshot) }, 0, "CloseHandle failed");
    assert!(!ids.is_empty(), "the snapshot did not list this thread");
    let created = nd_spawn::created_ids();
    ids.intersection(&created).copied().collect()
}

/// Counts threads created while a call runs, including one it has already joined.
///
/// A before/after set of live thread ids misses a worker that is spawned and
/// joined inside the call: the id is gone when the call returns.
/// `tests/one_thread.rs` counts `pthread_create` on Linux gnu for every C
/// entry. This probe counts the create itself on glibc, macOS, and Windows,
/// and still compares live ids so a thread left running is caught. Other
/// Linux targets keep that id snapshot and do not define `pthread_create`:
/// `dlsym(RTLD_NEXT)` is null in a static musl binary.
///
/// * Linux with glibc (`target_env = "gnu"`). This binary defines
///   `pthread_create` and forwards to libc with `dlsym(RTLD_NEXT)`. The
///   statically linked standard library resolves that symbol to this
///   definition, so `std::thread::spawn` increments the counter even when
///   the worker has exited. The increment happens before that call returns,
///   because the new thread can finish first and the harness is still
///   inside `pthread_create` for this test's own thread when the body arms
///   the counter. A failed create rolls the count back. A nested
///   `pthread_create` on the same thread counts as one worker. That nesting
///   is the counter's rule; it is not a claim that the C library calls the
///   symbol again. The abi tests pass on glibc 2.39 with the depth unused.
/// * macOS. dyld binds `std`'s `pthread_create` to libSystem, so a definition
///   in this binary is not that call, and a `__DATA,__interpose` record in
///   this binary did not observe it: the self-check's count stayed zero.
///   libpthread reports `PTHREAD_INTROSPECTION_THREAD_CREATE` on the creating
///   thread inside `pthread_create`, before the new thread runs, and
///   `PTHREAD_INTROSPECTION_THREAD_START` on the new thread before its start
///   routine. The probe installs a hook with
///   `pthread_introspection_hook_install` and calls the hook that was already
///   installed. CREATE counts. START for a thread CREATE already counted is
///   the same thread and does not count again. START with no CREATE counts.
///   DESTROY with no START rolls that CREATE back: `__bsdthread_create`
///   failed and no thread ran.
/// * Windows. The standard library imports `CreateThread` from
///   `kernel32.dll` and passes a null id pointer, so the probe reads
///   `GetThreadId` on the returned handle. A `CreateThread` defined in this
///   binary does not intercept that import, and a Toolhelp snapshot does not
///   list a thread that has exited. The probe replaces this executable's
///   import-address slot, which is the pointer the statically linked
///   standard library calls, and forwards to the address the loader wrote
///   there. The live snapshot keeps only those recorded ids that Toolhelp
///   still lists for this process. Toolhelp also lists loader and
///   thread-pool threads that `ntdll` starts without going through this
///   executable's `CreateThread`; they are not linkcell's. Linux
///   `/proc/self/task` and macOS `proc_pidinfo` have no such threads, so
///   listing every thread there is the same set.
#[cfg(any(
    all(target_os = "linux", target_env = "gnu"),
    target_os = "macos",
    windows
))]
mod nd_spawn {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    static ARMED: AtomicBool = AtomicBool::new(false);
    static SPAWNS: AtomicU64 = AtomicU64::new(0);

    pub(super) fn armed() -> bool {
        ARMED.load(Ordering::SeqCst)
    }

    pub(super) fn count() -> u64 {
        SPAWNS.load(Ordering::SeqCst)
    }

    /// Ids of threads this executable's `CreateThread` hook has created.
    /// Dead threads stay recorded; the live snapshot intersects this with
    /// Toolhelp.
    #[cfg(windows)]
    pub(super) fn created_ids() -> std::collections::BTreeSet<u64> {
        win::created_ids()
    }

    /// Records one create. The caller must do this before the OS call
    /// returns: the new thread can run to completion first, and the test
    /// harness's own create of this test's thread can still be in that
    /// call when the test body arms the counter.
    fn note() -> bool {
        if ARMED.load(Ordering::SeqCst) {
            SPAWNS.fetch_add(1, Ordering::SeqCst);
            true
        } else {
            false
        }
    }

    fn unnote() {
        SPAWNS.fetch_sub(1, Ordering::SeqCst);
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    mod unix {
        use std::os::raw::{c_char, c_int, c_void};
        use std::sync::atomic::{AtomicPtr, Ordering};

        use super::{note, unnote};

        type Start = extern "C" fn(*mut c_void) -> *mut c_void;
        type Create = unsafe extern "C" fn(*mut c_void, *const c_void, Start, *mut c_void) -> c_int;

        const RTLD_NEXT: *mut c_void = -1isize as *mut c_void;
        static REAL: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());

        extern "C" {
            fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
            fn abort() -> !;
        }

        std::thread_local! {
            static IN_RESOLVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
            /// Nesting of `pthread_create` on this thread. Only the outer call
            /// is a worker. This records that rule; glibc 2.39 was not observed
            /// to call the symbol again, and the abi tests pass there with this
            /// depth unused.
            static DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        }

        fn real() -> Create {
            let existing = REAL.load(Ordering::Acquire);
            if !existing.is_null() {
                // SAFETY: `REAL` is the `pthread_create` `dlsym(RTLD_NEXT)` returned.
                return unsafe { std::mem::transmute::<*mut c_void, Create>(existing) };
            }
            let reentered = IN_RESOLVE.with(|flag| flag.replace(true));
            if reentered {
                unsafe { abort() };
            }
            let found = unsafe { dlsym(RTLD_NEXT, c"pthread_create".as_ptr()) };
            if found.is_null() {
                unsafe { abort() };
            }
            REAL.store(found, Ordering::Release);
            IN_RESOLVE.with(|flag| flag.set(false));
            // SAFETY: `dlsym` returned libc's `pthread_create`.
            unsafe { std::mem::transmute::<*mut c_void, Create>(found) }
        }

        /// # Safety
        ///
        /// The arguments are those of `pthread_create(3)`.
        pub(super) unsafe extern "C" fn counted(
            thread: *mut c_void,
            attr: *const c_void,
            start: Start,
            arg: *mut c_void,
        ) -> c_int {
            let real = real();
            let outer = DEPTH.with(|depth| {
                let now = depth.get();
                depth.set(now + 1);
                now == 0
            });
            let counted = outer && note();
            let rc = unsafe { real(thread, attr, start, arg) };
            DEPTH.with(|depth| depth.set(depth.get() - 1));
            if rc != 0 && counted {
                unnote();
            }
            rc
        }

        pub(super) fn ensure() {
            let _ = real();
        }
    }

    /// Glibc binds this executable's definition. Other Linux targets do not
    /// define this symbol: `dlsym(RTLD_NEXT)` is null when musl is linked
    /// statically.
    ///
    /// # Safety
    ///
    /// The arguments are those of `pthread_create(3)`.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[no_mangle]
    pub unsafe extern "C" fn pthread_create(
        thread: *mut std::os::raw::c_void,
        attr: *const std::os::raw::c_void,
        start: extern "C" fn(*mut std::os::raw::c_void) -> *mut std::os::raw::c_void,
        arg: *mut std::os::raw::c_void,
    ) -> std::os::raw::c_int {
        unsafe { unix::counted(thread, attr, start, arg) }
    }

    /// Counts threads libpthread actually creates.
    ///
    /// `PTHREAD_INTROSPECTION_THREAD_CREATE` is 1 and
    /// `PTHREAD_INTROSPECTION_THREAD_START` is 2, from
    /// `<pthread/introspection.h>`. CREATE runs on the creating thread before
    /// `pthread_create` returns. START runs on the new thread before its start
    /// routine. A failed create delivers DESTROY (4) and no START.
    #[cfg(target_os = "macos")]
    mod mac {
        use std::os::raw::{c_uint, c_void};
        use std::sync::{Mutex, Once};

        use super::{note, unnote};

        const THREAD_CREATE: c_uint = 1;
        const THREAD_START: c_uint = 2;
        const THREAD_DESTROY: c_uint = 4;

        type Hook = unsafe extern "C" fn(c_uint, *mut c_void, *mut c_void, usize);

        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Phase {
            Created,
            Started,
        }

        /// Addresses, not pointers: a `*mut c_void` is not `Send`, and the
        /// hook runs on the creating thread and on the new thread.
        static SEEN: Mutex<Vec<(usize, Phase)>> = Mutex::new(Vec::new());
        static PREV: Mutex<Option<Hook>> = Mutex::new(None);

        extern "C" {
            fn pthread_introspection_hook_install(hook: Option<Hook>) -> Option<Hook>;
        }

        fn seen() -> std::sync::MutexGuard<'static, Vec<(usize, Phase)>> {
            SEEN.lock().unwrap_or_else(|err| err.into_inner())
        }

        fn on_create(thread: usize) {
            let mut seen = seen();
            if seen.iter().any(|(id, _)| *id == thread) {
                return;
            }
            if note() {
                seen.push((thread, Phase::Created));
            }
        }

        fn on_start(thread: usize) {
            let mut seen = seen();
            if let Some((_, phase)) = seen.iter_mut().find(|(id, _)| *id == thread) {
                *phase = Phase::Started;
                return;
            }
            if note() {
                seen.push((thread, Phase::Started));
            }
        }

        fn on_destroy(thread: usize) {
            let mut seen = seen();
            let Some(pos) = seen.iter().position(|(id, _)| *id == thread) else {
                return;
            };
            let started = seen[pos].1 == Phase::Started;
            seen.swap_remove(pos);
            if !started {
                unnote();
            }
        }

        unsafe extern "C" fn hook(
            event: c_uint,
            thread: *mut c_void,
            addr: *mut c_void,
            size: usize,
        ) {
            if !thread.is_null() {
                let id = thread as usize;
                match event {
                    THREAD_CREATE => on_create(id),
                    THREAD_START => on_start(id),
                    THREAD_DESTROY => on_destroy(id),
                    _ => {}
                }
            }
            let prev = *PREV.lock().unwrap_or_else(|err| err.into_inner());
            if let Some(prev) = prev {
                // SAFETY: `prev` is the hook `pthread_introspection_hook_install` returned.
                unsafe { prev(event, thread, addr, size) };
            }
        }

        pub(super) fn install() {
            static ONCE: Once = Once::new();
            ONCE.call_once(|| {
                seen().reserve(64);
                let prev = unsafe { pthread_introspection_hook_install(Some(hook)) };
                *PREV.lock().unwrap_or_else(|err| err.into_inner()) = prev;
            });
        }
    }

    #[cfg(windows)]
    mod win {
        use std::sync::atomic::{AtomicPtr, Ordering};

        use super::{note, unnote};

        type Start = Option<unsafe extern "system" fn(*mut std::ffi::c_void) -> u32>;
        type CreateThread = unsafe extern "system" fn(
            *mut std::ffi::c_void,
            usize,
            Start,
            *mut std::ffi::c_void,
            u32,
            *mut u32,
        ) -> *mut std::ffi::c_void;

        static REAL: AtomicPtr<std::ffi::c_void> = AtomicPtr::new(std::ptr::null_mut());

        std::thread_local! {
            /// Nested `CreateThread` on this thread is the same worker.
            static DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        }

        #[link(name = "kernel32")]
        extern "system" {
            fn GetModuleHandleW(name: *const u16) -> *mut std::ffi::c_void;
            fn GetThreadId(thread: *mut std::ffi::c_void) -> u32;
            fn VirtualProtect(
                addr: *mut std::ffi::c_void,
                size: usize,
                new_protect: u32,
                old_protect: *mut u32,
            ) -> i32;
        }

        static CREATED: std::sync::Mutex<std::collections::BTreeSet<u64>> =
            std::sync::Mutex::new(std::collections::BTreeSet::new());

        fn remember(id: u64) {
            // A created thread with no id would be invisible to the snapshot.
            if id == 0 {
                std::process::abort();
            }
            CREATED
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .insert(id);
        }

        pub(super) fn created_ids() -> std::collections::BTreeSet<u64> {
            CREATED
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .clone()
        }

        const PAGE_READWRITE: u32 = 0x04;

        fn read_u16(base: *const u8, off: usize, size: usize) -> Result<u16, String> {
            let end = off.checked_add(2).ok_or("PE offset overflows")?;
            if end > size {
                return Err(format!("PE offset {off} is outside the image"));
            }
            // SAFETY: `end` is within the mapped image `GetModuleHandleW` returned.
            let p = unsafe { base.add(off) };
            Ok(u16::from_le_bytes(unsafe { [*p, *p.add(1)] }))
        }

        fn read_u32(base: *const u8, off: usize, size: usize) -> Result<u32, String> {
            let end = off.checked_add(4).ok_or("PE offset overflows")?;
            if end > size {
                return Err(format!("PE offset {off} is outside the image"));
            }
            let p = unsafe { base.add(off) };
            Ok(u32::from_le_bytes(unsafe {
                [*p, *p.add(1), *p.add(2), *p.add(3)]
            }))
        }

        fn read_usize_at(ptr: *const u8) -> usize {
            // SAFETY: caller keeps `ptr` inside the image and pointer-aligned.
            unsafe { std::ptr::read_unaligned(ptr.cast::<usize>()) }
        }

        fn cstr_eq(base: *const u8, rva: u32, size: usize, want: &[u8]) -> Result<bool, String> {
            let start = rva as usize;
            if start >= size {
                return Err("import name is outside the image".to_string());
            }
            let mut got = Vec::new();
            let mut off = start;
            while off < size {
                let byte = unsafe { *base.add(off) };
                if byte == 0 {
                    return Ok(got == want);
                }
                got.push(byte);
                if got.len() > want.len() {
                    return Ok(false);
                }
                off += 1;
            }
            Err("import name is not terminated".to_string())
        }

        /// Replace every `CreateThread` import slot in this executable.
        pub(super) fn patch() -> Result<(), String> {
            let base = unsafe { GetModuleHandleW(std::ptr::null()) };
            if base.is_null() {
                return Err("GetModuleHandleW returned null".to_string());
            }
            let base = base.cast::<u8>();
            if read_u16(base, 0, 64)? != 0x5A4D {
                return Err("executable is not a PE image".to_string());
            }
            // `e_lfanew` is a `LONG` at offset 0x3C, inside the first page.
            let e_lfanew = read_u32(base, 0x3C, 0x400)? as usize;
            let opt = e_lfanew.checked_add(24).ok_or("PE header overflows")?;
            // `SizeOfImage` is at the same optional-header offset for PE32 and PE32+.
            // Read it through a bound that covers the header before that size is known.
            let header_bound = e_lfanew.checked_add(0x400).ok_or("PE header overflows")?;
            let size = read_u32(base, opt + 56, header_bound)? as usize;
            if size < opt + 128 {
                return Err("SizeOfImage does not cover the optional header".to_string());
            }
            if read_u32(base, e_lfanew, size)? != 0x0000_4550 {
                return Err("executable has no PE signature".to_string());
            }
            let magic = read_u16(base, opt, size)?;
            let import_dir = if magic == 0x20B {
                opt + 120
            } else if magic == 0x10B {
                opt + 104
            } else {
                return Err(format!("unknown PE magic {magic:#x}"));
            };
            if std::mem::size_of::<usize>() != if magic == 0x20B { 8 } else { 4 } {
                return Err("PE pointer width does not match this process".to_string());
            }
            let import_rva = read_u32(base, import_dir, size)? as usize;
            if import_rva == 0 {
                return Err("executable has no import directory".to_string());
            }
            let ordinal = 1usize << (usize::BITS - 1);
            let hook = counted_create_thread as *const () as usize;
            let mut patched = 0usize;
            for desc in 0..512usize {
                let at = import_rva + desc * 20;
                let name_rva = read_u32(base, at + 12, size)?;
                if name_rva == 0 {
                    break;
                }
                let int_rva = read_u32(base, at, size)? as usize;
                let iat_rva = read_u32(base, at + 16, size)? as usize;
                let names = if int_rva == 0 { iat_rva } else { int_rva };
                if names == 0 || iat_rva == 0 {
                    continue;
                }
                for slot in 0..8192usize {
                    let name_off = names + slot * std::mem::size_of::<usize>();
                    let iat_off = iat_rva + slot * std::mem::size_of::<usize>();
                    if name_off + std::mem::size_of::<usize>() > size
                        || iat_off + std::mem::size_of::<usize>() > size
                    {
                        return Err("import thunk is outside the image".to_string());
                    }
                    let named = read_usize_at(unsafe { base.add(name_off) });
                    if named == 0 {
                        break;
                    }
                    if named & ordinal != 0 {
                        continue;
                    }
                    // `IMAGE_IMPORT_BY_NAME`: a `u16` hint, then the name.
                    let name_rva = u32::try_from(named)
                        .map_err(|_| "import name RVA does not fit u32".to_string())?;
                    let name_at = name_rva.checked_add(2).ok_or("import name overflows")?;
                    if !cstr_eq(base, name_at, size, b"CreateThread")? {
                        continue;
                    }
                    let iat = unsafe { base.add(iat_off) as *mut usize };
                    let current = unsafe { std::ptr::read_volatile(iat) };
                    if current == hook {
                        patched += 1;
                        continue;
                    }
                    if REAL.load(Ordering::Acquire).is_null() {
                        REAL.store(current as *mut std::ffi::c_void, Ordering::SeqCst);
                    }
                    let mut old = 0u32;
                    let ok = unsafe {
                        VirtualProtect(
                            iat.cast(),
                            std::mem::size_of::<usize>(),
                            PAGE_READWRITE,
                            &mut old,
                        )
                    };
                    if ok == 0 {
                        return Err("VirtualProtect failed on the import slot".to_string());
                    }
                    unsafe { std::ptr::write_volatile(iat, hook) };
                    let mut restore = 0u32;
                    unsafe {
                        VirtualProtect(iat.cast(), std::mem::size_of::<usize>(), old, &mut restore);
                    }
                    patched += 1;
                }
            }
            if patched == 0 {
                return Err(
                    "CreateThread is not imported by this executable; a joined worker would be missed"
                        .to_string(),
                );
            }
            if REAL.load(Ordering::Acquire).is_null() {
                return Err("CreateThread import slot had no original address".to_string());
            }
            Ok(())
        }

        /// # Safety
        ///
        /// The arguments are those of `CreateThread`.
        unsafe extern "system" fn counted_create_thread(
            attrs: *mut std::ffi::c_void,
            stack: usize,
            start: Start,
            param: *mut std::ffi::c_void,
            flags: u32,
            tid: *mut u32,
        ) -> *mut std::ffi::c_void {
            let real = REAL.load(Ordering::SeqCst);
            if real.is_null() {
                std::process::abort();
            }
            let real: CreateThread =
                unsafe { std::mem::transmute::<*mut std::ffi::c_void, CreateThread>(real) };
            let outer = DEPTH.with(|depth| {
                let now = depth.get();
                depth.set(now + 1);
                now == 0
            });
            let counted = outer && note();
            let handle = unsafe { real(attrs, stack, start, param, flags, tid) };
            DEPTH.with(|depth| depth.set(depth.get() - 1));
            if handle.is_null() {
                if counted {
                    unnote();
                }
            } else {
                // Rust passes a null id pointer. The handle is still open here.
                remember(u64::from(unsafe { GetThreadId(handle) }));
            }
            handle
        }
    }

    pub(super) fn arm() -> Result<(), String> {
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        unix::ensure();
        #[cfg(target_os = "macos")]
        mac::install();
        #[cfg(windows)]
        win::patch()?;
        ARMED.store(true, Ordering::SeqCst);
        Ok(())
    }
}

/// Runs `call` and asserts that it neither leaves a new thread alive nor
/// creates one it joins before returning.
#[cfg(any(
    all(target_os = "linux", target_env = "gnu"),
    target_os = "macos",
    windows
))]
fn assert_starts_no_thread(label: &str, call: impl FnOnce()) {
    assert!(
        nd_spawn::armed(),
        "spawn counter is not armed; a joined worker would be missed"
    );
    let before = thread_ids();
    let spawned = nd_spawn::count();
    call();
    let started: Vec<u64> = thread_ids().difference(&before).copied().collect();
    let created = nd_spawn::count() - spawned;
    assert_eq!(
        started,
        Vec::<u64>::new(),
        "under lc_set_threads(1), {label} left a thread alive"
    );
    assert_eq!(
        created, 0,
        "under lc_set_threads(1), {label} created {created} thread(s) and joined them before returning"
    );
}

/// Runs `call` and asserts that no thread alive afterwards is new.
///
/// Linux targets other than glibc use this snapshot. A static musl binary
/// gets a null `dlsym(RTLD_NEXT)`, so this executable does not define
/// `pthread_create`.
#[cfg(all(target_os = "linux", not(target_env = "gnu")))]
fn assert_starts_no_thread(label: &str, call: impl FnOnce()) {
    let before = thread_ids();
    call();
    let started: Vec<u64> = thread_ids().difference(&before).copied().collect();
    assert_eq!(
        started,
        Vec::<u64>::new(),
        "under lc_set_threads(1), {label} started a thread"
    );
}

fn ortho_c(lx: f64, ly: f64, lz: f64) -> LcCell {
    LcCell {
        ax: lx,
        ay: 0.0,
        az: 0.0,
        bx: 0.0,
        by: ly,
        bz: 0.0,
        cx: 0.0,
        cy: 0.0,
        cz: lz,
        ox: 0.0,
        oy: 0.0,
        oz: 0.0,
    }
}

fn pack_xyz(xyz: &[[f64; 3]]) -> Vec<f64> {
    let mut packed = Vec::with_capacity(xyz.len() * 3);
    for p in xyz {
        packed.extend_from_slice(p);
    }
    packed
}

#[test]
fn safe_and_c_abi_write_the_same_packed_row() {
    let sim = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let mut rust_out = [-1; 2];
    knearest_into(&xyz, &sim, 1, None, None, &mut rust_out).unwrap();

    let box_c = linkcell::lc_cell {
        ax: 10.0,
        ay: 0.0,
        az: 0.0,
        bx: 0.0,
        by: 10.0,
        bz: 0.0,
        cx: 0.0,
        cy: 0.0,
        cz: 10.0,
        ox: 0.0,
        oy: 0.0,
        oz: 0.0,
    };
    let packed = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let mut c_out = [-1; 2];
    let rc = unsafe {
        linkcell::lc_knearest(
            packed.as_ptr(),
            2,
            &box_c,
            1,
            std::ptr::null(),
            0.0,
            c_out.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0);
    assert_eq!(c_out, rust_out);
    assert!(linkcell::lc_last_error().is_null());
}

#[test]
fn c_abi_empty_n_sets_thread_local_error() {
    let box_c = linkcell::lc_cell {
        ax: 10.0,
        ay: 0.0,
        az: 0.0,
        bx: 0.0,
        by: 10.0,
        bz: 0.0,
        cx: 0.0,
        cy: 0.0,
        cz: 10.0,
        ox: 0.0,
        oy: 0.0,
        oz: 0.0,
    };
    let dummy = 0.0;
    let mut out = -1;
    let rc =
        unsafe { linkcell::lc_knearest(&dummy, 0, &box_c, 1, std::ptr::null(), 0.0, &mut out) };
    assert_ne!(rc, 0);
    let msg = unsafe { std::ffi::CStr::from_ptr(linkcell::lc_last_error()) };
    assert_eq!(msg.to_str().unwrap(), "no points");
}

#[test]
fn buffer_size_is_not_empty() {
    let b = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [[0.0, 0.0, 0.0]];
    let mut out = [];
    assert_eq!(
        linkcell::knearest_into(&xyz, &b, 1, None, None, &mut out).unwrap_err(),
        Error::BufferSize
    );
}

#[test]
fn lc_knearest_matches_knearest_into_packed() {
    let sim = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.5, 0.0],
        [0.0, 0.0, 2.0],
        [9.7, 0.2, 0.1],
    ];
    let k = 3;
    let n = xyz.len();
    let mut rust_out = vec![-2i32; n * k];
    knearest_into(&xyz, &sim, k, None, None, &mut rust_out).unwrap();

    let packed = pack_xyz(&xyz);
    let box_c = ortho_c(10.0, 10.0, 10.0);
    let mut c_out = vec![-7i32; n * k];
    let rc = unsafe {
        lc_knearest(
            packed.as_ptr(),
            n,
            &box_c,
            k,
            std::ptr::null(),
            0.0,
            c_out.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0);
    assert_eq!(c_out, rust_out);
    for i in 0..n {
        for j in 0..k {
            assert_eq!(c_out[i * k + j], rust_out[i * k + j], "out[{i}*{k}+{j}]");
        }
    }
}

#[test]
fn lc_knearest_null_mask_matches_all_ones() {
    let sim = Cell::ortho(10.0, 10.0, 10.0).unwrap();
    let xyz = [
        [0.0, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [0.0, 1.4, 0.0],
        [0.0, 0.0, 1.6],
    ];
    let k = 2;
    let n = xyz.len();
    let packed = pack_xyz(&xyz);
    let box_c = ortho_c(10.0, 10.0, 10.0);
    let ones = vec![1 as c_int; n];
    let mut out_null = vec![-1i32; n * k];
    let mut out_ones = vec![-1i32; n * k];
    let rc_null = unsafe {
        lc_knearest(
            packed.as_ptr(),
            n,
            &box_c,
            k,
            std::ptr::null(),
            0.0,
            out_null.as_mut_ptr(),
        )
    };
    let rc_ones = unsafe {
        lc_knearest(
            packed.as_ptr(),
            n,
            &box_c,
            k,
            ones.as_ptr(),
            0.0,
            out_ones.as_mut_ptr(),
        )
    };
    assert_eq!(rc_null, 0);
    assert_eq!(rc_ones, 0);
    assert_eq!(out_null, out_ones);

    let mut rust_out = vec![-1i32; n * k];
    knearest_into(&xyz, &sim, k, None, None, &mut rust_out).unwrap();
    assert_eq!(out_null, rust_out);
}

#[test]
fn lc_knearest_zero_k_message_is_not_empty() {
    let box_c = ortho_c(10.0, 10.0, 10.0);
    let packed = [0.0, 0.0, 0.0];
    let mut out = -1;
    let rc = unsafe {
        lc_knearest(
            packed.as_ptr(),
            1,
            &box_c,
            0,
            std::ptr::null(),
            0.0,
            &mut out,
        )
    };
    assert_ne!(rc, 0);
    let msg = unsafe { std::ffi::CStr::from_ptr(linkcell::lc_last_error()) };
    assert_eq!(msg.to_str().unwrap(), "k must be at least 1");
    assert_ne!(msg.to_str().unwrap(), "no points");
}

#[test]
fn last_error_slots_are_independent_across_threads() {
    use std::thread;
    let a = thread::spawn(|| {
        let box_c = ortho_c(10.0, 10.0, 10.0);
        let dummy = 0.0;
        let mut out = -1;
        let rc = unsafe { lc_knearest(&dummy, 0, &box_c, 1, std::ptr::null(), 0.0, &mut out) };
        assert_ne!(rc, 0);
        let msg = unsafe { std::ffi::CStr::from_ptr(linkcell::lc_last_error()) };
        assert_eq!(msg.to_str().unwrap(), "no points");
    });
    let b = thread::spawn(|| {
        let box_c = ortho_c(10.0, 10.0, 10.0);
        let dummy = 0.0;
        let mut out = -1;
        let rc = unsafe { lc_knearest(&dummy, 1, &box_c, 0, std::ptr::null(), 0.0, &mut out) };
        assert_ne!(rc, 0);
        let msg = unsafe { std::ffi::CStr::from_ptr(linkcell::lc_last_error()) };
        assert_eq!(msg.to_str().unwrap(), "k must be at least 1");
    });
    a.join().unwrap();
    b.join().unwrap();
}

#[test]
fn lc_version_does_not_clear_last_error() {
    let box_c = ortho_c(10.0, 10.0, 10.0);
    let dummy = 0.0;
    let mut out = -1;
    let rc = unsafe { lc_knearest(&dummy, 0, &box_c, 1, std::ptr::null(), 0.0, &mut out) };
    assert_ne!(rc, 0);
    let before = unsafe { std::ffi::CStr::from_ptr(linkcell::lc_last_error()) }
        .to_str()
        .unwrap()
        .to_string();
    let _v = unsafe { std::ffi::CStr::from_ptr(linkcell::lc_version()) };
    let after = unsafe { std::ffi::CStr::from_ptr(linkcell::lc_last_error()) }
        .to_str()
        .unwrap();
    assert_eq!(before, "no points");
    assert_eq!(after, before);
}

#[test]
fn c_abi_overflow_is_not_empty() {
    let box_c = ortho_c(10.0, 10.0, 10.0);
    let dummy = 0.0;
    let mut out = -1;
    let n = (isize::MAX as usize) / 3 + 1;
    let rc = unsafe { lc_knearest(&dummy, n, &box_c, 1, std::ptr::null(), 0.0, &mut out) };
    assert_ne!(rc, 0);
    let msg = unsafe { std::ffi::CStr::from_ptr(linkcell::lc_last_error()) };
    assert_eq!(msg.to_str().unwrap(), "n * k overflows");
    assert_ne!(msg.to_str().unwrap(), "no points");
    assert_ne!(msg.to_str().unwrap(), "out buffer length must be n * k");
}

#[test]
fn lc_knearest_d2_writes_periodic_image() {
    let packed = [0.2, 0.0, 0.0, 9.4, 0.0, 0.0];
    let box_c = ortho_c(10.0, 10.0, 10.0);
    let mut nn = [-1i32; 2];
    let mut d2 = [0.0f64; 2];
    let rc = unsafe {
        lc_knearest_d2(
            packed.as_ptr(),
            2,
            &box_c,
            1,
            std::ptr::null(),
            0.0,
            nn.as_mut_ptr(),
            d2.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0);
    assert_eq!(nn, [1, 0]);
    assert!((d2[0] - 0.64).abs() < 1e-12);
}

#[test]
fn lc_knearest_many_two_frames() {
    let packed = [0.2, 0.0, 0.0, 9.4, 0.0, 0.0, 0.2, 0.0, 0.0, 9.4, 0.0, 0.0];
    let box_c = ortho_c(10.0, 10.0, 10.0);
    let mut nn = [-1i32; 4];
    let mut d2 = [0.0f64; 4];
    let rc = unsafe {
        lc_knearest_many(
            packed.as_ptr(),
            2,
            2,
            &box_c,
            1,
            std::ptr::null(),
            0.0,
            nn.as_mut_ptr(),
            d2.as_mut_ptr(),
        )
    };
    assert_eq!(rc, 0);
    assert_eq!(nn, [1, 0, 1, 0]);
    assert!((d2[3] - 0.64).abs() < 1e-12);
}

#[test]
fn lc_pairs_within_queries_then_writes_the_shift() {
    let packed = [0.2, 0.0, 0.0, 9.4, 0.0, 0.0];
    let box_c = ortho_c(10.0, 10.0, 10.0);
    let mut count = 0usize;
    let rc = unsafe {
        lc_pairs_within(
            packed.as_ptr(),
            2,
            &box_c,
            1.0,
            std::ptr::null(),
            0.0,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            &mut count,
        )
    };
    assert_eq!(rc, 0);
    assert_eq!(count, 2);
    let mut ii = [0i32; 1];
    let mut jj = [0i32; 1];
    let mut shift = [0i32; 3];
    let mut d2 = [0.0f64; 1];
    let short = unsafe {
        lc_pairs_within(
            packed.as_ptr(),
            2,
            &box_c,
            1.0,
            std::ptr::null(),
            0.0,
            0,
            ii.as_mut_ptr(),
            jj.as_mut_ptr(),
            shift.as_mut_ptr(),
            d2.as_mut_ptr(),
            1,
            &mut count,
        )
    };
    assert_ne!(short, 0);
    assert_eq!(count, 2);
    let msg = unsafe { std::ffi::CStr::from_ptr(linkcell::lc_last_error()) };
    assert_eq!(
        msg.to_str().unwrap(),
        "pair buffer is shorter than the pair count"
    );
    let mut ii = [0i32; 2];
    let mut jj = [0i32; 2];
    let mut shift = [0i32; 6];
    let mut d2 = [0.0f64; 2];
    let rc = unsafe {
        lc_pairs_within(
            packed.as_ptr(),
            2,
            &box_c,
            1.0,
            std::ptr::null(),
            0.0,
            0,
            ii.as_mut_ptr(),
            jj.as_mut_ptr(),
            shift.as_mut_ptr(),
            d2.as_mut_ptr(),
            2,
            &mut count,
        )
    };
    assert_eq!(rc, 0);
    assert_eq!(count, 2);
    let row = ii.iter().position(|&i| i == 0).unwrap();
    assert_eq!(jj[row], 1);
    assert_eq!(&shift[3 * row..3 * row + 3], &[-1, 0, 0]);
    assert!((d2[row] - 0.64).abs() < 1e-12);
}

fn lattice_xyz(side: usize, boxl: f64) -> Vec<f64> {
    let mut xyz = Vec::with_capacity(side * side * side * 3);
    for iz in 0..side {
        for iy in 0..side {
            for ix in 0..side {
                xyz.push((ix as f64 + 0.5) * boxl / side as f64);
                xyz.push((iy as f64 + 0.5) * boxl / side as f64);
                xyz.push((iz as f64 + 0.5) * boxl / side as f64);
            }
        }
    }
    xyz
}

fn c_rows(xyz: &[f64], box_c: &LcCell, cutoff: f64, half: c_int) -> Vec<(i32, i32, [i32; 3], u64)> {
    let n = xyz.len() / 3;
    let mut count = 0usize;
    let null = std::ptr::null_mut();
    let rc = unsafe {
        lc_pairs_within(
            xyz.as_ptr(),
            n,
            box_c,
            cutoff,
            std::ptr::null(),
            0.0,
            half,
            null,
            null,
            null,
            std::ptr::null_mut(),
            0,
            &mut count,
        )
    };
    assert_eq!(rc, 0);
    let mut ii = vec![0i32; count];
    let mut jj = vec![0i32; count];
    let mut ss = vec![0i32; 3 * count];
    let mut dd = vec![0.0f64; count];
    let rc = unsafe {
        lc_pairs_within(
            xyz.as_ptr(),
            n,
            box_c,
            cutoff,
            std::ptr::null(),
            0.0,
            half,
            ii.as_mut_ptr(),
            jj.as_mut_ptr(),
            ss.as_mut_ptr(),
            dd.as_mut_ptr(),
            count,
            &mut count,
        )
    };
    assert_eq!(rc, 0);
    let mut rows: Vec<_> = (0..count)
        .map(|t| {
            (
                ii[t],
                jj[t],
                [ss[3 * t], ss[3 * t + 1], ss[3 * t + 2]],
                dd[t].to_bits(),
            )
        })
        .collect();
    rows.sort();
    rows
}

#[test]
fn c_rows_match_the_rust_list() {
    let xyz = lattice_xyz(9, 18.0);
    let pts: Vec<[f64; 3]> = xyz.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let cell = linkcell::Cell::ortho(18.0, 18.0, 18.0).unwrap();
    let box_c = ortho_c(18.0, 18.0, 18.0);
    for half in [false, true] {
        let mut want: Vec<_> = linkcell::pairs_within(&pts, &cell, 4.0, None, None, half)
            .unwrap()
            .iter()
            .map(|p| (p.i as i32, p.j as i32, p.shift, p.dist2.to_bits()))
            .collect();
        want.sort();
        assert_eq!(
            c_rows(&xyz, &box_c, 4.0, half as c_int),
            want,
            "half={half}"
        );

        let mut count = 0usize;
        let rc = unsafe {
            lc_pairs_within_rows(
                xyz.as_ptr(),
                pts.len(),
                &box_c,
                4.0,
                std::ptr::null(),
                0.0,
                half as c_int,
                std::ptr::null_mut(),
                0,
                &mut count,
            )
        };
        assert_eq!(rc, 0);
        assert_eq!(count, want.len());
        let mut rows = vec![linkcell::lc_pair::default(); count];
        let rc = unsafe {
            lc_pairs_within_rows(
                xyz.as_ptr(),
                pts.len(),
                &box_c,
                4.0,
                std::ptr::null(),
                0.0,
                half as c_int,
                rows.as_mut_ptr(),
                count,
                &mut count,
            )
        };
        assert_eq!(rc, 0);
        let mut got: Vec<_> = rows
            .iter()
            .map(|r| (r.i, r.j, r.shift, r.dist2.to_bits()))
            .collect();
        got.sort();
        assert_eq!(got, want, "rows half={half}");
    }
}

#[test]
fn a_parked_query_is_not_reused_for_other_points() {
    let box_c = ortho_c(10.0, 10.0, 10.0);
    let near = [0.2, 0.0, 0.0, 9.4, 0.0, 0.0];
    let far = [0.2, 0.0, 0.0, 5.0, 0.0, 0.0];
    let mut count = 0usize;
    let null = std::ptr::null_mut();
    let rc = unsafe {
        lc_pairs_within(
            near.as_ptr(),
            2,
            &box_c,
            1.0,
            std::ptr::null(),
            0.0,
            0,
            null,
            null,
            null,
            std::ptr::null_mut(),
            0,
            &mut count,
        )
    };
    assert_eq!(rc, 0);
    assert_eq!(count, 2);
    // Same buffers, different points: the fill must search again.
    let mut ii = [7i32; 2];
    let mut jj = [7i32; 2];
    let mut ss = [7i32; 6];
    let mut dd = [7.0f64; 2];
    let rc = unsafe {
        lc_pairs_within(
            far.as_ptr(),
            2,
            &box_c,
            1.0,
            std::ptr::null(),
            0.0,
            0,
            ii.as_mut_ptr(),
            jj.as_mut_ptr(),
            ss.as_mut_ptr(),
            dd.as_mut_ptr(),
            2,
            &mut count,
        )
    };
    assert_eq!(rc, 0);
    assert_eq!(count, 0);
    assert_eq!(ii, [7, 7]);
    // A fill with the queried points still writes both rows.
    let rc = unsafe {
        lc_pairs_within(
            near.as_ptr(),
            2,
            &box_c,
            1.0,
            std::ptr::null(),
            0.0,
            0,
            ii.as_mut_ptr(),
            jj.as_mut_ptr(),
            ss.as_mut_ptr(),
            dd.as_mut_ptr(),
            2,
            &mut count,
        )
    };
    assert_eq!(rc, 0);
    assert_eq!(count, 2);
    let mut got = [(ii[0], jj[0]), (ii[1], jj[1])];
    got.sort();
    assert_eq!(got, [(0, 1), (1, 0)]);
}

#[test]
fn c_pair_layouts_keep_the_input_image_shift() {
    let cell = ortho_c(10.0, 10.0, 10.0);
    let xyz = [0.2, 0.0, 0.0, 29.4, 0.0, 0.0];
    for half in [0, 1] {
        let columns = c_rows(&xyz, &cell, 1.0, half);
        let mut count = 0;
        let mut rows = [linkcell::lc_pair::default(); 2];
        let rc = unsafe {
            lc_pairs_within_rows(
                xyz.as_ptr(),
                2,
                &cell,
                1.0,
                std::ptr::null(),
                0.0,
                half,
                rows.as_mut_ptr(),
                rows.len(),
                &mut count,
            )
        };
        assert_eq!(rc, 0);
        assert_eq!(count, if half == 0 { 2 } else { 1 });
        let mut got: Vec<_> = rows[..count]
            .iter()
            .map(|r| (r.i, r.j, r.shift, r.dist2.to_bits()))
            .collect();
        got.sort();
        assert_eq!(got, columns);
        assert_eq!((got[0].0, got[0].1, got[0].2), (0, 1, [-3, 0, 0]));
        if half == 0 {
            assert_eq!((got[1].0, got[1].1, got[1].2), (1, 0, [3, 0, 0]));
        }
        for r in &got {
            assert!((f64::from_bits(r.3) - 0.64).abs() < 1e-12);
        }
    }
}

extern "C" fn count_hook(ctx: *mut std::ffi::c_void) {
    // SAFETY: the test passes a pointer to an AtomicUsize that outlives every search.
    let hits = unsafe { &*(ctx as *const std::sync::atomic::AtomicUsize) };
    hits.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

/// A continuation set on one thread runs only at the end of that thread's
/// own searches, never inside a search started on another thread.
#[test]
fn continuation_runs_only_on_the_thread_that_set_it() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    const SEARCHES: usize = 200;
    let xyz: Vec<[f64; 3]> = (0..512)
        .map(|i| {
            let i = i as f64;
            [(i * 0.37) % 8.0, (i * 0.61) % 8.0, (i * 0.83) % 8.0]
        })
        .collect();
    let packed = pack_xyz(&xyz);
    let hits: Vec<AtomicUsize> = (0..2).map(|_| AtomicUsize::new(0)).collect();
    std::thread::scope(|s| {
        for (t, slot) in hits.iter().enumerate() {
            let packed = &packed;
            s.spawn(move || {
                let box_c = ortho_c(8.0, 8.0, 8.0);
                let ctx = slot as *const AtomicUsize as *mut std::ffi::c_void;
                linkcell::lc_set_continuation(ctx, Some(count_hook));
                let mut out = vec![-1; 512 * 4];
                for _ in 0..SEARCHES {
                    let rc = unsafe {
                        lc_knearest(
                            packed.as_ptr(),
                            512,
                            &box_c,
                            4,
                            std::ptr::null(),
                            0.0,
                            out.as_mut_ptr(),
                        )
                    };
                    assert_eq!(rc, 0, "thread {t}");
                }
                linkcell::lc_set_continuation(std::ptr::null_mut(), None);
            });
        }
        // A third thread searches with no hook of its own.
        let packed = &packed;
        s.spawn(move || {
            let box_c = ortho_c(8.0, 8.0, 8.0);
            let mut out = vec![-1; 512 * 4];
            for _ in 0..SEARCHES {
                let rc = unsafe {
                    lc_knearest(
                        packed.as_ptr(),
                        512,
                        &box_c,
                        4,
                        std::ptr::null(),
                        0.0,
                        out.as_mut_ptr(),
                    )
                };
                assert_eq!(rc, 0, "hookless thread");
            }
        });
    });
    for (t, slot) in hits.iter().enumerate() {
        assert_eq!(
            slot.load(Ordering::SeqCst),
            SEARCHES,
            "thread {t} hook count"
        );
    }
}

fn last_error() -> String {
    let msg = linkcell::lc_last_error();
    assert!(!msg.is_null());
    unsafe { std::ffi::CStr::from_ptr(msg) }
        .to_str()
        .unwrap()
        .to_owned()
}

/// Three points with `bad` at axis `axis` of point 1, packed.
fn poisoned_c(bad: f64, axis: usize) -> [f64; 9] {
    let mut packed = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0];
    packed[3 + axis] = bad;
    packed
}

const NON_FINITE: [f64; 3] = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

#[test]
fn lc_knearest_entries_reject_non_finite_coordinates() {
    let box_c = ortho_c(10.0, 10.0, 10.0);
    let want = Error::NonFinite.to_string();
    for bad in NON_FINITE {
        for axis in 0..3 {
            let packed = poisoned_c(bad, axis);
            let mut nn = [7i32; 6];
            let rc = unsafe {
                lc_knearest(
                    packed.as_ptr(),
                    3,
                    &box_c,
                    2,
                    std::ptr::null(),
                    0.0,
                    nn.as_mut_ptr(),
                )
            };
            assert_ne!(rc, 0);
            assert_eq!(last_error(), want);
            assert_eq!(nn, [-1; 6]);

            let mut nn = [7i32; 6];
            let mut d2 = [1.0f64; 6];
            let rc = unsafe {
                lc_knearest_d2(
                    packed.as_ptr(),
                    3,
                    &box_c,
                    2,
                    std::ptr::null(),
                    0.0,
                    nn.as_mut_ptr(),
                    d2.as_mut_ptr(),
                )
            };
            assert_ne!(rc, 0);
            assert_eq!(last_error(), want);
            assert_eq!(nn, [-1; 6]);
            assert!(d2.iter().all(|v| v.is_nan()));
        }
    }
}

#[test]
fn lc_knearest_many_rejects_a_non_finite_later_frame() {
    let box_c = ortho_c(10.0, 10.0, 10.0);
    for bad in NON_FINITE {
        let mut packed = [0.0f64; 18];
        packed[..9].copy_from_slice(&poisoned_c(0.0, 0));
        packed[9..].copy_from_slice(&poisoned_c(bad, 2));
        let mut nn = [7i32; 12];
        let mut d2 = [1.0f64; 12];
        let rc = unsafe {
            lc_knearest_many(
                packed.as_ptr(),
                3,
                2,
                &box_c,
                2,
                std::ptr::null(),
                0.0,
                nn.as_mut_ptr(),
                d2.as_mut_ptr(),
            )
        };
        assert_ne!(rc, 0);
        assert_eq!(last_error(), Error::NonFinite.to_string());
        assert_eq!(nn, [-1; 12]);
        assert!(d2.iter().all(|v| v.is_nan()));
    }
}

extern "C" fn csr_alloc(ctx: *mut std::ffi::c_void, total: usize) -> *mut c_int {
    let cols = unsafe { &mut *(ctx as *mut Vec<c_int>) };
    cols.resize(total, 0);
    cols.as_mut_ptr()
}

fn c_csr(packed: &[f64], n: usize, box_c: &linkcell::lc_cell, mutual: c_int) -> c_int {
    let mut offsets = vec![0 as c_int; n + 1];
    let mut cols: Vec<c_int> = Vec::new();
    unsafe {
        linkcell::lc_knearest_csr(
            packed.as_ptr(),
            n,
            box_c,
            2,
            std::ptr::null(),
            0.0,
            mutual,
            offsets.as_mut_ptr(),
            Some(csr_alloc),
            (&mut cols as *mut Vec<c_int>).cast(),
        )
    }
}

fn ortho_lc(l: f64) -> linkcell::lc_cell {
    linkcell::lc_cell {
        ax: l,
        ay: 0.0,
        az: 0.0,
        bx: 0.0,
        by: l,
        bz: 0.0,
        cx: 0.0,
        cy: 0.0,
        cz: l,
        ox: 0.0,
        oy: 0.0,
        oz: 0.0,
    }
}

#[test]
fn lc_knearest_csr_rejects_non_finite_coordinates() {
    let box_c = ortho_lc(10.0);
    assert_eq!(c_csr(&poisoned_c(0.0, 0), 3, &box_c, 0), 0);
    for bad in NON_FINITE {
        for axis in 0..3 {
            for mutual in [0, 1] {
                assert_ne!(c_csr(&poisoned_c(bad, axis), 3, &box_c, mutual), 0);
                assert_eq!(last_error(), Error::NonFinite.to_string());
            }
        }
    }
}

#[test]
fn lc_knearest_entries_reject_a_non_finite_origin() {
    let packed = poisoned_c(0.0, 0);
    for bad in NON_FINITE {
        let mut box_c = ortho_lc(10.0);
        box_c.oy = bad;
        let mut nn = [7i32; 6];
        let rc = unsafe {
            linkcell::lc_knearest(
                packed.as_ptr(),
                3,
                &box_c,
                2,
                std::ptr::null(),
                0.0,
                nn.as_mut_ptr(),
            )
        };
        assert_ne!(rc, 0);
        assert_eq!(last_error(), Error::NonFinite.to_string());
        assert_ne!(c_csr(&packed, 3, &box_c, 0), 0);
        assert_eq!(last_error(), Error::NonFinite.to_string());
    }
}

#[test]
fn lc_knearest_entries_reject_non_finite_box_lengths() {
    let packed = poisoned_c(0.0, 0);
    let want = Error::BadBox.to_string();
    for bad in NON_FINITE {
        for slot in 0..9 {
            let mut box_c = ortho_lc(10.0);
            match slot {
                0 => box_c.ax = bad,
                1 => box_c.ay = bad,
                2 => box_c.az = bad,
                3 => box_c.bx = bad,
                4 => box_c.by = bad,
                5 => box_c.bz = bad,
                6 => box_c.cx = bad,
                7 => box_c.cy = bad,
                _ => box_c.cz = bad,
            }
            let mut nn = [7i32; 6];
            let rc = unsafe {
                linkcell::lc_knearest(
                    packed.as_ptr(),
                    3,
                    &box_c,
                    2,
                    std::ptr::null(),
                    0.0,
                    nn.as_mut_ptr(),
                )
            };
            assert_ne!(rc, 0);
            assert_eq!(last_error(), want);
            assert_ne!(c_csr(&packed, 3, &box_c, 0), 0);
            assert_eq!(last_error(), want);
        }
    }
}

fn build_nd_index(data: &[f64], n: usize, d: usize, periods: Option<&[f64]>) -> *mut LcNdIndex {
    let mut out = std::ptr::null_mut();
    let rc = unsafe {
        lc_nd_build(
            data.as_ptr(),
            n,
            d,
            periods.map_or(std::ptr::null(), |p| p.as_ptr()),
            &mut out,
        )
    };
    assert_eq!(rc, 0, "build failed: {}", last_error());
    out
}

#[test]
fn lc_nd_build_and_queries_work() {
    let data = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let idx = build_nd_index(&data, 3, 2, None);
    assert_eq!(unsafe { lc_nd_len(idx) }, 3);
    assert_eq!(unsafe { lc_nd_dims(idx) }, 2);

    let mut nn = [-1; 5];
    let mut d2 = [f64::NAN; 5];
    let q = [0.0, 0.0];
    let rc = unsafe { lc_nd_knn(idx, q.as_ptr(), 5, nn.as_mut_ptr(), d2.as_mut_ptr()) };
    assert_eq!(rc, 0);
    assert_eq!(nn[0], 0);
    assert_eq!(d2[0], 0.0);

    let mut need = 0usize;
    let rc = unsafe {
        lc_nd_within(
            idx,
            q.as_ptr(),
            1.01,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            &mut need,
        )
    };
    assert_eq!(rc, 0);
    let mut out_i = vec![-1; need];
    let mut out_d2 = vec![0.0; need];
    let rc = unsafe {
        lc_nd_within(
            idx,
            q.as_ptr(),
            1.01,
            out_i.as_mut_ptr(),
            out_d2.as_mut_ptr(),
            need,
            &mut need,
        )
    };
    assert_eq!(rc, 0);
    assert!(out_i.contains(&0));

    let mut labels = vec![-9; 3];
    let mut core = vec![0; 3];
    let mut clusters = 0usize;
    let rc = unsafe {
        lc_nd_dbscan(
            idx,
            1.1,
            2,
            labels.as_mut_ptr(),
            core.as_mut_ptr(),
            &mut clusters,
        )
    };
    assert_eq!(rc, 0);
    assert!(clusters >= 1);

    unsafe { lc_nd_free(idx) };
}

#[test]
fn lc_nd_rejects_bad_dimensions_nulls_and_bad_periods() {
    let data = [0.0, 1.0];
    let periods = [1.0];
    let mut out = 0x123usize as *mut LcNdIndex;
    assert_ne!(
        unsafe { lc_nd_build(std::ptr::null(), 1, 1, periods.as_ptr(), &mut out) },
        0
    );
    assert!(out.is_null());
    out = 0x123usize as *mut LcNdIndex;
    assert_ne!(
        unsafe { lc_nd_build(data.as_ptr(), 1, 0, periods.as_ptr(), &mut out) },
        0
    );
    assert!(out.is_null());
    assert_ne!(
        unsafe { lc_nd_build(data.as_ptr(), 1, 1, periods.as_ptr(), std::ptr::null_mut()) },
        0
    );
    let bad_neg = [-1.0f64];
    out = 0x123usize as *mut LcNdIndex;
    assert_ne!(
        unsafe { lc_nd_build(data.as_ptr(), 1, 1, bad_neg.as_ptr(), &mut out) },
        0
    );
    assert!(out.is_null());
    let bad_nan = [f64::NAN];
    out = 0x123usize as *mut LcNdIndex;
    assert_ne!(
        unsafe { lc_nd_build(data.as_ptr(), 1, 1, bad_nan.as_ptr(), &mut out) },
        0
    );
    assert!(out.is_null());

    // n * d overflow before any read. `data` is two doubles; a late check
    // would form a slice of `usize::MAX * 2` and read it.
    out = 0x123usize as *mut LcNdIndex;
    assert_ne!(
        unsafe { lc_nd_build(data.as_ptr(), usize::MAX, 2, std::ptr::null(), &mut out) },
        0
    );
    assert!(out.is_null());
    assert_eq!(last_error(), "n * d overflows");

    // Non-finite payload rejection.
    let nonfinite = [f64::INFINITY];
    out = 0x123usize as *mut LcNdIndex;
    assert_ne!(
        unsafe { lc_nd_build(nonfinite.as_ptr(), 1, 1, std::ptr::null(), &mut out) },
        0
    );
    assert!(out.is_null());
}

#[test]
fn lc_nd_count_queries_report_short_buffers() {
    let data = [
        0.0, 0.0, //
        0.1, 0.0, //
        0.0, 0.1, //
        0.09, 0.09,
    ];
    let idx = build_nd_index(&data, 4, 2, None);
    let q = [0.0, 0.0];
    let mut need = 0usize;
    let rc = unsafe {
        lc_nd_within(
            idx,
            q.as_ptr(),
            0.2,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            &mut need,
        )
    };
    assert_eq!(rc, 0);
    let mut got = need;
    let mut short_i = vec![-1; need.saturating_sub(1)];
    let rc = unsafe {
        lc_nd_within(
            idx,
            q.as_ptr(),
            0.2,
            short_i.as_mut_ptr(),
            std::ptr::null_mut(),
            short_i.len(),
            &mut got,
        )
    };
    assert_ne!(rc, 0);
    assert_eq!(got, need);

    let mut need_pairs = 0usize;
    let rc = unsafe {
        lc_nd_pairs_within(
            idx,
            0.2,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            &mut need_pairs,
        )
    };
    assert_eq!(rc, 0);
    let mut out_i = vec![-1; need_pairs.saturating_sub(1)];
    let mut out_j = vec![-1; need_pairs.saturating_sub(1)];
    let mut got_pairs = need_pairs;
    let rc = unsafe {
        lc_nd_pairs_within(
            idx,
            0.2,
            out_i.as_mut_ptr(),
            out_j.as_mut_ptr(),
            std::ptr::null_mut(),
            out_i.len(),
            &mut got_pairs,
        )
    };
    assert_ne!(rc, 0);
    assert_eq!(got_pairs, need_pairs);
    unsafe { lc_nd_free(idx) };
}

#[test]
fn lc_nd_rejects_non_finite_coordinates_and_null_query_pointers() {
    let nan_data = [f64::NAN, 0.0];
    let mut out = std::ptr::null_mut();
    assert_ne!(
        unsafe { lc_nd_build(nan_data.as_ptr(), 1, 2, std::ptr::null(), &mut out) },
        0
    );
    assert_eq!(last_error(), Error::NonFinite.to_string());

    let good_data = [0.0, 0.0, 1.0, 0.0];
    let idx = build_nd_index(&good_data, 2, 2, None);
    let mut nn = [0; 2];
    let q = [0.0, 0.0];
    assert_ne!(
        unsafe {
            lc_nd_knn(
                std::ptr::null(),
                q.as_ptr(),
                2,
                nn.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        },
        0
    );
    assert_ne!(
        unsafe {
            lc_nd_knn(
                idx,
                std::ptr::null(),
                2,
                nn.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        },
        0
    );
    let q_nan = [f64::NAN, 0.0];
    assert_ne!(
        unsafe {
            lc_nd_knn(
                idx,
                q_nan.as_ptr(),
                2,
                nn.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        },
        0
    );
    let mut count = 0usize;
    assert_ne!(
        unsafe {
            lc_nd_within(
                idx,
                std::ptr::null(),
                1.0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                &mut count,
            )
        },
        0
    );
    assert_ne!(
        unsafe {
            lc_nd_pairs_within(
                idx,
                1.0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
            )
        },
        0
    );
    assert_ne!(
        unsafe {
            lc_nd_dbscan(
                idx,
                1.0,
                2,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        },
        0
    );
    assert_eq!(unsafe { lc_nd_len(std::ptr::null()) }, 0);
    assert_eq!(unsafe { lc_nd_dims(std::ptr::null()) }, 0);
    unsafe { lc_nd_free(idx) };
}

type NdHits = Result<Vec<(usize, u64)>, String>;
type NdRows = Result<Vec<(usize, usize, u64)>, String>;
type NdLabels = Result<(Vec<c_int>, Vec<c_int>, usize), String>;

/// A full scan under the crate's own `dist2`, which every `lc_nd_*` query
/// must match. A point is inside radius `r` when `dist2 <= fl(r * r)`; when
/// that product is 0, only a point that coincides with the query is. Hits
/// are ordered by exact square, then index. A query fails when any distance
/// it reads overflows, and `k = 0` reads none.
struct NdScan<'a> {
    space: PeriodicSpace,
    data: &'a [f64],
    d: usize,
}

impl<'a> NdScan<'a> {
    /// `periods` is dense, as `lc_nd_build` takes it: 0 is an open axis.
    fn new(data: &'a [f64], d: usize, periods: &[f64]) -> Self {
        let periods = periods.iter().map(|&l| (l > 0.0).then_some(l)).collect();
        Self {
            space: PeriodicSpace::new(periods).unwrap(),
            data,
            d,
        }
    }

    fn n(&self) -> usize {
        self.data.len() / self.d
    }

    fn point(&self, i: usize) -> &[f64] {
        &self.data[i * self.d..(i + 1) * self.d]
    }

    fn dist2(&self, a: &[f64], b: &[f64]) -> Result<f64, String> {
        self.space.dist2(a, b).map_err(|err| {
            assert_eq!(err, Error::DistanceOverflow);
            err.to_string()
        })
    }

    fn inside(&self, a: &[f64], b: &[f64], d2: f64, r2: f64) -> bool {
        d2 < r2 || (d2 == r2 && (r2 > 0.0 || self.space.dist2_exact(a, b).unwrap().is_zero()))
    }

    fn ranked(&self, q: &[f64], mut hits: Vec<(usize, f64)>) -> Vec<(usize, u64)> {
        hits.sort_by_cached_key(|&(i, _)| (self.space.dist2_exact(self.point(i), q).unwrap(), i));
        hits.into_iter().map(|(i, d2)| (i, d2.to_bits())).collect()
    }

    fn knn(&self, q: &[f64], k: usize) -> NdHits {
        if k == 0 {
            return Ok(Vec::new());
        }
        let hits = (0..self.n())
            .map(|i| Ok((i, self.dist2(self.point(i), q)?)))
            .collect::<Result<Vec<_>, String>>()?;
        let mut ranked = self.ranked(q, hits);
        ranked.truncate(k);
        Ok(ranked)
    }

    fn within(&self, q: &[f64], r: f64) -> NdHits {
        let r2 = r * r;
        let mut hits = Vec::new();
        for i in 0..self.n() {
            let d2 = self.dist2(self.point(i), q)?;
            if self.inside(self.point(i), q, d2, r2) {
                hits.push((i, d2));
            }
        }
        Ok(self.ranked(q, hits))
    }

    fn pairs(&self, r: f64) -> NdRows {
        let r2 = r * r;
        let mut rows = Vec::new();
        for i in 0..self.n() {
            for j in i + 1..self.n() {
                let (a, b) = (self.point(i), self.point(j));
                let d2 = self.dist2(a, b)?;
                if self.inside(a, b, d2, r2) {
                    rows.push((i, j, d2.to_bits()));
                }
            }
        }
        Ok(rows)
    }

    /// Labels, core flags and the cluster count. Seeds are taken in index
    /// order, and each expansion visits neighbours in index order.
    fn dbscan(&self, eps: f64, min_pts: usize) -> NdLabels {
        let n = self.n();
        let mut near: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        for (i, j, _) in self.pairs(eps)? {
            near[i].push(j);
            near[j].push(i);
        }
        for list in &mut near {
            list.sort_unstable();
        }
        let core: Vec<c_int> = near
            .iter()
            .map(|list| c_int::from(list.len() >= min_pts))
            .collect();
        let mut labels: Vec<c_int> = vec![-2; n];
        let mut clusters = 0;
        for seed in 0..n {
            if labels[seed] != -2 {
                continue;
            }
            if core[seed] == 0 {
                labels[seed] = -1;
                continue;
            }
            labels[seed] = clusters;
            let mut queued = vec![false; n];
            let mut queue = std::collections::VecDeque::new();
            for &j in &near[seed] {
                if j != seed {
                    queued[j] = true;
                    queue.push_back(j);
                }
            }
            while let Some(p) = queue.pop_front() {
                if labels[p] == -1 {
                    labels[p] = clusters;
                }
                if labels[p] != -2 {
                    continue;
                }
                labels[p] = clusters;
                if core[p] == 1 {
                    for &j in &near[p] {
                        if !queued[j] {
                            queued[j] = true;
                            queue.push_back(j);
                        }
                    }
                }
            }
            clusters += 1;
        }
        Ok((labels, core, clusters as usize))
    }
}

/// `lc_nd_knn`, which writes `k` slots: the hits, then `-1` / `NaN`.
fn c_nd_knn(idx: *const LcNdIndex, q: &[f64], k: usize) -> NdHits {
    let mut ids: Vec<c_int> = vec![-7; k];
    let mut d2 = vec![-7.0; k];
    if unsafe { lc_nd_knn(idx, q.as_ptr(), k, ids.as_mut_ptr(), d2.as_mut_ptr()) } != 0 {
        return Err(last_error());
    }
    let found = ids.iter().take_while(|&&j| j >= 0).count();
    assert!(
        ids[found..].iter().all(|&j| j == -1) && d2[found..].iter().all(|v| v.is_nan()),
        "lc_nd_knn tail {ids:?} {d2:?}"
    );
    Ok(ids[..found]
        .iter()
        .zip(&d2)
        .map(|(&j, v)| (j as usize, v.to_bits()))
        .collect())
}

/// `lc_nd_within`: a count with null outputs, then a fill of that size.
fn c_nd_within(idx: *const LcNdIndex, q: &[f64], r: f64) -> NdHits {
    let null = std::ptr::null_mut();
    let mut need = usize::MAX;
    if unsafe { lc_nd_within(idx, q.as_ptr(), r, null, null.cast(), 0, &mut need) } != 0 {
        return Err(last_error());
    }
    let mut ids: Vec<c_int> = vec![-7; need];
    let mut d2 = vec![-7.0; need];
    let mut count = usize::MAX;
    let rc = unsafe {
        lc_nd_within(
            idx,
            q.as_ptr(),
            r,
            ids.as_mut_ptr(),
            d2.as_mut_ptr(),
            need,
            &mut count,
        )
    };
    assert_eq!(
        (rc, count),
        (0, need),
        "lc_nd_within fill: {}",
        last_error()
    );
    Ok(ids
        .iter()
        .zip(&d2)
        .map(|(&j, v)| (j as usize, v.to_bits()))
        .collect())
}

/// `lc_nd_pairs_within`: a count with null outputs, then a fill.
fn c_nd_pairs(idx: *const LcNdIndex, r: f64) -> NdRows {
    let null = std::ptr::null_mut();
    let mut need = usize::MAX;
    if unsafe { lc_nd_pairs_within(idx, r, null, null, null.cast(), 0, &mut need) } != 0 {
        return Err(last_error());
    }
    let mut i: Vec<c_int> = vec![-7; need];
    let mut j: Vec<c_int> = vec![-7; need];
    let mut d2 = vec![-7.0; need];
    let mut count = usize::MAX;
    let rc = unsafe {
        lc_nd_pairs_within(
            idx,
            r,
            i.as_mut_ptr(),
            j.as_mut_ptr(),
            d2.as_mut_ptr(),
            need,
            &mut count,
        )
    };
    assert_eq!(
        (rc, count),
        (0, need),
        "lc_nd_pairs_within fill: {}",
        last_error()
    );
    Ok((0..need)
        .map(|row| (i[row] as usize, j[row] as usize, d2[row].to_bits()))
        .collect())
}

fn c_nd_dbscan(idx: *const LcNdIndex, n: usize, eps: f64, min_pts: usize) -> NdLabels {
    let mut labels: Vec<c_int> = vec![-7; n];
    let mut core: Vec<c_int> = vec![-7; n];
    let mut clusters = usize::MAX;
    let rc = unsafe {
        lc_nd_dbscan(
            idx,
            eps,
            min_pts,
            labels.as_mut_ptr(),
            core.as_mut_ptr(),
            &mut clusters,
        )
    };
    if rc != 0 {
        return Err(last_error());
    }
    Ok((labels, core, clusters))
}

/// Builds `data` through the C ABI and compares every query with a full
/// scan, errors included. Returns how many queries failed and how many
/// succeeded.
fn assert_c_full_scan(
    data: &[f64],
    d: usize,
    periods: &[f64],
    queries: &[Vec<f64>],
    radii: &[f64],
    what: &str,
) -> [usize; 2] {
    let scan = NdScan::new(data, d, periods);
    let n = scan.n();
    let idx = build_nd_index(data, n, d, Some(periods));
    let mut seen = [0usize; 2];
    let mut tally = |ok: bool| seen[usize::from(ok)] += 1;
    for q in queries {
        for k in [0, 1, 2, n + 1] {
            let want = scan.knn(q, k);
            tally(want.is_ok());
            assert_eq!(c_nd_knn(idx, q, k), want, "{what}: lc_nd_knn({q:?}, {k})");
        }
        for &r in radii {
            let want = scan.within(q, r);
            tally(want.is_ok());
            assert_eq!(
                c_nd_within(idx, q, r),
                want,
                "{what}: lc_nd_within({q:?}, {r:e})"
            );
        }
    }
    for &r in radii {
        let want = scan.pairs(r);
        tally(want.is_ok());
        assert_eq!(
            c_nd_pairs(idx, r),
            want,
            "{what}: lc_nd_pairs_within({r:e})"
        );
    }
    for &eps in radii.iter().filter(|&&r| r > 0.0) {
        for min_pts in [1, 2] {
            let want = scan.dbscan(eps, min_pts);
            tally(want.is_ok());
            assert_eq!(
                c_nd_dbscan(idx, n, eps, min_pts),
                want,
                "{what}: lc_nd_dbscan({eps:e}, {min_pts})"
            );
        }
    }
    unsafe { lc_nd_free(idx) };
    seen
}

/// Largest value whose square is finite. A gap of `1.01` times it overflows.
fn nd_huge() -> f64 {
    f64::from_bits(0x5fef_ffff_ffff_ffff)
}

/// Every ordering of `0..n`.
fn nd_orderings(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for rest in nd_orderings(n - 1) {
        for slot in 0..n {
            let mut order = rest.clone();
            order.insert(slot, n - 1);
            out.push(order);
        }
    }
    out
}

/// Small sets of huge, tiny, duplicate and signed coordinates on 1 to 3
/// axes, open and periodic, in every build order: each `lc_nd_*` result or
/// error equals a full scan.
#[test]
fn lc_nd_queries_match_a_full_scan_in_every_order() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let x = nd_huge();
    let palette = [
        x,
        -x,
        0.5 * x,
        -0.5 * x,
        -0.1 * x,
        0.3 * x,
        1.0e153,
        -1.0e153,
        1.0,
        -1.0,
        0.0,
        1.0e-170,
        -1.0e-170,
        2.0f64.powi(-600),
    ];
    let axes = [0.0, 8.0 * x, 4.0 * x, 3.5 * x, 4.0];
    let radii = [0.0, 1.0e-300, 1.0e-170, 1.0, 1.0e100, 1.0e153, 1.0e300];
    let mut rng = StdRng::seed_from_u64(0x0cab_1a11);
    let mut seen = [0usize; 2];
    for case in 0..30 {
        let d = 1 + case % 3;
        let n = 2 + (case / 3) % 3;
        let periods: Vec<f64> = (0..d)
            .map(|_| axes[rng.random_range(0..axes.len())])
            .collect();
        let mut row = || -> Vec<f64> {
            (0..d)
                .map(|_| palette[rng.random_range(0..palette.len())])
                .collect()
        };
        let rows: Vec<Vec<f64>> = (0..n).map(|_| row()).collect();
        let mut queries = rows.clone();
        queries.push(row());
        for order in nd_orderings(n) {
            let data: Vec<f64> = order
                .iter()
                .flat_map(|&i| rows[i].iter().copied())
                .collect();
            let got = assert_c_full_scan(
                &data,
                d,
                &periods,
                &queries,
                &radii,
                &format!("{periods:?} {data:?}"),
            );
            seen[0] += got[0];
            seen[1] += got[1];
        }
    }
    assert!(seen[0] > 0 && seen[1] > 0, "{seen:?}");
}

/// The overflow repros through the C ABI. `x`, `-x/2`, `0` and
/// `x`, `-x/10`, `3x/10`, `-1` hold a pair whose square overflows, on the
/// last of 1 to 3 axes. Every build order fails every query that reads it.
#[test]
fn lc_nd_overflow_repros_fail_in_every_order() {
    let x = nd_huge();
    let overflow = Error::DistanceOverflow.to_string();
    let radii = [0.0, 1.0e-300, 1.0, 1.0e100, 1.0e153];
    for axis in [0.0, 8.0 * x, 4.0 * x, 3.5 * x] {
        for d in 1..=3 {
            let mut periods = vec![0.0; d];
            periods[d - 1] = axis;
            let lift = |v: f64, i: usize| -> Vec<f64> {
                let mut row: Vec<f64> = (0..d).map(|k| (i + k) as f64 * 0.125).collect();
                row[d - 1] = v;
                row
            };
            let q = lift(-0.5 * x, 1);
            for values in [&[x, -0.5 * x, 0.0][..], &[x, -0.1 * x, 0.3 * x, -1.0][..]] {
                let n = values.len();
                for order in nd_orderings(n) {
                    let data: Vec<f64> = order.iter().flat_map(|&i| lift(values[i], i)).collect();
                    let what = format!("{periods:?} {order:?} {values:?}");
                    let idx = build_nd_index(&data, n, d, Some(&periods));
                    if n == 3 {
                        for k in [1, 2] {
                            assert_eq!(c_nd_knn(idx, &q, k).unwrap_err(), overflow, "{what}");
                        }
                        for r in radii {
                            let got = c_nd_within(idx, &q, r).unwrap_err();
                            assert_eq!(got, overflow, "{what} {r:e}");
                        }
                        assert!(c_nd_within(idx, &lift(0.0, 2), 1.0).is_ok(), "{what}");
                    }
                    assert_eq!(c_nd_pairs(idx, 1.0).unwrap_err(), overflow, "{what}");
                    let got = c_nd_dbscan(idx, n, 1.0, 1).unwrap_err();
                    assert_eq!(got, overflow, "{what}");
                    unsafe { lc_nd_free(idx) };
                    let queries = [q.clone(), lift(0.0, 2), lift(x, 0)];
                    assert_c_full_scan(&data, d, &periods, &queries, &radii, &what);
                }
            }
        }
    }
}

/// Rounded squares that tie keep the order of their exact squares, and a
/// radius whose square is 0 admits only points that coincide with the query.
#[test]
fn lc_nd_orders_rounded_ties_by_the_exact_square() {
    let idx = build_nd_index(&[1.0e-170, 0.0], 2, 1, None);
    assert_eq!(c_nd_knn(idx, &[0.0], 1), Ok(vec![(1, 0)]));
    assert_eq!(c_nd_knn(idx, &[0.0], 2), Ok(vec![(1, 0), (0, 0)]));
    for r in [0.0, 1.0e-300, 1.0e-170] {
        assert_eq!(c_nd_within(idx, &[0.0], r), Ok(vec![(1, 0)]), "{r:e}");
    }
    assert_eq!(c_nd_pairs(idx, 1.0e-170), Ok(Vec::new()));
    assert_eq!(c_nd_pairs(idx, 1.0), Ok(vec![(0, 1, 0)]));
    unsafe { lc_nd_free(idx) };

    let tiny = 2.0f64.powi(-600);
    let idx = build_nd_index(&[tiny, 0.0], 2, 1, None);
    assert_eq!(c_nd_within(idx, &[0.0], 0.0), Ok(vec![(1, 0)]));
    assert_eq!(c_nd_within(idx, &[tiny], 0.0), Ok(vec![(0, 0)]));
    unsafe { lc_nd_free(idx) };
}

/// Points `i * 1e-163` have subnormal squares. `fl(27e-163 * 27e-163)` is
/// `2^-1074`, and the first 28 squares round to at most that, so radius
/// `27e-163` from the origin returns those 28 points.
#[test]
fn lc_nd_subnormal_squares_match_a_full_scan() {
    let n = 64;
    let r = 27.0e-163;
    assert_eq!(r * r, f64::from_bits(1));
    let radii = [0.0, 1.0e-163, 16.0e-163, 27.0e-163, 28.0e-163, 1.0e-161];
    for (d, period) in [(1, 0.0), (1, 1.0), (2, 0.0), (2, 1.0), (48, 0.0)] {
        let data: Vec<f64> = (0..n)
            .flat_map(|i| std::iter::repeat(i as f64 * 1.0e-163).take(d))
            .collect();
        let periods = vec![period; d];
        let idx = build_nd_index(&data, n, d, Some(&periods));
        let origin = vec![0.0; d];
        let got = c_nd_within(idx, &origin, r).unwrap();
        let want = NdScan::new(&data, d, &periods).within(&origin, r).unwrap();
        assert_eq!(got, want, "d={d} period={period}");
        if d == 1 {
            let ids: Vec<usize> = got.iter().map(|h| h.0).collect();
            assert_eq!(ids, (0..28).collect::<Vec<_>>());
        }
        unsafe { lc_nd_free(idx) };
        let queries: Vec<Vec<f64>> = [0usize, 5, 27, 63]
            .iter()
            .map(|&i| data[i * d..(i + 1) * d].to_vec())
            .collect();
        let what = format!("subnormal d={d} period={period}");
        assert_c_full_scan(&data, d, &periods, &queries, &radii, &what);
    }
}

/// Period-table limits of `lc_nd_build`.
///
/// A product that fits, including `n == 0`, builds. The next `n` whose
/// product does not fit in `usize` fails before the coordinates are read.
/// `d` above `(isize::MAX as usize) / size_of::<Option<f64>>()` fails before
/// allocation. That largest `d` itself asks for a table of `isize::MAX - 15`
/// bytes and fails when the allocator refuses it.
#[test]
fn lc_nd_build_huge_dims_fails_and_leaves_null_out() {
    let sample = [0.0f64, 0.0, 0.0, 0.0];
    let build = |n: usize, d: usize| {
        let mut out = 0x1usize as *mut LcNdIndex;
        let rc = unsafe { lc_nd_build(sample.as_ptr(), n, d, std::ptr::null(), &mut out) };
        (rc, out)
    };
    for (n, d) in [(0usize, 1usize), (1, 1), (2, 2)] {
        let (rc, out) = build(n, d);
        assert_eq!(rc, 0, "n={n} d={d}");
        assert!(!out.is_null());
        unsafe { lc_nd_free(out) };
    }

    let (rc, out) = build(usize::MAX / 2 + 1, 2);
    assert_ne!(rc, 0);
    assert!(out.is_null());
    assert_eq!(last_error(), "n * d overflows");

    let largest = isize::MAX as usize / std::mem::size_of::<Option<f64>>();
    for d in [largest + 1, usize::MAX] {
        let (rc, out) = build(0, d);
        assert_ne!(rc, 0, "d={d}");
        assert!(out.is_null());
        assert_eq!(
            last_error(),
            "the period table does not fit in isize::MAX bytes"
        );
    }

    let (rc, out) = build(0, largest);
    assert_ne!(rc, 0);
    assert!(out.is_null());
    assert_eq!(last_error(), "the period table could not be allocated");
}

/// Kernel id of the calling thread. The value is one of the ids
/// [`thread_ids`] returns.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn current_thread_id() -> u64 {
    // The link is `<pid>/task/<tid>` relative to `/proc`, not a bare tid.
    let name = std::fs::read_link("/proc/thread-self")
        .unwrap_or_else(|err| panic!("/proc/thread-self: {err}"));
    let text = name
        .file_name()
        .and_then(|part| part.to_str())
        .unwrap_or_else(|| panic!("/proc/thread-self has no tid"));
    text.parse()
        .unwrap_or_else(|err| panic!("/proc/thread-self: {err}"))
}

/// Kernel id of the calling thread. The value is one of the ids
/// [`thread_ids`] returns.
#[cfg(target_os = "macos")]
fn current_thread_id() -> u64 {
    extern "C" {
        fn pthread_self() -> *mut std::ffi::c_void;
        fn pthread_threadid_np(thread: *mut std::ffi::c_void, id: *mut u64) -> c_int;
    }
    let mut id = 0u64;
    let rc = unsafe { pthread_threadid_np(pthread_self(), &mut id) };
    assert_eq!(rc, 0, "pthread_threadid_np returned {rc}");
    id
}

/// A Windows thread-pool worker stays in Toolhelp after its callback returns.
/// `ntdll` does not start it through this executable's `CreateThread`, so
/// the probe must not count it. Linkcell's own `std::thread` still goes
/// through that import and is counted by the spawn self-check.
#[cfg(windows)]
fn assert_threadpool_worker_is_not_linkcells() {
    #[link(name = "kernel32")]
    extern "system" {
        fn CreateThreadpoolWork(
            callback: Option<
                unsafe extern "system" fn(
                    *mut std::ffi::c_void,
                    *mut std::ffi::c_void,
                    *mut std::ffi::c_void,
                ),
            >,
            ctx: *mut std::ffi::c_void,
            env: *mut std::ffi::c_void,
        ) -> *mut std::ffi::c_void;
        fn SubmitThreadpoolWork(work: *mut std::ffi::c_void);
        fn WaitForThreadpoolWorkCallbacks(work: *mut std::ffi::c_void, cancel: i32);
        fn CloseThreadpoolWork(work: *mut std::ffi::c_void);
    }
    unsafe extern "system" fn done(
        _instance: *mut std::ffi::c_void,
        _ctx: *mut std::ffi::c_void,
        _work: *mut std::ffi::c_void,
    ) {
    }
    let before = thread_ids();
    let spawned = nd_spawn::count();
    let work =
        unsafe { CreateThreadpoolWork(Some(done), std::ptr::null_mut(), std::ptr::null_mut()) };
    assert!(!work.is_null(), "CreateThreadpoolWork failed");
    unsafe {
        SubmitThreadpoolWork(work);
        WaitForThreadpoolWorkCallbacks(work, 0);
        CloseThreadpoolWork(work);
    }
    assert_eq!(
        nd_spawn::count(),
        spawned,
        "a thread-pool worker was created with this executable's CreateThread"
    );
    assert_eq!(
        thread_ids(),
        before,
        "a thread-pool worker was counted as a linkcell thread"
    );
}

/// Kernel id of the calling thread. A thread this executable created with
/// `CreateThread` is one of the ids [`thread_ids`] returns while it is alive.
#[cfg(windows)]
fn current_thread_id() -> u64 {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThreadId() -> u32;
    }
    u64::from(unsafe { GetCurrentThreadId() })
}

/// A joined thread can remain in the kernel task list after `join` returns.
/// Wait until every id absent from `before` has left. An id other than
/// `joined` fails immediately. `joined` fails if it is still listed after
/// two seconds.
#[cfg(any(
    all(target_os = "linux", target_env = "gnu"),
    target_os = "macos",
    windows
))]
fn wait_until_joined_tids_leave(before: &std::collections::BTreeSet<u64>, joined: u64) {
    let bound = std::time::Duration::from_secs(2);
    let start = std::time::Instant::now();
    loop {
        let now = thread_ids();
        if now == *before {
            return;
        }
        let extra: Vec<u64> = now.difference(before).copied().collect();
        if extra.iter().any(|id| *id != joined) {
            panic!("a thread other than joined tid {joined} is in the id snapshot: {extra:?}");
        }
        if start.elapsed() >= bound {
            panic!("joined tid {joined} stayed in the task list for {bound:?}");
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

/// Under `lc_set_threads(1)` no ND call starts a thread. The suite shares
/// this process with tests that do start threads, so the check runs in a
/// child whose only work is these calls.
///
/// On glibc, macOS, and Windows each call is bracketed by a spawn counter
/// and by a snapshot of thread identifiers. The snapshot alone misses a
/// worker that is created and joined inside the call. The counter is
/// checked first by creating and joining one thread: that create must be
/// visible. The joined id must then leave the task list. A joined id can
/// remain listed for a moment after `join`; the probe waits for that id,
/// and a wait that expires is a failure.
///
/// On other Linux targets the counter is not installed. `dlsym(RTLD_NEXT)`
/// is null in a static musl binary, and defining `pthread_create` aborts
/// the process. Those targets keep the thread-id snapshot only.
#[cfg(any(target_os = "linux", target_os = "macos", windows))]
#[test]
fn lc_nd_respects_set_threads_one_without_growing_threads() {
    if std::env::var_os("LC_ND_ONE_THREAD_PROBE").is_none() {
        let exe = std::env::current_exe().unwrap();
        let out = std::process::Command::new(exe)
            .arg("lc_nd_respects_set_threads_one_without_growing_threads")
            .args(["--exact", "--test-threads", "1", "--show-output"])
            .env("LC_ND_ONE_THREAD_PROBE", "1")
            .output()
            .unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            out.status.success(),
            "one-thread probe status {:?}\n{text}",
            out.status
        );
        assert!(
            text.contains("LC_ND_ONE_THREAD_PROBE_OK"),
            "child did not run the probe body\n{text}"
        );
        assert!(
            text.contains("test result: ok. 1 passed;"),
            "child did not report exactly one passing test\n{text}"
        );
        assert!(
            text.contains("test lc_nd_respects_set_threads_one_without_growing_threads ... ok"),
            "child did not pass this test\n{text}"
        );
        return;
    }
    #[cfg(any(
        all(target_os = "linux", target_env = "gnu"),
        target_os = "macos",
        windows
    ))]
    {
        nd_spawn::arm().unwrap_or_else(|err| panic!("spawn counter: {err}"));
        let spawned = nd_spawn::count();
        let before_ids = thread_ids();
        // Windows lists only threads this executable created with
        // `CreateThread`. The caller was not one of those. Linux and macOS
        // list every live thread, and the caller's id has to be one of them:
        // that is the same space the joined id is compared in.
        #[cfg(not(windows))]
        {
            let self_id = current_thread_id();
            assert!(
                before_ids.contains(&self_id),
                "this thread's id {self_id} is absent from the live-thread snapshot {before_ids:?}"
            );
        }
        let joined = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let slot = std::sync::Arc::clone(&joined);
        std::thread::spawn(move || {
            slot.store(current_thread_id(), std::sync::atomic::Ordering::SeqCst);
        })
        .join()
        .unwrap();
        let joined = joined.load(std::sync::atomic::Ordering::SeqCst);
        assert_ne!(joined, 0, "the joined thread did not record its id");
        assert_eq!(
            nd_spawn::count(),
            spawned + 1,
            "spawn counter missed a thread that was created and joined"
        );
        wait_until_joined_tids_leave(&before_ids, joined);
        assert_eq!(
            thread_ids(),
            before_ids,
            "a thread created and joined stayed in the id snapshot"
        );
    }
    #[cfg(windows)]
    assert_threadpool_worker_is_not_linkcells();
    let data = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let q = [0.0, 0.0];
    unsafe { lc_set_threads(1) };
    let mut idx = std::ptr::null_mut();
    assert_starts_no_thread("lc_nd_build", || {
        idx = build_nd_index(&data, 4, 2, None);
    });
    for _ in 0..200 {
        assert_starts_no_thread("lc_nd_knn", || {
            let mut nn = [-1; 4];
            let rc =
                unsafe { lc_nd_knn(idx, q.as_ptr(), 4, nn.as_mut_ptr(), std::ptr::null_mut()) };
            assert_eq!(rc, 0);
        });
        assert_starts_no_thread("lc_nd_within", || {
            let mut count = 0usize;
            let rc = unsafe {
                lc_nd_within(
                    idx,
                    q.as_ptr(),
                    2.0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                    &mut count,
                )
            };
            assert_eq!(rc, 0);
            assert_eq!(count, 4);
        });
        assert_starts_no_thread("lc_nd_pairs_within", || {
            let mut count = 0usize;
            let rc = unsafe {
                lc_nd_pairs_within(
                    idx,
                    2.0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                    &mut count,
                )
            };
            assert_eq!(rc, 0);
            assert_eq!(count, 6);
        });
        assert_starts_no_thread("lc_nd_dbscan", || {
            let mut labels = [-1; 4];
            let mut core = [0; 4];
            let mut clusters = 0usize;
            let rc = unsafe {
                lc_nd_dbscan(
                    idx,
                    2.0,
                    2,
                    labels.as_mut_ptr(),
                    core.as_mut_ptr(),
                    &mut clusters,
                )
            };
            assert_eq!(rc, 0);
            assert_eq!(clusters, 1);
        });
    }
    assert_starts_no_thread("lc_nd_free", || unsafe { lc_nd_free(idx) });
    unsafe { lc_set_threads(0) };
    println!("LC_ND_ONE_THREAD_PROBE_OK");
}

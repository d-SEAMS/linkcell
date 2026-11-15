//! One parallel region per search.
//!
//! A search is a list of stages, each a number of chunks. Every worker
//! claims chunks of the open stage from one counter, and the worker that
//! finishes the last chunk opens the next stage, so a stage boundary waits
//! for the chunks in flight and for no worker that has not started. The
//! caller is one of the workers. The others are detached jobs on the pool:
//! one that starts after the last stage has closed finds no chunk and
//! returns without touching the caller's data, so the call never waits for
//! a worker the scheduler has not run yet.
//!
//! A worker with nothing to claim spins briefly, then yields, then sleeps
//! on a condvar until the stage it is waiting on closes. The wait does not
//! depend on `OMP_WAIT_POLICY` or on Rayon's idle loop: on a machine whose
//! cpus are all busy, a sleeping worker leaves its cpu to the worker that
//! still holds a chunk.

use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// One stage: `chunks` calls of `run(scratch, chunk)`, any order, any worker.
pub(crate) struct Stage<'a, S> {
    pub chunks: usize,
    pub run: &'a (dyn Fn(&mut S, usize) + Sync),
}

/// Starts one detached job on the pool that owns the search.
pub(crate) type Spawn<'a> = &'a dyn Fn(Box<dyn FnOnce() + Send + 'static>);

type Erased<S> = *const (dyn Fn(&mut S, usize) + Sync + 'static);

struct Shared<S> {
    run: Vec<Erased<S>>,
    chunks: Vec<usize>,
    next: Vec<AtomicUsize>,
    done: Vec<AtomicUsize>,
    phase: AtomicUsize,
    sleepers: AtomicUsize,
    failed: AtomicBool,
    failure: Mutex<Option<Box<dyn std::any::Any + Send>>>,
    lock: Mutex<()>,
    wake: Condvar,
}

// SAFETY: `run` points at the caller's `Sync` closures. A worker calls one
// only after it has claimed a chunk of an open stage, and the caller does
// not return before every claimed chunk is done (see `work`).
unsafe impl<S> Send for Shared<S> {}
unsafe impl<S> Sync for Shared<S> {}

/// Spins on the stage counter before yielding. About 20 microseconds of
/// `pause`, the length of a short stage tail on an idle machine.
const SPINS: u32 = 1 << 9;
/// Yields before sleeping.
const YIELDS: u32 = 16;

impl<S> Shared<S> {
    fn wait_past(&self, stage: usize) {
        for _ in 0..SPINS {
            if self.phase.load(Ordering::Acquire) > stage {
                return;
            }
            std::hint::spin_loop();
        }
        for _ in 0..YIELDS {
            if self.phase.load(Ordering::Acquire) > stage {
                return;
            }
            std::thread::yield_now();
        }
        let mut guard = self.lock.lock().unwrap_or_else(|err| err.into_inner());
        self.sleepers.fetch_add(1, Ordering::SeqCst);
        while self.phase.load(Ordering::SeqCst) <= stage {
            guard = self.wake.wait(guard).unwrap_or_else(|err| err.into_inner());
        }
        self.sleepers.fetch_sub(1, Ordering::SeqCst);
    }

    fn open(&self, stage: usize) {
        self.phase.store(stage, Ordering::SeqCst);
        if self.sleepers.load(Ordering::SeqCst) > 0 {
            drop(self.lock.lock().unwrap_or_else(|err| err.into_inner()));
            self.wake.notify_all();
        }
    }
}

/// Claim and run chunks until the last stage has closed.
fn work<S: Default>(shared: &Shared<S>) {
    let stages = shared.chunks.len();
    let mut scratch: Option<S> = None;
    loop {
        let p = shared.phase.load(Ordering::Acquire);
        if p >= stages {
            return;
        }
        let c = shared.next[p].fetch_add(1, Ordering::Relaxed);
        if c >= shared.chunks[p] {
            shared.wait_past(p);
            continue;
        }
        if !shared.failed.load(Ordering::Relaxed) {
            let s = scratch.get_or_insert_with(S::default);
            // SAFETY: chunk `c` of open stage `p` is claimed and not done,
            // so the caller is still inside `run` and the closure is live.
            let f = unsafe { &*shared.run[p] };
            if let Err(err) = catch_unwind(AssertUnwindSafe(|| f(s, c))) {
                shared.failed.store(true, Ordering::Relaxed);
                let mut slot = shared.failure.lock().unwrap_or_else(|e| e.into_inner());
                slot.get_or_insert(err);
            }
        }
        if shared.done[p].fetch_add(1, Ordering::AcqRel) + 1 == shared.chunks[p] {
            shared.open(p + 1);
        }
    }
}

/// Run `stages` in order on `width` workers, the caller one of them.
/// Every chunk of a stage finishes before any chunk of the next starts.
/// One worker, or no pool, runs everything on the caller.
pub(crate) fn run<S: Default + 'static>(
    width: usize,
    spawn: Option<Spawn<'_>>,
    stages: &[Stage<'_, S>],
) {
    let spawn = match spawn {
        Some(spawn) if width > 1 => spawn,
        _ => {
            let mut scratch = S::default();
            for stage in stages {
                for c in 0..stage.chunks {
                    (stage.run)(&mut scratch, c);
                }
            }
            return;
        }
    };
    // A stage with no chunks would never be closed by a worker.
    let chunks: Vec<usize> = stages.iter().map(|s| s.chunks.max(1)).collect();
    let empty = |_: &mut S, _: usize| {};
    let run = stages
        .iter()
        .map(|s| {
            let f: &(dyn Fn(&mut S, usize) + Sync) = if s.chunks == 0 { &empty } else { s.run };
            // SAFETY: lifetime erased; see `Shared`.
            unsafe {
                std::mem::transmute::<*const (dyn Fn(&mut S, usize) + Sync + '_), Erased<S>>(f)
            }
        })
        .collect();
    let shared = Arc::new(Shared {
        run,
        next: chunks.iter().map(|_| AtomicUsize::new(0)).collect(),
        done: chunks.iter().map(|_| AtomicUsize::new(0)).collect(),
        chunks,
        phase: AtomicUsize::new(0),
        sleepers: AtomicUsize::new(0),
        failed: AtomicBool::new(false),
        failure: Mutex::new(None),
        lock: Mutex::new(()),
        wake: Condvar::new(),
    });
    for _ in 1..width {
        let mine = Arc::clone(&shared);
        spawn(Box::new(move || work(&mine)));
    }
    work(&shared);
    let failure = shared
        .failure
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take();
    if let Some(err) = failure {
        resume_unwind(err);
    }
}

type ErasedSpawn = *const (dyn Fn(Box<dyn FnOnce() + Send + 'static>) + 'static);

thread_local! {
    static INSTALLED: std::cell::Cell<Option<(usize, ErasedSpawn)>> =
        const { std::cell::Cell::new(None) };
}

/// Searches `body` starts on this thread use `width` workers started by
/// `spawn`, until `body` returns.
#[cfg_attr(not(all(feature = "capi", feature = "parallel")), allow(dead_code))]
pub(crate) fn install<R>(width: usize, spawn: Spawn<'_>, body: impl FnOnce() -> R) -> R {
    struct Restore(Option<(usize, ErasedSpawn)>);
    impl Drop for Restore {
        fn drop(&mut self) {
            INSTALLED.with(|slot| slot.set(self.0));
        }
    }
    // SAFETY: lifetime erased; the slot is restored before `spawn` dies.
    let erased = unsafe {
        std::mem::transmute::<*const (dyn Fn(Box<dyn FnOnce() + Send + 'static>) + '_), ErasedSpawn>(
            spawn,
        )
    };
    let _restore = Restore(INSTALLED.with(|slot| slot.replace(Some((width, erased)))));
    body()
}

/// Whether a caller on this thread chose the workers with [`install`].
#[cfg(linkcell_openmp)]
pub(crate) fn installed() -> bool {
    INSTALLED.with(|slot| slot.get()).is_some()
}

/// Workers and pool for a search on this thread: the ones [`install`]
/// set, or else the current Rayon pool.
pub(crate) fn with_current<R>(body: impl FnOnce(usize, Option<Spawn<'_>>) -> R) -> R {
    if let Some((width, spawn)) = INSTALLED.with(|slot| slot.get()) {
        // SAFETY: `install` is still running on this thread.
        let spawn: Spawn<'_> = unsafe { &*spawn };
        return body(width, Some(spawn));
    }
    #[cfg(feature = "parallel")]
    {
        let spawn = |job: Box<dyn FnOnce() + Send + 'static>| rayon::spawn(job);
        body(rayon::current_num_threads(), Some(&spawn))
    }
    #[cfg(not(feature = "parallel"))]
    {
        body(1, None)
    }
}

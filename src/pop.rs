//! POP timing for a scaling run.
//!
//! Set `LINKCELL_POP` and the mesh build and the walk record useful
//! time on each rayon worker. Useful time is time inside the user
//! loops, not time spent waiting on other workers. The formulas are
//! the POP single-level hierarchy: load balance is the average useful
//! time over the maximum, communication efficiency is that maximum
//! over the wall time, and parallel efficiency is their product.

use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

const MAX_THREADS: usize = 128;

fn slots() -> &'static [AtomicU64] {
    static SLOTS: OnceLock<Box<[AtomicU64]>> = OnceLock::new();
    SLOTS.get_or_init(|| (0..MAX_THREADS).map(|_| AtomicU64::new(0)).collect())
}

thread_local! {
    static SLOT: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
    static IN_POOL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

static NEXT: AtomicUsize = AtomicUsize::new(0);
static NTHREADS: AtomicUsize = AtomicUsize::new(1);

pub(crate) fn enabled() -> bool {
    static FLAG: AtomicU8 = AtomicU8::new(0);
    match FLAG.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = std::env::var_os("LINKCELL_POP").is_some();
            FLAG.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

fn slot() -> usize {
    SLOT.with(|cell| {
        let cur = cell.get();
        if cur != usize::MAX {
            return cur;
        }
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        cell.set(id);
        id
    })
}

/// Worker slot for this thread. A caller outside the pool folds its
/// serial time into worker 0, which is the master in the POP model.
fn worker_slot() -> usize {
    let in_pool = IN_POOL.with(|cell| cell.get());
    if in_pool {
        slot()
    } else {
        0
    }
}

/// Give every worker a slot. Call once before the timed repetitions.
///
/// Does nothing unless `LINKCELL_POP` is set.
pub fn prepare() {
    if !enabled() {
        return;
    }
    #[cfg(feature = "parallel")]
    {
        rayon::broadcast(|_| {
            IN_POOL.with(|cell| cell.set(true));
            let _ = slot();
        });
        NTHREADS.store(rayon::current_num_threads(), Ordering::Relaxed);
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = slot();
        NTHREADS.store(1, Ordering::Relaxed);
    }
}

/// Zero the useful-time counters. Slot assignment stays.
///
/// Does nothing unless `LINKCELL_POP` is set.
pub fn reset() {
    if !enabled() {
        return;
    }
    for entry in slots() {
        entry.store(0, Ordering::Relaxed);
    }
}

fn add_slot(slot_id: usize, ns: u64) {
    if slot_id < slots().len() {
        slots()[slot_id].fetch_add(ns, Ordering::Relaxed);
    }
}

/// Useful nanoseconds on each worker, in slot order.
///
/// Empty when `LINKCELL_POP` is unset.
pub fn snapshot() -> Vec<u64> {
    if !enabled() {
        return Vec::new();
    }
    let n = NTHREADS.load(Ordering::Relaxed).clamp(1, MAX_THREADS);
    (0..n).map(|i| slots()[i].load(Ordering::Relaxed)).collect()
}

/// Times one rayon job, or one serial section, and adds it on drop.
pub(crate) struct JobTimer {
    start: Option<Instant>,
    slot: usize,
}

impl JobTimer {
    pub(crate) fn new() -> Self {
        if !enabled() {
            return Self {
                start: None,
                slot: 0,
            };
        }
        Self {
            start: Some(Instant::now()),
            slot: worker_slot(),
        }
    }
}

impl Drop for JobTimer {
    fn drop(&mut self) {
        if let Some(start) = self.start.take() {
            add_slot(self.slot, start.elapsed().as_nanos() as u64);
        }
    }
}

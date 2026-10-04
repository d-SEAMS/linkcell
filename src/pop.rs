//! POP3 timing for a scaling run.
//!
//! [POP3](https://pop-coe.eu/) is the multiplicative hierarchy. Useful
//! time is time inside the user loops, not time spent waiting on other
//! workers. Load balance is the average useful time over the maximum.
//! Communication efficiency is that maximum over the wall time.
//! Parallel efficiency is their product. Computation scaling and
//! global efficiency need a 1-thread reference, which
//! `scripts/pop-report.py` applies. Instruction, IPC, and frequency
//! scaling need a PMU, and this host has none.
//!
//! Set `LINKCELL_POP`, or call [`engage`], and the mesh build and the
//! walk record useful time on each rayon worker. The scale probes call
//! [`engage`] so a profile always uses this hierarchy.

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

static FLAG: AtomicU8 = AtomicU8::new(0);

pub(crate) fn enabled() -> bool {
    match FLAG.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = std::env::var_os("LINKCELL_POP").is_some();
            let _ = FLAG.compare_exchange(
                0,
                if on { 2 } else { 1 },
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
            FLAG.load(Ordering::Relaxed) == 2
        }
    }
}

/// Turn POP3 counters on for this process.
///
/// Scale probes call this before [`prepare`] so the run records useful
/// time even when `LINKCELL_POP` is unset. A later call wins over an
/// earlier read of the environment.
pub fn engage() {
    FLAG.store(2, Ordering::Relaxed);
}

/// POP3 factors for one run: load balance, communication efficiency,
/// parallel efficiency.
///
/// `useful` is one nanosecond total per worker. `wall_ns` is the
/// elapsed time of the same repetitions.
pub fn efficiencies(useful: &[u64], wall_ns: u64) -> (f64, f64, f64) {
    if useful.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let sum: u64 = useful.iter().copied().sum();
    let max = useful.iter().copied().max().unwrap_or(0);
    let avg = sum as f64 / useful.len() as f64;
    let lb = if max == 0 { 0.0 } else { avg / max as f64 };
    let ce = if wall_ns == 0 {
        0.0
    } else {
        max as f64 / wall_ns as f64
    };
    (lb, ce, lb * ce)
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
/// Does nothing until [`engage`] or `LINKCELL_POP`.
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
/// Does nothing until [`engage`] or `LINKCELL_POP`.
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
/// Empty until [`engage`] or `LINKCELL_POP`.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn efficiencies_match_pop3() {
        let (lb, ce, pe) = efficiencies(&[2, 4], 8);
        assert!((lb - 0.75).abs() < 1e-12);
        assert!((ce - 0.5).abs() < 1e-12);
        assert!((pe - 0.375).abs() < 1e-12);
    }
}

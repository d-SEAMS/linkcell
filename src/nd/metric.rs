//! Metric shim for N-dimensional neighbour search.
//!
//! This module is intentionally small and isolated: once minimage ships its
//! N-dimensional metric traits and bounds, this implementation should be
//! replaced by the upstream one.

use crate::Error;

/// Distance and pruning-bound metric for N-dimensional neighbour search.
pub trait Metric {
    /// Dimensionality.
    fn dims(&self) -> usize;
    /// Squared distance between two points.
    fn dist2(&self, a: &[f64], b: &[f64]) -> Result<f64, Error>;
    /// Lower bound (squared) from scalar coordinate `x` to interval `[lo, hi]`
    /// on one axis.
    fn axis_min_dist2(&self, x: f64, lo: f64, hi: f64, axis: usize) -> Result<f64, Error>;
    /// Lower bound (squared) from point `p` to an axis-aligned box.
    fn aabb_min_dist2(&self, p: &[f64], lo: &[f64], hi: &[f64]) -> Result<f64, Error> {
        if p.len() != self.dims() || lo.len() != self.dims() || hi.len() != self.dims() {
            return Err(Error::BadDimensions);
        }
        let mut s = 0.0;
        for i in 0..self.dims() {
            s += self.axis_min_dist2(p[i], lo[i], hi[i], i)?;
        }
        Ok(s)
    }
}

/// Temporary metric shim for mixed periodic/open Euclidean dimensions.
///
/// TODO(minimage-nd): replace this type with the upstream minimage N-D metric
/// implementation when released.
#[derive(Clone, Debug)]
pub struct PeriodicEuclidean {
    periods: Vec<Option<f64>>,
}

impl PeriodicEuclidean {
    /// `periods[k] = Some(L)` means axis `k` is periodic with period `L`;
    /// `None` means open axis.
    pub fn new(periods: Vec<Option<f64>>) -> Result<Self, Error> {
        for &p in &periods {
            if let Some(v) = p {
                if !v.is_finite() || v <= 0.0 {
                    return Err(Error::BadPeriod);
                }
            }
        }
        Ok(Self { periods })
    }

    /// Dense convention: `periods[k] > 0` means periodic, otherwise open.
    pub fn from_dense(periods: &[f64]) -> Result<Self, Error> {
        let mut out = Vec::with_capacity(periods.len());
        for &p in periods {
            if p > 0.0 {
                if !p.is_finite() {
                    return Err(Error::BadPeriod);
                }
                out.push(Some(p));
            } else {
                out.push(None);
            }
        }
        Self::new(out)
    }

    fn wrap01(x: f64) -> f64 {
        // Match linkcell/minimage floor_fast convention (truncation semantics).
        let mut y = x - x.trunc();
        if y < 0.0 {
            y += 1.0;
        }
        y
    }

    fn wrap_delta(delta: f64, period: f64) -> f64 {
        let mut y = delta - period * (delta / period).trunc();
        if y < 0.0 {
            y += period;
        }
        y
    }

    fn periodic_axis_min_abs(&self, x: f64, lo: f64, hi: f64, l: f64) -> f64 {
        let width = hi - lo;
        if width >= l {
            return 0.0;
        }
        let base = Self::wrap01(lo / l) * l;
        let end = base + width;
        let xw = Self::wrap01(x / l) * l;
        let mut best = f64::INFINITY;
        for shift in [-l, 0.0, l] {
            let xs = xw + shift;
            let d = if xs < base {
                base - xs
            } else if xs > end {
                xs - end
            } else {
                0.0
            };
            if d < best {
                best = d;
            }
        }
        best
    }
}

impl Metric for PeriodicEuclidean {
    fn dims(&self) -> usize {
        self.periods.len()
    }

    fn dist2(&self, a: &[f64], b: &[f64]) -> Result<f64, Error> {
        if a.len() != self.dims() || b.len() != self.dims() {
            return Err(Error::BadDimensions);
        }
        if !a.iter().all(|x| x.is_finite()) || !b.iter().all(|x| x.is_finite()) {
            return Err(Error::NonFinite);
        }
        let mut s = 0.0;
        for i in 0..self.dims() {
            let d = if let Some(l) = self.periods[i] {
                let w = Self::wrap_delta(a[i] - b[i], l);
                w.min(l - w)
            } else {
                (a[i] - b[i]).abs()
            };
            s += d * d;
        }
        Ok(s)
    }

    fn axis_min_dist2(&self, x: f64, lo: f64, hi: f64, axis: usize) -> Result<f64, Error> {
        if axis >= self.dims() || !x.is_finite() || !lo.is_finite() || !hi.is_finite() || lo > hi {
            return Err(Error::BadDimensions);
        }
        let d = if let Some(l) = self.periods[axis] {
            self.periodic_axis_min_abs(x, lo, hi, l)
        } else if x < lo {
            lo - x
        } else if x > hi {
            x - hi
        } else {
            0.0
        };
        Ok(d * d)
    }
}

#[cfg(test)]
mod tests {
    use super::{Metric, PeriodicEuclidean};

    fn brute_axis_min_dist2(x: f64, lo: f64, hi: f64, l: f64) -> f64 {
        // Independent brute-force oracle on the circle.
        if hi - lo >= l {
            return 0.0;
        }
        let mut best = f64::INFINITY;
        let steps = 20_000usize;
        for t in 0..=steps {
            let y = lo + (hi - lo) * (t as f64 / steps as f64);
            let mut d = x - y;
            d = d - l * (d / l).trunc();
            if d < 0.0 {
                d += l;
            }
            d = d.min(l - d);
            best = best.min(d * d);
        }
        best
    }

    #[test]
    fn axis_bound_is_sound_for_wide_periodic_intervals() {
        let metric = PeriodicEuclidean::new(vec![Some(7.0)]).unwrap();
        for i in 0..512 {
            let x = -40.0 + 80.0 * (i as f64 / 511.0);
            let lo = -30.0 + 0.13 * i as f64;
            let width = 3.6 + (i % 17) as f64 * 0.22; // > L/2 often, sometimes > L
            let hi = lo + width;
            let got = metric.axis_min_dist2(x, lo, hi, 0).unwrap();
            let want = brute_axis_min_dist2(x, lo, hi, 7.0);
            assert!(got <= want + 1.0e-9);
            if hi - lo >= 7.0 {
                assert_eq!(got, 0.0);
            }
        }
    }

    #[test]
    fn aabb_bound_is_sound_for_periodic_boxes() {
        let metric = PeriodicEuclidean::new(vec![Some(5.0), Some(9.0), None]).unwrap();
        for i in 0..256 {
            let p = [
                -12.0 + i as f64 * 0.17,
                7.0 - i as f64 * 0.09,
                -2.0 + i as f64 * 0.03,
            ];
            let lo = [-8.0 + i as f64 * 0.01, -5.0, -1.0];
            let hi = [lo[0] + 3.2 + (i % 11) as f64 * 0.35, 8.0, 2.0];
            let lb = metric.aabb_min_dist2(&p, &lo, &hi).unwrap();
            // Any point inside the box must be no closer than the lower bound.
            let mut worst_violation = 0.0f64;
            for a in 0..20 {
                for b in 0..20 {
                    for c in 0..20 {
                        let q = [
                            lo[0] + (hi[0] - lo[0]) * a as f64 / 19.0,
                            lo[1] + (hi[1] - lo[1]) * b as f64 / 19.0,
                            lo[2] + (hi[2] - lo[2]) * c as f64 / 19.0,
                        ];
                        let d2 = metric.dist2(&p, &q).unwrap();
                        worst_violation = worst_violation.max(lb - d2);
                    }
                }
            }
            assert!(worst_violation <= 1.0e-9);
        }
    }
}

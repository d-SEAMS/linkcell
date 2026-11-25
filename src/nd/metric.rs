//! Metric shim for N-dimensional neighbour search.
//!
//! This module is intentionally small and isolated: once minimage ships its
//! N-dimensional metric traits and bounds, this implementation should be
//! replaced by the upstream one.

use crate::Error;

/// Draft-shaped metric trait for a mechanical future swap to minimage.
pub trait Metric<T = f64> {
    /// Dimensionality.
    fn dims(&self) -> usize;
    /// Squared distance between two points.
    fn dist2(&self, a: &[T], b: &[T]) -> Result<T, Error>;
    /// Distance between two points.
    fn distance(&self, a: &[T], b: &[T]) -> Result<T, Error>;
    /// Lower bound from a scalar coordinate to one interval.
    fn axis_min_dist2(&self, x: T, lo: T, hi: T, axis: usize) -> Result<T, Error>;
    /// Lower bound from point to axis-aligned box.
    fn aabb_min_dist2(&self, p: &[T], lo: &[T], hi: &[T]) -> Result<T, Error>;
    /// Alias of `aabb_min_dist2` in the draft API.
    fn box_dist2(&self, p: &[T], lo: &[T], hi: &[T]) -> Result<T, Error>;
    /// Distance from point to axis-aligned box.
    fn box_distance(&self, p: &[T], lo: &[T], hi: &[T]) -> Result<T, Error>;
    /// Batched squared distances, row-major points.
    fn dist2_batch(&self, a: &[T], b: &[T], out: &mut [T]) -> Result<(), Error>;
    /// Batched distances, row-major points.
    fn distance_batch(&self, a: &[T], b: &[T], out: &mut [T]) -> Result<(), Error>;
}

/// Open Euclidean metric in `dims` dimensions.
#[derive(Clone, Copy, Debug)]
pub struct Euclidean {
    dims: usize,
}

impl Euclidean {
    /// Build an open Euclidean metric of dimensionality `dims`.
    pub fn new(dims: usize) -> Result<Self, Error> {
        if dims == 0 {
            return Err(Error::BadDimensions);
        }
        Ok(Self { dims })
    }
}

/// Temporary periodic-space metric shim (`None` axis = open).
///
/// TODO(minimage-nd): replace this type with the upstream minimage N-D metric
/// implementation when released.
#[derive(Clone, Debug)]
pub struct PeriodicSpace {
    periods: Vec<Option<f64>>,
}

/// Backward-compatible alias kept for r1 API users.
pub type PeriodicEuclidean = PeriodicSpace;

impl PeriodicSpace {
    /// Build a periodic/open mixed space metric from optional per-axis periods.
    pub fn new(periods: Vec<Option<f64>>) -> Result<Self, Error> {
        if periods.is_empty() {
            return Err(Error::BadDimensions);
        }
        for &p in &periods {
            if let Some(v) = p {
                if !v.is_finite() || v <= 0.0 {
                    return Err(Error::BadPeriod);
                }
            }
        }
        Ok(Self { periods })
    }

    /// Dense convention: `periods[k] > 0` periodic, `0` open, `< 0` invalid.
    pub fn from_dense(periods: &[f64]) -> Result<Self, Error> {
        if periods.is_empty() {
            return Err(Error::BadDimensions);
        }
        let mut out = Vec::with_capacity(periods.len());
        for &p in periods {
            if p.is_nan() || p < 0.0 {
                return Err(Error::BadPeriod);
            }
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

    /// Wrap one coordinate to `[0, L)`.
    pub fn wrap_point_axis(&self, x: f64, axis: usize) -> Result<f64, Error> {
        if axis >= self.periods.len() || !x.is_finite() {
            return Err(Error::BadDimensions);
        }
        Ok(match self.periods[axis] {
            Some(l) => Self::wrap01(x / l) * l,
            None => x,
        })
    }

    /// Wrap one delta to `[-L/2, L/2)` with `+L/2 -> -L/2`.
    pub fn wrap_delta_axis(&self, delta: f64, axis: usize) -> Result<f64, Error> {
        if axis >= self.periods.len() || !delta.is_finite() {
            return Err(Error::BadDimensions);
        }
        Ok(match self.periods[axis] {
            Some(l) => {
                let mut y = delta - l * (delta / l).trunc();
                if y < 0.0 {
                    y += l;
                }
                if y >= 0.5 * l {
                    y -= l;
                }
                y
            }
            None => delta,
        })
    }

    /// Wrap a row-major batch of points to each periodic axis.
    pub fn wrap_points_batch(
        &self,
        points: &[f64],
        out: &mut [f64],
        n_points: usize,
    ) -> Result<(), Error> {
        let d = self.periods.len();
        if points.len() != n_points * d || out.len() != points.len() {
            return Err(Error::BadDimensions);
        }
        for i in 0..n_points {
            for k in 0..d {
                out[i * d + k] = self.wrap_point_axis(points[i * d + k], k)?;
            }
        }
        Ok(())
    }

    /// Wrap a row-major batch of deltas to each periodic axis.
    pub fn wrap_deltas_batch(
        &self,
        deltas: &[f64],
        out: &mut [f64],
        n_points: usize,
    ) -> Result<(), Error> {
        let d = self.periods.len();
        if deltas.len() != n_points * d || out.len() != deltas.len() {
            return Err(Error::BadDimensions);
        }
        for i in 0..n_points {
            for k in 0..d {
                out[i * d + k] = self.wrap_delta_axis(deltas[i * d + k], k)?;
            }
        }
        Ok(())
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

    fn check_point_shape(&self, p: &[f64]) -> Result<(), Error> {
        if p.len() != self.periods.len() || !p.iter().all(|v| v.is_finite()) {
            return Err(Error::BadDimensions);
        }
        Ok(())
    }
}

impl Metric<f64> for Euclidean {
    fn dims(&self) -> usize {
        self.dims
    }

    fn dist2(&self, a: &[f64], b: &[f64]) -> Result<f64, Error> {
        if a.len() != self.dims || b.len() != self.dims {
            return Err(Error::BadDimensions);
        }
        if !a.iter().all(|x| x.is_finite()) || !b.iter().all(|x| x.is_finite()) {
            return Err(Error::NonFinite);
        }
        Ok(a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum())
    }

    fn distance(&self, a: &[f64], b: &[f64]) -> Result<f64, Error> {
        Ok(self.dist2(a, b)?.sqrt())
    }

    fn axis_min_dist2(&self, x: f64, lo: f64, hi: f64, axis: usize) -> Result<f64, Error> {
        if axis >= self.dims || !x.is_finite() || !lo.is_finite() || !hi.is_finite() || lo > hi {
            return Err(Error::BadDimensions);
        }
        let d = if x < lo {
            lo - x
        } else if x > hi {
            x - hi
        } else {
            0.0
        };
        Ok(d * d)
    }

    fn aabb_min_dist2(&self, p: &[f64], lo: &[f64], hi: &[f64]) -> Result<f64, Error> {
        if p.len() != self.dims || lo.len() != self.dims || hi.len() != self.dims {
            return Err(Error::BadDimensions);
        }
        let mut s = 0.0;
        for i in 0..self.dims {
            s += self.axis_min_dist2(p[i], lo[i], hi[i], i)?;
        }
        Ok(s)
    }

    fn box_dist2(&self, p: &[f64], lo: &[f64], hi: &[f64]) -> Result<f64, Error> {
        self.aabb_min_dist2(p, lo, hi)
    }

    fn box_distance(&self, p: &[f64], lo: &[f64], hi: &[f64]) -> Result<f64, Error> {
        Ok(self.box_dist2(p, lo, hi)?.sqrt())
    }

    fn dist2_batch(&self, a: &[f64], b: &[f64], out: &mut [f64]) -> Result<(), Error> {
        if a.len() != b.len() || a.len() != out.len() * self.dims {
            return Err(Error::BadDimensions);
        }
        for (i, slot) in out.iter_mut().enumerate() {
            let off = i * self.dims;
            *slot = self.dist2(&a[off..off + self.dims], &b[off..off + self.dims])?;
        }
        Ok(())
    }

    fn distance_batch(&self, a: &[f64], b: &[f64], out: &mut [f64]) -> Result<(), Error> {
        self.dist2_batch(a, b, out)?;
        for v in out {
            *v = v.sqrt();
        }
        Ok(())
    }
}

impl Metric<f64> for PeriodicSpace {
    fn dims(&self) -> usize {
        self.periods.len()
    }

    fn dist2(&self, a: &[f64], b: &[f64]) -> Result<f64, Error> {
        self.check_point_shape(a)?;
        self.check_point_shape(b)?;
        let mut s = 0.0;
        for i in 0..self.periods.len() {
            let d = self.wrap_delta_axis(a[i] - b[i], i)?;
            s += d * d;
        }
        Ok(s)
    }

    fn distance(&self, a: &[f64], b: &[f64]) -> Result<f64, Error> {
        Ok(self.dist2(a, b)?.sqrt())
    }

    fn axis_min_dist2(&self, x: f64, lo: f64, hi: f64, axis: usize) -> Result<f64, Error> {
        if axis >= self.periods.len()
            || !x.is_finite()
            || !lo.is_finite()
            || !hi.is_finite()
            || lo > hi
        {
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

    fn aabb_min_dist2(&self, p: &[f64], lo: &[f64], hi: &[f64]) -> Result<f64, Error> {
        if p.len() != self.periods.len()
            || lo.len() != self.periods.len()
            || hi.len() != self.periods.len()
        {
            return Err(Error::BadDimensions);
        }
        let mut s = 0.0;
        for i in 0..self.periods.len() {
            s += self.axis_min_dist2(p[i], lo[i], hi[i], i)?;
        }
        Ok(s)
    }

    fn box_dist2(&self, p: &[f64], lo: &[f64], hi: &[f64]) -> Result<f64, Error> {
        self.aabb_min_dist2(p, lo, hi)
    }

    fn box_distance(&self, p: &[f64], lo: &[f64], hi: &[f64]) -> Result<f64, Error> {
        Ok(self.box_dist2(p, lo, hi)?.sqrt())
    }

    fn dist2_batch(&self, a: &[f64], b: &[f64], out: &mut [f64]) -> Result<(), Error> {
        if a.len() != b.len() || a.len() != out.len() * self.periods.len() {
            return Err(Error::BadDimensions);
        }
        let d = self.periods.len();
        for (i, slot) in out.iter_mut().enumerate() {
            let off = i * d;
            *slot = self.dist2(&a[off..off + d], &b[off..off + d])?;
        }
        Ok(())
    }

    fn distance_batch(&self, a: &[f64], b: &[f64], out: &mut [f64]) -> Result<(), Error> {
        self.dist2_batch(a, b, out)?;
        for v in out {
            *v = v.sqrt();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Metric, PeriodicSpace};

    fn brute_axis_min_dist2(x: f64, lo: f64, hi: f64, l: f64) -> f64 {
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
            if d >= 0.5 * l {
                d -= l;
            }
            best = best.min(d * d);
        }
        best
    }

    #[test]
    fn delta_tie_maps_plus_half_to_minus_half() {
        let metric = PeriodicSpace::new(vec![Some(10.0)]).unwrap();
        let got = metric.wrap_delta_axis(5.0, 0).unwrap();
        assert_eq!(got, -5.0);
    }

    #[test]
    fn axis_bound_is_sound_for_wide_periodic_intervals() {
        let metric = PeriodicSpace::new(vec![Some(7.0)]).unwrap();
        for i in 0..512 {
            let x = -40.0 + 80.0 * (i as f64 / 511.0);
            let lo = -30.0 + 0.13 * i as f64;
            let width = 3.6 + (i % 17) as f64 * 0.22;
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
        let metric = PeriodicSpace::new(vec![Some(5.0), Some(9.0), None]).unwrap();
        for i in 0..256 {
            let p = [
                -12.0 + i as f64 * 0.17,
                7.0 - i as f64 * 0.09,
                -2.0 + i as f64 * 0.03,
            ];
            let lo = [-8.0 + i as f64 * 0.01, -5.0, -1.0];
            let hi = [lo[0] + 3.2 + (i % 11) as f64 * 0.35, 8.0, 2.0];
            let lb = metric.aabb_min_dist2(&p, &lo, &hi).unwrap();
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

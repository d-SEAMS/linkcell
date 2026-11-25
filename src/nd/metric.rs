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

    /// Representative of `src` stored by [`crate::VpTreeIndex`].
    ///
    /// The default copies `src`. A periodic metric folds each periodic axis
    /// into `[0, L)` so every later distance is taken on that representative.
    fn write_canonical(&self, src: &[f64], dst: &mut [f64]) -> Result<(), Error> {
        if src.len() != self.dims() || dst.len() != src.len() {
            return Err(Error::BadDimensions);
        }
        if !src.iter().copied().all(f64::is_finite) {
            return Err(Error::NonFinite);
        }
        dst.copy_from_slice(src);
        Ok(())
    }

    /// Absolute slack added to every vantage-point triangle test.
    ///
    /// Zero when a relative Higham factor covers the error of
    /// [`Metric::distance`]. Periodic metrics return the bound proved for
    /// the reduction into `[0, L)`.
    fn vp_absolute_slack(&self) -> f64 {
        0.0
    }
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
        if y >= 1.0 {
            // Tiny negative inputs can round to exactly one ulp below 1 then to 1.0.
            y = 0.0;
        }
        y
    }

    /// Spacing to the next binary64 value of a positive finite `x`.
    fn ulp_at(x: f64) -> f64 {
        if !x.is_finite() || x == 0.0 {
            return f64::from_bits(1);
        }
        let next = if x > 0.0 {
            f64::from_bits(x.to_bits().saturating_add(1))
        } else {
            f64::from_bits(x.to_bits().saturating_sub(1))
        };
        let gap = (next - x).abs();
        if gap.is_finite() && gap > 0.0 {
            gap
        } else {
            f64::INFINITY
        }
    }

    /// Reduce one finite coordinate into `[0, L)`.
    ///
    /// Values already in `[0, L)` are returned unchanged, so a second call is
    /// the identity. Otherwise `y = x - trunc(x/L)*L` is one fused rounding,
    /// then `y` is shifted by `L` until it lies in range. `trunc(x/L)` is off
    /// by at most one when `|x| < 2^52 * L`; the loop absorbs a few more.
    /// When `x/L` is not finite, `x` has no unique residue and the
    /// representative is `0`.
    fn canon(x: f64, l: f64) -> Result<f64, Error> {
        if x >= 0.0 && x < l {
            return Ok(x);
        }
        let quot = x / l;
        if !quot.is_finite() {
            return Ok(0.0);
        }
        let q = quot.trunc();
        if !q.is_finite() {
            return Ok(0.0);
        }
        let mut y = (-q).mul_add(l, x);
        if !y.is_finite() {
            return Err(Error::Overflow);
        }
        for _ in 0..16 {
            if y >= 0.0 && y < l {
                return Ok(y);
            }
            if y >= l {
                let next = y - l;
                if next == y || !next.is_finite() {
                    return Ok(0.0);
                }
                y = next;
            } else {
                let next = y + l;
                if next == y || !next.is_finite() {
                    return Ok(0.0);
                }
                y = next;
            }
        }
        if y >= 0.0 && y < l {
            Ok(y)
        } else {
            Ok(0.0)
        }
    }

    /// Circular distance of two representatives in `[0, L)`.
    ///
    /// For `0 <= a < L` and `0 <= b < L`, `|fl(a-b)|` is strictly below `L`
    /// (the largest difference of such values is `nextDown(L)`, and it does
    /// not round up to `L`). The real circular distance is 1-Lipschitz in that
    /// difference, the subtraction errs by at most `u*L`, and `fl(L - w)`
    /// adds another `u*L`. The computed result therefore lies within
    /// `2*u*L <= 4*ulp(L)` of the real circular distance.
    fn circular(a: f64, b: f64, l: f64) -> Result<f64, Error> {
        let mut w = (a - b).abs();
        if !w.is_finite() {
            return Err(Error::Overflow);
        }
        if w >= l {
            w = 0.0;
        }
        let alt = l - w;
        if !alt.is_finite() {
            return Err(Error::Overflow);
        }
        Ok(if alt < w { alt } else { w })
    }

    /// `4 * ulp(L)` per periodic axis, as a Euclidean norm, times 8.
    ///
    /// Three distances enter each triangle test, so the defect is at most
    /// `3 * ||η||`. Eight leaves room for rounding the slack itself.
    fn proven_slack(periods: &[Option<f64>]) -> f64 {
        let mut sum = 0.0;
        for period in periods {
            let Some(l) = *period else {
                continue;
            };
            let eta = 4.0 * Self::ulp_at(l);
            let term = eta * eta;
            if !term.is_finite() {
                return f64::INFINITY;
            }
            sum += term;
            if !sum.is_finite() {
                return f64::INFINITY;
            }
        }
        let slack = 8.0 * sum.sqrt();
        if slack.is_finite() && slack >= 0.0 {
            slack
        } else {
            f64::INFINITY
        }
    }

    /// Wrap one coordinate to `[0, L)`.
    pub fn wrap_point_axis(&self, x: f64, axis: usize) -> Result<f64, Error> {
        if axis >= self.periods.len() || !x.is_finite() {
            return Err(Error::BadDimensions);
        }
        Ok(match self.periods[axis] {
            Some(l) => Self::canon(x, l)?,
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
                let y = Self::canon(delta, l)?;
                let half = 0.5 * l;
                if !half.is_finite() {
                    return Err(Error::Overflow);
                }
                if y >= half {
                    let z = y - l;
                    if !z.is_finite() {
                        return Err(Error::Overflow);
                    }
                    z
                } else {
                    y
                }
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
        if p.len() != self.periods.len() {
            return Err(Error::BadDimensions);
        }
        if !p.iter().all(|v| v.is_finite()) {
            return Err(Error::NonFinite);
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
        let mut s = 0.0;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = *x - *y;
            if !d.is_finite() {
                return Err(Error::Overflow);
            }
            let term = d * d;
            if !term.is_finite() {
                return Err(Error::Overflow);
            }
            s += term;
            if !s.is_finite() {
                return Err(Error::Overflow);
            }
        }
        Ok(s)
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
            let d = if let Some(l) = self.periods[i] {
                // Reduce each endpoint once. The fold of a raw difference
                // spanning many periods errs by about u*|a-b| and is not a
                // metric at that scale. On [0, L) the circular distance is.
                let aw = Self::canon(a[i], l)?;
                let bw = Self::canon(b[i], l)?;
                Self::circular(aw, bw, l)?
            } else {
                let w = (a[i] - b[i]).abs();
                if !w.is_finite() {
                    return Err(Error::Overflow);
                }
                w
            };
            let term = d * d;
            if !term.is_finite() {
                return Err(Error::Overflow);
            }
            s += term;
            if !s.is_finite() {
                return Err(Error::Overflow);
            }
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

    fn write_canonical(&self, src: &[f64], dst: &mut [f64]) -> Result<(), Error> {
        self.check_point_shape(src)?;
        if dst.len() != src.len() {
            return Err(Error::BadDimensions);
        }
        for i in 0..src.len() {
            dst[i] = match self.periods[i] {
                Some(l) => Self::canon(src[i], l)?,
                None => src[i],
            };
        }
        Ok(())
    }

    fn vp_absolute_slack(&self) -> f64 {
        Self::proven_slack(&self.periods)
    }
}

#[cfg(test)]
mod tests {
    use super::{Euclidean, Metric, PeriodicSpace};
    use crate::Error;

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
    fn point_wrap_stays_in_half_open_interval_for_tiny_negatives() {
        let metric = PeriodicSpace::new(vec![Some(3.0)]).unwrap();
        for &x in &[-1.0e-20, -1.0e-300] {
            let wrapped = metric.wrap_point_axis(x, 0).unwrap();
            assert!(wrapped >= 0.0);
            assert!(wrapped < 3.0);
        }
    }

    #[test]
    fn periodic_dist2_is_symmetric_and_self_zero() {
        let metric = PeriodicSpace::new(vec![Some(0.5), None, Some(3.0)]).unwrap();
        let a = [0.0, -2.0, 4.5];
        let b = [0.1, -1.0, -7.5];
        assert_eq!(metric.dist2(&a, &a).unwrap(), 0.0);
        assert_eq!(metric.dist2(&b, &b).unwrap(), 0.0);
        assert_eq!(metric.dist2(&a, &b).unwrap(), metric.dist2(&b, &a).unwrap());
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

    fn periodic_ref_axis(a: i64, b: i64, period: Option<i64>) -> i64 {
        if let Some(l) = period {
            let mut d = (a - b).abs() % l;
            if d > l - d {
                d = l - d;
            }
            d
        } else {
            (a - b).abs()
        }
    }

    #[test]
    fn metric_dist2_matches_independent_integer_reference() {
        let periods = vec![Some(11.0), None, Some(7.0), None];
        let metric = PeriodicSpace::new(periods.clone()).unwrap();
        for i in -10i64..=10 {
            for j in -10i64..=10 {
                let a_i = [i, 2 * i, -3 * i, i - 4];
                let b_i = [j, 2 * j, -3 * j, j - 4];
                let a = [a_i[0] as f64, a_i[1] as f64, a_i[2] as f64, a_i[3] as f64];
                let b = [b_i[0] as f64, b_i[1] as f64, b_i[2] as f64, b_i[3] as f64];
                let got = metric.dist2(&a, &b).unwrap();
                let mut ref_i2 = 0i128;
                for axis in 0..4 {
                    let p = periods[axis].map(|v| v as i64);
                    let d = periodic_ref_axis(a_i[axis], b_i[axis], p) as i128;
                    ref_i2 += d * d;
                }
                assert_eq!(got, ref_i2 as f64);
            }
        }
    }

    #[test]
    fn finite_overflow_in_dist2_is_error() {
        // Half a huge period is finite, and its square is not.
        let periodic = PeriodicSpace::new(vec![Some(1.0e200)]).unwrap();
        assert_eq!(periodic.dist2(&[0.0], &[6.0e199]), Err(Error::Overflow));

        // The raw difference overflows binary64. Both coordinates are the
        // same point on the circle, so the reduced distance is zero.
        let images = PeriodicSpace::new(vec![Some(1.0e308)]).unwrap();
        assert_eq!(images.dist2(&[1.0e308], &[-1.0e308]).unwrap(), 0.0);

        let euclid = Euclidean::new(1).unwrap();
        assert_eq!(euclid.dist2(&[1.0e308], &[-1.0e308]), Err(Error::Overflow));
    }

    #[test]
    fn canon_is_idempotent_and_inside_the_period() {
        let l = 2.0 * std::f64::consts::PI;
        let metric = PeriodicSpace::new(vec![Some(l)]).unwrap();
        let samples = [
            0.0,
            -0.0,
            l,
            -l,
            0.5 * l,
            -1.0e-20,
            -1.0e-300,
            1.0e-300,
            20.0 * l,
            -20.0 * l,
            1.0e20,
            -1.0e20,
            12_345.678_9,
            -5.675156211725825,
        ];
        for x in samples {
            let y = metric.wrap_point_axis(x, 0).unwrap();
            assert!(y >= 0.0 && y < l, "{x} -> {y}");
            assert_eq!(metric.wrap_point_axis(y, 0).unwrap(), y);
            assert_eq!(metric.dist2(&[x], &[x]).unwrap(), 0.0);
        }
    }
}

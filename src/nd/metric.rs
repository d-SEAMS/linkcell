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
    /// The relative Higham factor covers normal-range rounding. This term is
    /// the extra absolute defect: squared distances that underflow, and the
    /// periodic residue error. The default is `0`. [`Euclidean`] and
    /// [`PeriodicSpace`] replace it with a positive floor for every non-zero
    /// dimension.
    fn vp_absolute_slack(&self) -> f64 {
        0.0
    }
}

/// `2^-537`. Its square is the subnormal unit in the last place, `2^-1074`.
///
/// Every positive binary64 whose square underflows is strictly below this
/// value, so a computed `sqrt(fl(c*c))` differs from `c` by less than
/// `2^-537` when the square is subnormal.
fn square_underflow_unit() -> f64 {
    f64::from_bits(486_u64 << 52)
}

fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    if x > 0.0 {
        f64::from_bits(x.to_bits() + 1)
    } else {
        f64::from_bits(x.to_bits() - 1)
    }
}

fn next_down(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        return -f64::from_bits(1);
    }
    if x > 0.0 {
        f64::from_bits(x.to_bits() - 1)
    } else {
        f64::from_bits(x.to_bits() + 1)
    }
}

/// Spacing to the next binary64 value of a finite `x`.
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

/// Upper bound on `a*b` for non-negative finite factors.
fn mul_up(a: f64, b: f64) -> f64 {
    if !a.is_finite() || !b.is_finite() || a < 0.0 || b < 0.0 {
        return f64::INFINITY;
    }
    if a == 0.0 || b == 0.0 {
        return 0.0;
    }
    let p = a * b;
    if !p.is_finite() {
        return f64::INFINITY;
    }
    let resid = a.mul_add(b, -p);
    if resid > 0.0 {
        let up = next_up(p);
        if up.is_finite() {
            up
        } else {
            f64::INFINITY
        }
    } else {
        p
    }
}

/// Upper bound on `a+b` for non-negative finite terms.
fn add_up(a: f64, b: f64) -> f64 {
    if !a.is_finite() || !b.is_finite() || a < 0.0 || b < 0.0 {
        return f64::INFINITY;
    }
    let s = a + b;
    if !s.is_finite() {
        return f64::INFINITY;
    }
    let up = next_up(s);
    if up.is_finite() {
        up
    } else {
        f64::INFINITY
    }
}

/// Lower bound on `a-b` when `a >= b`. One ulp below the rounded difference.
fn diff_down(a: f64, b: f64) -> f64 {
    let d = a - b;
    if !(d.is_finite() && d > 0.0) {
        return 0.0;
    }
    let err = ulp_at(d);
    if d > err {
        d - err
    } else {
        0.0
    }
}

/// Lower bound on `d*d`.
fn square_down(d: f64) -> f64 {
    if !(d.is_finite() && d > 0.0) {
        return 0.0;
    }
    let s = d * d;
    if !s.is_finite() || s <= 0.0 {
        return 0.0;
    }
    let resid = d.mul_add(d, -s);
    if resid < 0.0 {
        let down = next_down(s);
        if down > 0.0 {
            down
        } else {
            0.0
        }
    } else {
        s
    }
}

/// Lower bound on `a+b` for non-negative terms.
fn add_down(a: f64, b: f64) -> f64 {
    if a < 0.0 || b < 0.0 || !a.is_finite() || !b.is_finite() {
        return 0.0;
    }
    let s = a + b;
    if !s.is_finite() || s <= 0.0 {
        return 0.0;
    }
    let err = ulp_at(s);
    if s > err {
        s - err
    } else {
        0.0
    }
}

/// Absolute prune floor for squared-distance underflow, in distance units.
///
/// Each axis can hide a gap of `2^-537` inside a square that underflows.
/// The Euclidean norm of that vector is at most `sqrt(dims)` times the
/// unit. Three distances enter a triangle test, and eight leaves room for
/// rounding this floor. `sqrt(dims)` is rounded upward.
fn square_underflow_floor(dims: usize) -> f64 {
    if dims == 0 {
        return 0.0;
    }
    let scale = next_up((dims as f64).sqrt());
    mul_up(mul_up(8.0, scale), square_underflow_unit())
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

    /// Reduce one finite coordinate into `[0, L)`.
    ///
    /// `rem_euclid` is the IEEE remainder. It is the exact mathematical
    /// residue whenever that residue is a binary64 number, and it lands in
    /// `[0, L)`. A residue closer to `L` than to any smaller binary64 value
    /// is returned as `L`; on the circle that seam is the same point as `0`,
    /// so the representative stored here is `0`. A value already in `[0, L)`
    /// is unchanged, and `-0` becomes `+0`.
    fn canon(x: f64, l: f64) -> Result<f64, Error> {
        let y = x.rem_euclid(l);
        if !y.is_finite() {
            return Err(Error::Overflow);
        }
        if y > 0.0 && y < l {
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
    /// adds another `u*L`. The seam representative `0` stands for a true
    /// residue within half an ulp of `L`, which adds at most one ulp. The
    /// computed result therefore lies within `4*ulp(L)` of the real circular
    /// distance of the original coordinates.
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

    /// Absolute prune slack: periodic residue error plus square underflow.
    ///
    /// `η_i = 4 * ulp(L_i)` and `||η||_2 <= sqrt(n) * max η_i`. The product
    /// of squares is never formed, so a tiny period cannot underflow the
    /// slack to zero. Eight times that norm covers a three-distance defect
    /// of `3 * ||η||_2` and rounding of the slack. The square-underflow
    /// floor is added on top.
    fn proven_slack(periods: &[Option<f64>]) -> f64 {
        let mut max_eta = 0.0_f64;
        let mut n_periodic = 0.0_f64;
        for period in periods {
            let Some(l) = *period else {
                continue;
            };
            let eta = mul_up(4.0, ulp_at(l));
            if !eta.is_finite() {
                return f64::INFINITY;
            }
            if eta > max_eta {
                max_eta = eta;
            }
            n_periodic += 1.0;
        }
        let under = square_underflow_floor(periods.len());
        if n_periodic == 0.0 || max_eta == 0.0 {
            return under;
        }
        let norm = mul_up(next_up(n_periodic.sqrt()), max_eta);
        let periodic = mul_up(8.0, norm);
        add_up(periodic, under)
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

    /// Lower bound on the absolute circular distance from `x` to `[lo, hi]`.
    ///
    /// The bound is zero when the interval may cover a full period, when `x`
    /// may lie on the arc, or when the gap is inside `4 * ulp(L)`. Otherwise
    /// it is that gap rounded down. It is a lower bound, not a tight one.
    fn periodic_axis_lower_abs(&self, x: f64, lo: f64, hi: f64, l: f64) -> f64 {
        let w = hi - lo;
        if !w.is_finite() || w < 0.0 || !l.is_finite() || l <= 0.0 {
            return 0.0;
        }
        let w_up = if w == 0.0 {
            0.0
        } else {
            let up = w + ulp_at(w);
            if !up.is_finite() {
                return 0.0;
            }
            up
        };
        if w_up >= l {
            return 0.0;
        }
        let (Ok(base), Ok(xw)) = (Self::canon(lo, l), Self::canon(x, l)) else {
            return 0.0;
        };
        let mut rel = xw - base;
        if rel < 0.0 {
            rel += l;
        }
        if !rel.is_finite() {
            return 0.0;
        }
        // Residue representatives and `rel` each miss the true forward
        // distance by at most two ulps of `L`, and the gap subtraction adds
        // half an ulp. Four ulps cover that. The threshold and the returned
        // gap are rounded so this function cannot land above the true gap.
        let budget = mul_up(4.0, ulp_at(l));
        if !budget.is_finite() || rel <= add_up(w_up, budget) {
            return 0.0;
        }
        let gap_fwd = rel - w_up;
        let gap_back = l - rel;
        let gap = if gap_fwd < gap_back {
            gap_fwd
        } else {
            gap_back
        };
        if !gap.is_finite() || gap <= 0.0 {
            return 0.0;
        }
        diff_down(gap, budget)
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
            diff_down(lo, x)
        } else if x > hi {
            diff_down(x, hi)
        } else {
            0.0
        };
        Ok(square_down(d))
    }

    fn aabb_min_dist2(&self, p: &[f64], lo: &[f64], hi: &[f64]) -> Result<f64, Error> {
        if p.len() != self.dims || lo.len() != self.dims || hi.len() != self.dims {
            return Err(Error::BadDimensions);
        }
        let mut s = 0.0;
        for i in 0..self.dims {
            s = add_down(s, self.axis_min_dist2(p[i], lo[i], hi[i], i)?);
        }
        Ok(s)
    }

    fn vp_absolute_slack(&self) -> f64 {
        square_underflow_floor(self.dims)
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
                // IEEE residue in [0, L). Exact when that residue is a
                // binary64 number; the seam `L` is stored as `0`.
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
            self.periodic_axis_lower_abs(x, lo, hi, l)
        } else if x < lo {
            diff_down(lo, x)
        } else if x > hi {
            diff_down(x, hi)
        } else {
            0.0
        };
        Ok(square_down(d))
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
            s = add_down(s, self.axis_min_dist2(p[i], lo[i], hi[i], i)?);
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

        // ±L are the same point on the circle. The raw difference overflows,
        // and the exact remainder does not: both residues are 0.
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
            2.0f64.powi(55),
            -(2.0f64.powi(55)),
            2.0f64.powi(62),
            -(2.0f64.powi(62)),
        ];
        for x in samples {
            let y = metric.wrap_point_axis(x, 0).unwrap();
            assert!(y >= 0.0 && y < l, "{x} -> {y}");
            assert_eq!(metric.wrap_point_axis(y, 0).unwrap(), y);
            assert_eq!(metric.dist2(&[x], &[x]).unwrap(), 0.0);
        }
    }

    #[test]
    fn remainder_matches_integer_residue() {
        let m = PeriodicSpace::new(vec![Some(10.0)]).unwrap();
        let two55 = 2.0f64.powi(55);
        let two62 = 2.0f64.powi(62);
        assert_eq!(m.dist2(&[two55], &[0.0]).unwrap(), 4.0);
        assert_eq!(m.dist2(&[two55], &[8.0]).unwrap(), 0.0);
        assert_eq!(m.dist2(&[two62], &[4.0]).unwrap(), 0.0);
        assert_eq!(m.dist2(&[two62], &[0.0]).unwrap(), 16.0);
        assert_eq!(m.dist2(&[1.0], &[11.0]).unwrap(), 0.0);
        assert_eq!(m.dist2(&[0.0], &[11.0]).unwrap(), 1.0);

        let m7 = PeriodicSpace::new(vec![Some(7.0)]).unwrap();
        assert_eq!(m7.dist2(&[1.25], &[8.25]).unwrap(), 0.0);

        let m1 = PeriodicSpace::new(vec![Some(1.0)]).unwrap();
        assert_eq!(m1.dist2(&[0.0], &[1.0]).unwrap(), 0.0);
        assert_eq!(m1.dist2(&[0.0], &[-1.0]).unwrap(), 0.0);
        assert_eq!(m1.dist2(&[0.0], &[0.5]).unwrap(), 0.25);
        let below = f64::from_bits(1.0f64.to_bits() - 1);
        assert_eq!(m1.dist2(&[0.0], &[below]).unwrap(), 2.0f64.powi(-106));
        // -1e-20 is closer to 0 than half an ulp of L=1, so the seam stores 0.
        let half_ulp = 0.5 * (1.0 - below);
        assert!(1.0e-20 < half_ulp);
        assert_eq!(m1.dist2(&[0.0], &[-1.0e-20]).unwrap(), 0.0);
    }

    #[test]
    fn slack_stays_positive_for_tiny_periods_and_open_axes() {
        let unit = super::square_underflow_unit();
        assert_eq!(unit * unit, f64::from_bits(1));
        let open = Euclidean::new(1).unwrap();
        let open_slack = open.vp_absolute_slack();
        assert!(open_slack.is_finite() && open_slack >= 8.0 * unit);

        let none = PeriodicSpace::new(vec![None]).unwrap();
        let none_slack = none.vp_absolute_slack();
        assert!(none_slack.is_finite() && none_slack >= 8.0 * unit);

        let tiny = PeriodicSpace::new(vec![Some(1.0e-150)]).unwrap();
        let tiny_slack = tiny.vp_absolute_slack();
        let ulp = super::ulp_at(1.0e-150);
        assert_eq!((4.0 * ulp) * (4.0 * ulp), 0.0);
        assert!(tiny_slack.is_finite() && tiny_slack > 8.0 * unit);
    }

    #[test]
    fn axis_lower_bound_does_not_exceed_integer_witness() {
        let period = PeriodicSpace::new(vec![Some(10.0)]).unwrap();
        let x = 1.0e13;
        let gap = period.axis_min_dist2(x, 3.0, 5.0, 0).unwrap();
        assert!(gap > 0.0 && gap <= 9.0, "{gap}");
        let on_arc = period.axis_min_dist2(x, 0.0, 1.0, 0).unwrap();
        assert_eq!(on_arc, 0.0);
        let two62 = 2.0f64.powi(62);
        let far = period.axis_min_dist2(two62, 0.0, 2.0, 0).unwrap();
        assert!(far > 0.0 && far <= 4.0, "{far}");

        let seven = PeriodicSpace::new(vec![Some(7.0)]).unwrap();
        let mod7 = seven.axis_min_dist2(x, 0.0, 1.0, 0).unwrap();
        assert!(mod7 > 0.0 && mod7 <= 4.0, "{mod7}");

        let wide = period.axis_min_dist2(x, 0.0, 10.0, 0).unwrap();
        assert_eq!(wide, 0.0);

        let open = Euclidean::new(1).unwrap();
        let outside = open.axis_min_dist2(x, x + 2.0, x + 5.0, 0).unwrap();
        assert!(outside > 0.0 && outside <= 4.0, "{outside}");
        assert_eq!(open.axis_min_dist2(x, x - 1.0, x + 1.0, 0).unwrap(), 0.0);

        let box_m = PeriodicSpace::new(vec![Some(10.0), Some(10.0)]).unwrap();
        let lb = box_m
            .aabb_min_dist2(&[x, 0.0], &[3.0, 3.0], &[5.0, 5.0])
            .unwrap();
        assert!(lb > 0.0 && lb <= 18.0, "{lb}");

        let lo = 0.3;
        let hi = 1.7;
        let bound = seven.axis_min_dist2(x, lo, hi, 0).unwrap();
        let mut nearest = f64::INFINITY;
        for i in 0..=4_000 {
            let y = lo + (hi - lo) * (i as f64 / 4_000.0);
            nearest = nearest.min(seven.dist2(&[x], &[y]).unwrap());
        }
        assert!(bound <= nearest, "{bound} > {nearest}");

        let open_edge = open.axis_min_dist2(x, x + 0.3, x + 0.7, 0).unwrap();
        let endpoint = open.dist2(&[x], &[x + 0.3]).unwrap();
        assert!(open_edge <= endpoint, "{open_edge} > {endpoint}");
    }
}

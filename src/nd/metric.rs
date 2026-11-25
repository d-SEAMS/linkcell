//! Metric shim for N-dimensional neighbour search.
//!
//! This module is intentionally small and isolated: once minimage ships its
//! N-dimensional metric traits and bounds, this implementation should be
//! replaced by the upstream one.

use super::dyadic::{self, DyadicSum};
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
    /// The default copies `src`. Periodic distance is applied later from the
    /// raw coordinates: a value in `[0, L)` cannot hold a negative offset
    /// smaller than `ulp(L)`.
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
///
/// The rounding residue comes from the exact significand product. A fused
/// multiply-add would link a runtime-dispatched FMA routine into the
/// library, outside `simd::select()` and the ISA allowlist.
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
    let (ma, ea) = dyadic::decode_f64(a);
    let (mb, eb) = dyadic::decode_f64(b);
    let (mp, ep) = dyadic::decode_f64(p);
    let mut resid = DyadicSum::new();
    resid.add_mant(ma * mb, ea + eb);
    resid.add_mant(-mp, ep);
    if resid.signum() > 0 {
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

/// Lower bound on `d*d`. Never exceeds the exact square, including when
/// that square is subnormal.
fn square_down(d: f64) -> f64 {
    dyadic::square_down(d)
}

/// Exact sum of a few binary64 terms. The terms are not required to be
/// non-overlapping: each [`two_sum`] keeps both outputs, so the list sums
/// to the exact real value.
struct Expansion {
    v: [f64; 8],
    n: usize,
}

impl Expansion {
    fn new() -> Self {
        Self { v: [0.0; 8], n: 0 }
    }

    fn push(&mut self, x: f64) -> Result<(), Error> {
        if x == 0.0 {
            return Ok(());
        }
        if !x.is_finite() || self.n == self.v.len() {
            return Err(Error::DistanceOverflow);
        }
        self.v[self.n] = x;
        self.n += 1;
        Ok(())
    }

    fn parts(&self) -> &[f64] {
        &self.v[..self.n]
    }

    fn grow(&mut self, x: f64) -> Result<(), Error> {
        if x == 0.0 {
            return Ok(());
        }
        if !x.is_finite() {
            return Err(Error::DistanceOverflow);
        }
        let mut q = x;
        let mut out = Expansion::new();
        for &term in self.parts() {
            let (s, h) = two_sum(q, term);
            if !s.is_finite() {
                return Err(Error::DistanceOverflow);
            }
            out.push(h)?;
            q = s;
        }
        out.push(q)?;
        *self = out;
        Ok(())
    }
}

/// Error-free transformation of `a + b`.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

fn open_expansion(x: f64, y: f64) -> Result<Expansion, Error> {
    if !(x - y).is_finite() {
        return Err(Error::DistanceOverflow);
    }
    let (hi, lo) = two_sum(x, -y);
    let mut exp = Expansion::new();
    exp.push(lo)?;
    exp.push(hi)?;
    Ok(exp)
}

fn add_expansion_square(acc: &mut DyadicSum, exp: &Expansion) {
    for (i, &left) in exp.parts().iter().enumerate() {
        let (mi, ei) = dyadic::decode_f64(left);
        acc.add_mant(mi * mi, ei * 2);
        for &right in exp.parts().iter().skip(i + 1) {
            let (mj, ej) = dyadic::decode_f64(right);
            acc.add_mant(mi * mj * 2, ei + ej);
        }
    }
}

fn sum_axis_squares(parts: &[Expansion]) -> Result<f64, Error> {
    let mut acc = DyadicSum::new();
    for exp in parts {
        add_expansion_square(&mut acc, exp);
    }
    acc.round_nearest()
}

/// Sign of `2 * sum(exp) - period`: positive means the sum is above `period / 2`.
fn twice_minus_period(exp: &Expansion, period: f64) -> i8 {
    let mut acc = DyadicSum::new();
    for &term in exp.parts() {
        let (m, e) = dyadic::decode_f64(term);
        acc.add_mant(m * 2, e);
    }
    let (ml, el) = dyadic::decode_f64(period);
    acc.add_mant(-ml, el);
    acc.signum()
}

/// Sign of `2 * sum(exp) + period`: negative means the sum is below `-period / 2`.
fn twice_plus_period(exp: &Expansion, period: f64) -> i8 {
    let mut acc = DyadicSum::new();
    for &term in exp.parts() {
        let (m, e) = dyadic::decode_f64(term);
        acc.add_mant(m * 2, e);
    }
    let (ml, el) = dyadic::decode_f64(period);
    acc.add_mant(ml, el);
    acc.signum()
}

/// Exact circular distance as a sum of binary64 terms.
///
/// Each coordinate is reduced with the IEEE remainder first, and only so the
/// subtraction stays finite. The remainder of the difference is what is
/// folded into `[-L/2, L/2]`. A coordinate is never reflected by subtracting
/// `L` before that difference exists.
fn periodic_expansion(x: f64, y: f64, l: f64) -> Result<Expansion, Error> {
    let mut rx = x % l;
    let mut ry = y % l;
    if !rx.is_finite() || !ry.is_finite() {
        return Err(Error::DistanceOverflow);
    }
    if !(rx - ry).is_finite() {
        if rx.abs() >= ry.abs() {
            rx = shift_toward_zero(rx, l)?;
        } else {
            ry = shift_toward_zero(ry, l)?;
        }
        if !(rx - ry).is_finite() {
            return Err(Error::DistanceOverflow);
        }
    }
    let (hi, lo) = two_sum(rx, -ry);
    let mut exp = Expansion::new();
    exp.push(lo)?;
    exp.push(hi)?;
    for _ in 0..4 {
        if twice_minus_period(&exp, l) > 0 {
            exp.grow(-l)?;
        } else if twice_plus_period(&exp, l) < 0 {
            exp.grow(l)?;
        } else {
            return Ok(exp);
        }
    }
    Err(Error::DistanceOverflow)
}

fn shift_toward_zero(r: f64, l: f64) -> Result<f64, Error> {
    let s = if r > 0.0 { r - l } else { r + l };
    if s.is_finite() {
        Ok(s)
    } else {
        Err(Error::DistanceOverflow)
    }
}

/// Greatest binary64 that does not exceed the absolute value of `exp`.
fn expansion_abs_down(exp: &Expansion) -> f64 {
    if exp.n == 0 {
        return 0.0;
    }
    let mut probe = DyadicSum::new();
    for &term in exp.parts() {
        let (m, e) = dyadic::decode_f64(term);
        probe.add_mant(m, e);
    }
    let sign = probe.signum();
    if sign == 0 {
        return 0.0;
    }
    let mut acc = DyadicSum::new();
    for &term in exp.parts() {
        let (m, e) = dyadic::decode_f64(term);
        let m = if sign < 0 { -m } else { m };
        acc.add_mant(m, e);
    }
    acc.round_down()
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

    /// Representative in `[0, L)`.
    ///
    /// The absolute value is reduced with the IEEE remainder. A negative
    /// input then uses `L - r` when that difference lies in `(0, L)`. A
    /// negative offset smaller than half an ulp of `L` becomes `0` here.
    /// [`Metric::dist2`] does not read this representative.
    fn canon(x: f64, l: f64) -> Result<f64, Error> {
        let r = x.abs() % l;
        if !r.is_finite() {
            return Err(Error::DistanceOverflow);
        }
        if x >= 0.0 || r == 0.0 {
            if r > 0.0 && r < l {
                Ok(r)
            } else {
                Ok(0.0)
            }
        } else {
            let y = l - r;
            if !y.is_finite() {
                return Err(Error::DistanceOverflow);
            }
            if y > 0.0 && y < l {
                Ok(y)
            } else {
                Ok(0.0)
            }
        }
    }

    /// Minimum-image representative in `[-L/2, L/2]`, with `+L/2` mapped to `-L/2`.
    ///
    /// `r` is the IEEE remainder of `|x|`, which is exact. `L - r` is formed
    /// only when `r >= L/2`, where Sterbenz' lemma makes it exact. A negative
    /// `x` with `r < L/2` stays negative. It is not passed through
    /// `rem_euclid`, which would add `L` and round a small gap up to `L`.
    fn minimum_image(x: f64, l: f64) -> Result<f64, Error> {
        let ax = x.abs();
        if !ax.is_finite() {
            return Err(Error::DistanceOverflow);
        }
        let r = ax % l;
        if !r.is_finite() {
            return Err(Error::DistanceOverflow);
        }
        if r == 0.0 {
            return Ok(0.0);
        }
        let twice = r + r;
        let image = if twice.is_finite() && twice == l {
            -r
        } else if !twice.is_finite() || twice > l {
            let alt = l - r;
            if !(alt.is_finite() && alt >= 0.0) {
                return Err(Error::DistanceOverflow);
            }
            if x >= 0.0 {
                -alt
            } else {
                alt
            }
        } else if x < 0.0 {
            -r
        } else {
            r
        };
        // A full period is the same point as 0. `0.5 * L` underflows for a
        // subnormal period, and a comparison with that zero used to return `-L`.
        if !image.is_finite() {
            return Err(Error::DistanceOverflow);
        }
        if image == l || image == -l {
            Ok(0.0)
        } else {
            Ok(image)
        }
    }

    /// Lower bound on the absolute circular distance.
    ///
    /// The value is the greatest binary64 that does not exceed the exact
    /// short image of `x - y`. It is not a difference of coordinates that
    /// were reflected before the subtraction.
    fn circular_abs(x: f64, y: f64, l: f64) -> Result<f64, Error> {
        Ok(expansion_abs_down(&periodic_expansion(x, y, l)?))
    }

    /// Absolute prune slack: a periodic term plus the square-underflow floor.
    ///
    /// The floor is an upper bound. A positive binary64 whose square
    /// underflows is strictly below `2^-537`. The periodic term is
    /// `8 * sqrt(p) * max(4 * ulp(L_i))`, where `p` is the number of
    /// periodic axes (not the point count). The products are rounded
    /// upward and no product of squares is formed, so a tiny period cannot
    /// zero the term. It is a prune guard, not a claim that `dist2` errs by
    /// `4 * ulp(L)`.
    fn proven_slack(periods: &[Option<f64>]) -> f64 {
        let mut max_eta = 0.0_f64;
        let mut periodic_axes = 0.0_f64;
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
            periodic_axes += 1.0;
        }
        let under = square_underflow_floor(periods.len());
        if periodic_axes == 0.0 || max_eta == 0.0 {
            return under;
        }
        let norm = mul_up(next_up(periodic_axes.sqrt()), max_eta);
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
            Some(l) => Self::minimum_image(delta, l)?,
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
    /// A degenerate interval uses [`Self::circular_abs`], the same remainder
    /// as `dist2`, so a small gap on a huge period stays positive. A wider
    /// interval is zero when its width may reach half a period. Otherwise the
    /// bound is the distance to the nearer endpoint when `x` is off the short
    /// arc between the endpoints, and zero when it may lie on that arc.
    fn periodic_axis_lower_abs(&self, x: f64, lo: f64, hi: f64, l: f64) -> f64 {
        let w = hi - lo;
        if !w.is_finite() || w < 0.0 || !l.is_finite() || l <= 0.0 {
            return 0.0;
        }
        let Ok(d_lo) = Self::circular_abs(x, lo, l) else {
            return 0.0;
        };
        if w == 0.0 {
            return diff_down(d_lo, 0.0);
        }
        let w_up = w + ulp_at(w);
        if !w_up.is_finite() || w_up >= l {
            return 0.0;
        }
        let half = l * 0.5;
        if !half.is_finite() || w_up > half {
            return 0.0;
        }
        let (Ok(d_hi), Ok(span)) = (Self::circular_abs(x, hi, l), Self::circular_abs(lo, hi, l))
        else {
            return 0.0;
        };
        let sum = d_lo + d_hi;
        if !sum.is_finite() {
            return 0.0;
        }
        let near = if d_lo < d_hi { d_lo } else { d_hi };
        let scale = sum.max(span).max(near);
        let tol = mul_up(8.0, ulp_at(scale));
        if !tol.is_finite() || sum <= add_up(span, tol) {
            return 0.0;
        }
        diff_down(near, 0.0)
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
        let mut parts = Vec::with_capacity(self.dims);
        for (x, y) in a.iter().zip(b.iter()) {
            parts.push(open_expansion(*x, *y)?);
        }
        sum_axis_squares(&parts)
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
        let mut parts = Vec::with_capacity(self.periods.len());
        for i in 0..self.periods.len() {
            let exp = if let Some(l) = self.periods[i] {
                periodic_expansion(a[i], b[i], l)?
            } else {
                open_expansion(a[i], b[i])?
            };
            parts.push(exp);
        }
        sum_axis_squares(&parts)
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

    fn vp_absolute_slack(&self) -> f64 {
        Self::proven_slack(&self.periods)
    }
}

#[cfg(test)]
mod tests {
    use super::{Euclidean, Metric, PeriodicSpace};
    use crate::Error;

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
        assert_eq!(
            periodic.dist2(&[0.0], &[6.0e199]),
            Err(Error::DistanceOverflow)
        );

        // ±L are the same point on the circle. The raw difference overflows,
        // and the exact remainder does not: both residues are 0.
        let images = PeriodicSpace::new(vec![Some(1.0e308)]).unwrap();
        assert_eq!(images.dist2(&[1.0e308], &[-1.0e308]).unwrap(), 0.0);

        // The raw difference overflows. Each coordinate is reduced modulo 10
        // first: both remainders are 2, and the squared gap is 16.
        let reduced = PeriodicSpace::new(vec![Some(10.0)]).unwrap();
        assert_eq!(reduced.dist2(&[1.7e308], &[-1.7e308]).unwrap(), 16.0);

        let euclid = Euclidean::new(1).unwrap();
        assert_eq!(
            euclid.dist2(&[1.0e308], &[-1.0e308]),
            Err(Error::DistanceOverflow)
        );
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
        // -1e-20 is below half an ulp of L=1, so the `[0, L)` wrap snaps to 0.
        // The distance keeps the representable gap.
        let tiny = -1.0e-20_f64;
        assert!(tiny.abs() < 0.5 * (1.0 - below));
        assert_eq!(m1.wrap_point_axis(tiny, 0).unwrap(), 0.0);
        assert_eq!(m1.dist2(&[0.0], &[tiny]).unwrap(), tiny.abs() * tiny.abs());
        assert_ne!(m1.dist2(&[0.0], &[tiny]).unwrap(), 0.0);
    }

    #[test]
    fn huge_period_keeps_small_negative_delta() {
        let cases = [
            (2.0f64.powi(53), -0.5, 0.25),
            (1.0e16, -1.0, 1.0),
            (1.0e300, -1.0, 1.0),
            (0.1 * 2.0f64.powi(60), -0.125, 0.015625),
        ];
        for (l, x, want) in cases {
            let metric = PeriodicSpace::new(vec![Some(l)]).unwrap();
            assert_eq!(metric.dist2(&[x], &[0.0]).unwrap(), want, "L={l:e}");
            assert_eq!(metric.dist2(&[0.0], &[x]).unwrap(), want, "L={l:e}");
            let bound = metric.axis_min_dist2(x, 0.0, 0.0, 0).unwrap();
            assert!(bound > 0.0 && bound <= want, "bound {bound} want {want}");
        }
        let wide = PeriodicSpace::new(vec![Some(1.0e16), Some(1.0e16)]).unwrap();
        let axis = wide.axis_min_dist2(-1.0, 0.0, 0.0, 0).unwrap();
        assert!(axis > 0.0 && axis <= 1.0, "{axis}");
        let lb = wide
            .aabb_min_dist2(&[-1.0, -0.5], &[0.0, 0.0], &[0.0, 0.0])
            .unwrap();
        assert!(lb > 0.0 && lb <= 1.25, "{lb}");
        let image = PeriodicSpace::new(vec![Some(2.0f64.powi(53))]).unwrap();
        assert_eq!(image.wrap_delta_axis(-0.5, 0).unwrap(), -0.5);
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
    }

    #[test]
    fn sum_of_squares_matches_a_binary64_total() {
        let u = 2.0_f64.powi(-27);
        let p = [1.0 + u, u, u, u];
        let z = [0.0; 4];
        let exact = 1.0 + 2.0_f64.powi(-26) + 2.0_f64.powi(-52);
        let got = Euclidean::new(4).unwrap().dist2(&p, &z).unwrap();
        assert_eq!(got.to_bits(), exact.to_bits());
    }

    #[test]
    fn difference_residue_keeps_gaps_that_straddle_half_period() {
        // ulp(2^52) is 1, so a half-unit gap at L/2 is not a binary64 coordinate.
        // The adjacent representable point is one unit away and must stay one unit.
        let l = 2.0_f64.powi(53);
        let half = 2.0_f64.powi(52);
        let m = PeriodicSpace::new(vec![Some(l)]).unwrap();
        assert_eq!(m.dist2(&[half - 1.0], &[half]).unwrap(), 1.0);
        assert_eq!(m.dist2(&[half + 1.0], &[half]).unwrap(), 1.0);
        // Two periods of length 4 keep the integer gap representable.
        let l4 = 4.0_f64;
        let m4 = PeriodicSpace::new(vec![Some(l4)]).unwrap();
        assert_eq!(m4.dist2(&[1.0 + 2.0 * l4], &[2.0]).unwrap(), 1.0);
        // A 1.5 gap is representable one binade lower.
        let l52 = 2.0_f64.powi(52);
        let half52 = 2.0_f64.powi(51);
        let m52 = PeriodicSpace::new(vec![Some(l52)]).unwrap();
        assert_ne!(half52 - 1.5, half52);
        assert_eq!(m52.dist2(&[half52 - 1.5], &[half52]).unwrap(), 2.25);

        let wide = PeriodicSpace::new(vec![Some(1.0e16)]).unwrap();
        assert_eq!(
            wide.dist2(&[-5.0e15], &[4_999_999_999_999_999.0]).unwrap(),
            1.0
        );
        assert_eq!(wide.wrap_delta_axis(-1.0, 0).unwrap(), -1.0);

        let huge = PeriodicSpace::new(vec![Some(1.0e300)]).unwrap();
        assert_eq!(
            huge.dist2(&[-5.0e299], &[4.999_999_999_999_999_5e299])
                .unwrap_err(),
            Error::DistanceOverflow
        );
        let finite = PeriodicSpace::new(vec![Some(1.0e308)]).unwrap();
        assert_eq!(finite.dist2(&[1.0], &[0.0]).unwrap(), 1.0);
        assert_eq!(finite.dist2(&[1.0e308], &[-1.0e308]).unwrap(), 0.0);
    }

    #[test]
    fn wrap_delta_does_not_return_a_full_subnormal_period() {
        let l = 5.0e-324_f64;
        let m = PeriodicSpace::new(vec![Some(l)]).unwrap();
        for &x in &[0.0_f64, l, -l, 1.0, -1.0, f64::MIN_POSITIVE, 1.0e300] {
            let got = m.wrap_delta_axis(x, 0).unwrap();
            assert!(got.is_finite(), "{x:e} -> {got}");
            assert_ne!(got, -l, "{x:e} returned -L");
            assert_ne!(got, l, "{x:e} returned L");
        }
        let tiny = PeriodicSpace::new(vec![Some(1.0e-10)]).unwrap();
        assert!(tiny.wrap_delta_axis(1.0e300, 0).unwrap().is_finite());
    }
}

//! Exact rationals for binary64 inputs.
//!
//! Every finite `f64` is a dyadic rational. Circular distances are integer
//! division of those rationals. Nothing here calls `Metric::dist2`.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive};

pub fn rat_from_f64(x: f64) -> BigRational {
    assert!(x.is_finite(), "{x}");
    if x == 0.0 {
        return BigRational::from_integer(BigInt::from(0));
    }
    let bits = x.to_bits();
    let neg = bits >> 63 != 0;
    let exp_bits = ((bits >> 52) & 0x7ff) as i32;
    let frac = (bits & ((1_u64 << 52) - 1)) as u128;
    let (mant, power) = if exp_bits == 0 {
        (frac, -1074)
    } else {
        (frac + (1_u128 << 52), exp_bits - 1075)
    };
    let mut num = BigInt::from(mant);
    let rat = if power >= 0 {
        num <<= power as usize;
        BigRational::from_integer(num)
    } else {
        let den = BigInt::one() << ((-power) as usize);
        BigRational::new(num, den)
    };
    if neg {
        -rat
    } else {
        rat
    }
}

/// Squared circular distance of the real values of two binary64 coordinates.
pub fn periodic_sq(a: f64, b: f64, l: f64) -> BigRational {
    let diff = (rat_from_f64(a) - rat_from_f64(b)).abs();
    let period = rat_from_f64(l);
    let q = (&diff / &period).floor();
    let mut r = &diff - &q * &period;
    let two = BigRational::from_integer(BigInt::from(2));
    if &r * &two > period {
        r = &period - &r;
    }
    &r * &r
}

/// Squared circular distance from `x` to the arc `[lo, hi]` of length below one period.
pub fn periodic_interval_sq(x: f64, lo: f64, hi: f64, l: f64) -> BigRational {
    let zero = BigRational::from_integer(BigInt::from(0));
    let x = rat_from_f64(x);
    let lo = rat_from_f64(lo);
    let hi = rat_from_f64(hi);
    let period = rat_from_f64(l);
    let w = &hi - &lo;
    if w < zero || w >= period {
        return zero;
    }
    let dx = &x - &lo;
    let q = (&dx / &period).floor();
    let forward = &dx - &q * &period;
    if forward <= w {
        return zero;
    }
    let ahead = &forward - &w;
    let behind = &period - &forward;
    let gap = if ahead < behind { ahead } else { behind };
    &gap * &gap
}

/// Squared distance from `x` to the open interval `[lo, hi]`.
pub fn open_interval_sq(x: f64, lo: f64, hi: f64) -> BigRational {
    let x = rat_from_f64(x);
    let lo = rat_from_f64(lo);
    let hi = rat_from_f64(hi);
    let gap = if x < lo {
        &lo - &x
    } else if x > hi {
        &x - &hi
    } else {
        return BigRational::from_integer(BigInt::from(0));
    };
    &gap * &gap
}

pub fn eq_f64(exact: &BigRational, got: f64) -> bool {
    got.is_finite() && rat_from_f64(got) == *exact
}

/// `(x - image) / L` is an integer.
pub fn congruent(x: f64, image: f64, l: f64) -> bool {
    let q = (rat_from_f64(x) - rat_from_f64(image)) / rat_from_f64(l);
    q.is_integer()
}

/// The binary64 value of `exact` when `exact` is one.
pub fn as_f64(exact: &BigRational) -> Option<f64> {
    let v = exact.to_f64()?;
    if v.is_finite() && rat_from_f64(v) == *exact {
        Some(v)
    } else {
        None
    }
}

/// `as_f64` when the square is binary64, otherwise the `ToPrimitive` conversion.
pub fn approx_f64(exact: &BigRational) -> f64 {
    as_f64(exact).unwrap_or_else(|| exact.to_f64().expect("rational square converts"))
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

/// `got` is a nearest binary64 to `exact`, ties to even.
pub fn is_nearest(got: f64, exact: &BigRational) -> bool {
    if !got.is_finite() {
        return false;
    }
    let err = (rat_from_f64(got) - exact).abs();
    for neighbor in [next_down(got), next_up(got)] {
        if !neighbor.is_finite() {
            continue;
        }
        let other = (rat_from_f64(neighbor) - exact).abs();
        if other < err {
            return false;
        }
        if other == err && neighbor.to_bits() & 1 == 0 && got.to_bits() & 1 == 1 {
            return false;
        }
    }
    true
}

/// Error of `got` measured in units of the larger adjacent binary64 gap.
pub fn error_ulps(got: f64, exact: &BigRational) -> f64 {
    if !got.is_finite() {
        return f64::INFINITY;
    }
    let err = (rat_from_f64(got) - exact).abs();
    if err == BigRational::from_integer(BigInt::from(0)) {
        return 0.0;
    }
    let ax = got.abs();
    let bits = ax.to_bits();
    let up = if bits == u64::MAX {
        f64::INFINITY
    } else {
        f64::from_bits(bits + 1) - ax
    };
    let down = if bits == 0 {
        ax
    } else {
        ax - f64::from_bits(bits - 1)
    };
    let gap = if up.is_finite() { up.max(down) } else { down };
    (&err / &rat_from_f64(gap))
        .to_f64()
        .unwrap_or(f64::INFINITY)
}

/// The short arc between the centered residues crosses the branch cut.
pub fn straddles_fold(a: f64, b: f64, l: f64) -> bool {
    let period = rat_from_f64(l);
    let half = &period / BigRational::from_integer(BigInt::from(2));
    let center = |x: f64| {
        let x = rat_from_f64(x);
        let q = (&x / &period).floor();
        let r = &x - &q * &period;
        if r >= half {
            &r - &period
        } else {
            r
        }
    };
    (center(a) - center(b)).abs() > half
}

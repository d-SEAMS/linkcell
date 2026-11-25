//! Exact dyadic sums of binary64 values, then one binary64 rounding.
//!
//! A finite `f64` is an integer times a power of two. Squares and cross
//! terms of those integers fit in `i128`. The accumulator below is that
//! integer, so a later rounding is the only place a value stops being exact.

use crate::Error;

const LIMB_BITS: i32 = 60;
const LIMB_BASE: i32 = -2400;
const N_LIMBS: usize = 96;

/// Exact sum of dyadic terms `mant * 2^exp`.
pub(super) struct DyadicSum {
    limbs: [i128; N_LIMBS],
    /// Carry above the top limb. Zero when the sum fits in the window.
    sign_carry: i128,
    overflow: bool,
}

impl DyadicSum {
    pub(super) fn new() -> Self {
        Self {
            limbs: [0; N_LIMBS],
            sign_carry: 0,
            overflow: false,
        }
    }

    pub(super) fn add_mant(&mut self, mant: i128, exp: i32) {
        if mant == 0 || self.overflow {
            return;
        }
        let neg = mant < 0;
        let mut mag = if neg { -mant } else { mant };
        if mag < 0 {
            self.overflow = true;
            return;
        }
        let mut e = exp;
        while mag != 0 {
            let chunk = mag & ((1_i128 << 50) - 1);
            let signed = if neg { -chunk } else { chunk };
            self.add_chunk(signed, e);
            if self.overflow {
                return;
            }
            mag >>= 50;
            e = e.saturating_add(50);
        }
    }

    fn add_chunk(&mut self, chunk: i128, exp: i32) {
        if chunk == 0 || self.overflow {
            return;
        }
        let rel = exp as i64 - LIMB_BASE as i64;
        if rel < 0 {
            // Below 2^-2400. That is far under the tie bit of the smallest
            // subnormal, so it cannot change a binary64 rounding.
            return;
        }
        let limb_bits = LIMB_BITS as i64;
        let idx = (rel / limb_bits) as usize;
        let off = (rel % limb_bits) as u32;
        if idx >= N_LIMBS {
            self.overflow = true;
            return;
        }
        self.limbs[idx] += chunk << off;
        self.carry_from(idx);
    }

    fn carry_from(&mut self, start: usize) {
        let mut carry = 0_i128;
        let mut first = true;
        for limb in self.limbs.iter_mut().skip(start) {
            if !first && carry == 0 {
                return;
            }
            first = false;
            let (q, r) = div_floor_limb(*limb + carry);
            *limb = r;
            carry = q;
        }
        if carry != 0 {
            self.sign_carry += carry;
            if self.sign_carry > 0 {
                self.overflow = true;
            }
        }
    }

    /// Sign of the exact sum: -1, 0, or 1. Zero means the sum is 0.
    pub(super) fn signum(&mut self) -> i8 {
        self.carry_from(0);
        if self.overflow || self.sign_carry > 0 {
            return 1;
        }
        if self.sign_carry < 0 {
            return -1;
        }
        if self.limbs.iter().any(|&limb| limb != 0) {
            1
        } else {
            0
        }
    }

    pub(super) fn round_nearest(&mut self) -> Result<f64, Error> {
        self.round(false)
    }

    pub(super) fn round_down(&mut self) -> f64 {
        self.round(true).unwrap_or(f64::MAX)
    }

    fn round(&mut self, down: bool) -> Result<f64, Error> {
        self.carry_from(0);
        if self.overflow || self.sign_carry > 0 {
            return if down {
                Ok(f64::MAX)
            } else {
                Err(Error::DistanceOverflow)
            };
        }
        if self.sign_carry < 0 {
            return if down {
                Ok(0.0)
            } else {
                Err(Error::DistanceOverflow)
            };
        }
        let Some(top) = self.top_bit() else {
            return Ok(0.0);
        };
        if top >= 1024 {
            return if down {
                Ok(f64::MAX)
            } else {
                Err(Error::DistanceOverflow)
            };
        }
        if top < -1074 {
            if down || top < -1075 || (top == -1075 && !self.any_bit_below(top)) {
                return Ok(0.0);
            }
            return Ok(f64::from_bits(1));
        }
        let (sig_bits, lsb_pos) = if top >= -1022 {
            (53_i32, top - 52)
        } else {
            (top - (-1074) + 1, -1074)
        };
        let mut sig = 0_u64;
        for i in 0..sig_bits {
            if self.bit(top - i) {
                sig |= 1_u64 << (sig_bits - 1 - i);
            }
        }
        let round_up = if down {
            false
        } else {
            let round_pos = lsb_pos - 1;
            let round_bit = self.bit(round_pos);
            let sticky = self.any_bit_below(round_pos);
            if round_bit && sticky {
                true
            } else {
                round_bit && (sig & 1) == 1
            }
        };
        if round_up {
            sig += 1;
        }
        if top >= -1022 {
            if sig >> 53 != 0 {
                let exp = top + 1;
                if exp > 1023 {
                    return Err(Error::DistanceOverflow);
                }
                return Ok(f64_from_exp_sig(exp, 0));
            }
            let frac = sig & ((1_u64 << 52) - 1);
            return Ok(f64_from_exp_sig(top, frac));
        }
        if sig >> sig_bits != 0 {
            let new_top = top + 1;
            if new_top >= -1022 {
                return Ok(f64::from_bits(1_u64 << 52));
            }
            let shift = (new_top - (-1074)) as u32;
            return Ok(f64::from_bits(1_u64 << shift));
        }
        Ok(f64::from_bits(sig))
    }

    fn top_bit(&self) -> Option<i32> {
        for i in (0..N_LIMBS).rev() {
            let limb = self.limbs[i];
            if limb != 0 {
                let bit = 127 - limb.leading_zeros() as i32;
                return Some(LIMB_BASE + i as i32 * LIMB_BITS + bit);
            }
        }
        None
    }

    fn bit(&self, pos: i32) -> bool {
        if pos < LIMB_BASE {
            return false;
        }
        let rel = pos - LIMB_BASE;
        let idx = (rel / LIMB_BITS) as usize;
        let off = (rel % LIMB_BITS) as u32;
        if idx >= N_LIMBS {
            return false;
        }
        ((self.limbs[idx] >> off) & 1) == 1
    }

    fn any_bit_below(&self, pos: i32) -> bool {
        if pos <= LIMB_BASE {
            return false;
        }
        let rel = pos - LIMB_BASE;
        let idx = (rel / LIMB_BITS) as usize;
        let off = (rel % LIMB_BITS) as u32;
        if idx >= N_LIMBS {
            return self.limbs.iter().any(|&limb| limb != 0);
        }
        if off > 0 {
            let mask = (1_i128 << off) - 1;
            if self.limbs[idx] & mask != 0 {
                return true;
            }
        }
        self.limbs[..idx].iter().any(|&limb| limb != 0)
    }
}

pub(super) fn decode_f64(x: f64) -> (i128, i32) {
    if x == 0.0 {
        return (0, 0);
    }
    let bits = x.to_bits();
    let neg = bits >> 63 != 0;
    let exp_bits = ((bits >> 52) & 0x7ff) as i32;
    let frac = (bits & ((1_u64 << 52) - 1)) as i128;
    let (mant, exp) = if exp_bits == 0 {
        (frac, -1074)
    } else {
        (frac | (1_i128 << 52), exp_bits - 1023 - 52)
    };
    if neg {
        (-mant, exp)
    } else {
        (mant, exp)
    }
}

fn div_floor_limb(v: i128) -> (i128, i128) {
    let mut q = v >> LIMB_BITS;
    let mut r = v - (q << LIMB_BITS);
    if r < 0 {
        r += 1_i128 << LIMB_BITS;
        q -= 1;
    }
    (q, r)
}

fn f64_from_exp_sig(exp: i32, frac: u64) -> f64 {
    let biased = (exp + 1023) as u64;
    f64::from_bits((biased << 52) | (frac & ((1_u64 << 52) - 1)))
}

/// Greatest binary64 that does not exceed `d * d`.
pub(super) fn square_down(d: f64) -> f64 {
    let d = d.abs();
    if !(d.is_finite() && d > 0.0) {
        return 0.0;
    }
    let (m, e) = decode_f64(d);
    let mut acc = DyadicSum::new();
    acc.add_mant(m * m, e * 2);
    acc.round_down()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nearest(mant: i128, exp: i32) -> f64 {
        let mut acc = DyadicSum::new();
        acc.add_mant(mant, exp);
        acc.round_nearest().unwrap()
    }

    #[test]
    fn powers_of_two_round_to_themselves() {
        assert_eq!(nearest(1, 0), 1.0);
        assert_eq!(nearest(1, -1), 0.5);
        assert_eq!(nearest(1, -52), 2.0f64.powi(-52));
        assert_eq!(nearest(1, -1074).to_bits(), 1);
        assert_eq!(nearest(1, -1075), 0.0);
        assert_eq!(nearest(3, -1076).to_bits(), 1);
        assert_eq!(nearest(1, 10), 1024.0);
    }

    #[test]
    fn tie_to_even_stays_on_the_even_significand() {
        // 1 + 2^-53 is halfway between 1 and the next binary64. 1 is even.
        let mut acc = DyadicSum::new();
        acc.add_mant(1, 0);
        acc.add_mant(1, -53);
        assert_eq!(acc.round_nearest().unwrap(), 1.0);
    }

    #[test]
    fn counterexample_sum_of_squares_is_the_binary64_value() {
        // (1 + 2^-27)^2 + 3 * (2^-27)^2 = 1 + 2^-26 + 2^-52, which is binary64.
        let u = 2.0_f64.powi(-27);
        let coords = [1.0 + u, u, u, u];
        let mut acc = DyadicSum::new();
        for x in coords {
            let (m, e) = decode_f64(x);
            acc.add_mant(m * m, e * 2);
        }
        let exact = 1.0 + 2.0_f64.powi(-26) + 2.0_f64.powi(-52);
        assert_eq!(acc.round_nearest().unwrap().to_bits(), exact.to_bits());
    }

    #[test]
    fn square_down_never_exceeds_the_exact_square() {
        let mut raised = 0_u32;
        let mut subnormal_squares = 0_u32;
        for exp in 1..2046_u64 {
            for mant in [1_u64, 3, 5, 9, (1_u64 << 52) - 1, 0x0005_a5a5_a5a5] {
                let d = f64::from_bits((exp << 52) | mant);
                let down = square_down(d);
                assert!(
                    !exceeds_exact_square(d, down),
                    "square_down({d:e}) = {down:e} exceeds the exact square"
                );
                let naive = d * d;
                if naive.is_finite() && down < naive {
                    raised += 1;
                }
                if down > 0.0 && down < f64::MIN_POSITIVE {
                    subnormal_squares += 1;
                    assert!(down < naive || naive == down);
                }
            }
        }
        for bits in 1..4096_u64 {
            let d = f64::from_bits(bits);
            assert_eq!(square_down(d), 0.0);
            assert!(!exceeds_exact_square(d, 0.0));
        }
        assert!(raised > 0, "no rounded product sat above the exact square");
        assert!(
            subnormal_squares > 0,
            "the sweep never produced a subnormal square"
        );
    }

    /// `got * 2^0` compared with `d^2` by aligning two integers. `true` when `got` is strictly larger.
    fn exceeds_exact_square(d: f64, got: f64) -> bool {
        if !(got.is_finite() && got > 0.0) {
            return false;
        }
        let (md, ed) = decode_f64(d.abs());
        let (mg, eg) = decode_f64(got);
        let sq = (md as u128) * (md as u128);
        let e_sq = ed * 2;
        !le_dyadic(mg as u128, eg, sq, e_sq)
    }

    fn le_dyadic(a: u128, ea: i32, b: u128, eb: i32) -> bool {
        if a == 0 {
            return true;
        }
        if b == 0 {
            return false;
        }
        if ea >= eb {
            let s = (ea - eb) as u32;
            if s >= 128 {
                return false;
            }
            match a.overflowing_shl(s) {
                (_, true) => false,
                (shifted, false) => shifted <= b,
            }
        } else {
            let s = (eb - ea) as u32;
            if s >= 128 {
                return true;
            }
            match b.overflowing_shl(s) {
                (_, true) => true,
                (shifted, false) => a <= shifted,
            }
        }
    }
}

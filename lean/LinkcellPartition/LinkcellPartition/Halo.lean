import Mathlib.Algebra.Order.Floor.Ring
import Mathlib.Tactic

namespace Linkcell

/-- Bin radius along one axis. `h` is the perpendicular bin width. -/
noncomputable def binRadius (w h : ℝ) : ℕ := ⌈w / h⌉₊

/-- `binRadius w h` bins of width `h` cover a segment of length `w`. -/
theorem binRadius_covers {w h : ℝ} (_hw : 0 ≤ w) (hh : 0 < h) :
    w ≤ (binRadius w h : ℝ) * h := by
  have hne : h ≠ 0 := ne_of_gt hh
  have hceil : w / h ≤ (⌈w / h⌉₊ : ℝ) := Nat.le_ceil (w / h)
  have := mul_le_mul_of_nonneg_right hceil (le_of_lt hh)
  simpa [binRadius, div_mul_cancel₀ w hne] using this

/-- Half-open bins. A point of bin `j ≥ i` within distance `w` of a point
of bin `i` has index gap at most `binRadius w h`. -/
theorem index_gap_of_ball {i j : ℕ} {h w p q : ℝ}
    (hh : 0 < h) (_hw : 0 ≤ w)
    (hp_hi : p < ((i + 1 : ℕ) : ℝ) * h)
    (hq_lo : (j : ℝ) * h ≤ q)
    (hqw : q - p ≤ w) (hij : i ≤ j) :
    j - i ≤ binRadius w h := by
  classical
  by_cases hclose : j = i
  · simp [hclose]
  · have hlt : i < j := lt_of_le_of_ne hij (Ne.symm hclose)
    have hstep : i + 1 ≤ j := Nat.succ_le_of_lt hlt
    have hdiff : ((j - (i + 1) : ℕ) : ℝ) * h < q - p := by
      calc
        ((j - (i + 1) : ℕ) : ℝ) * h
            = ((j : ℝ) - ((i + 1 : ℕ) : ℝ)) * h := by rw [Nat.cast_sub hstep]
        _ = (j : ℝ) * h - ((i + 1 : ℕ) : ℝ) * h := by ring
        _ ≤ q - ((i + 1 : ℕ) : ℝ) * h := sub_le_sub_right hq_lo _
        _ < q - p := sub_lt_sub_left hp_hi q
    have hltw : ((j - (i + 1) : ℕ) : ℝ) * h < w := lt_of_lt_of_le hdiff hqw
    have hk : ((j - (i + 1) : ℕ) : ℝ) < w / h := (lt_div_iff₀ hh).2 hltw
    have hnat : j - (i + 1) < binRadius w h := (Nat.lt_ceil).2 hk
    have hsucc : j - (i + 1) + 1 ≤ binRadius w h := Nat.succ_le_of_lt hnat
    have hpos : 0 < j - i := Nat.sub_pos_of_lt hlt
    have hone : 1 ≤ j - i := Nat.succ_le_of_lt hpos
    have hcancel : j - (i + 1) + 1 = j - i := by
      have : j - i - 1 + 1 = j - i := Nat.sub_add_cancel hone
      simpa [Nat.sub_sub, Nat.add_comm 1 i] using this
    simpa [hcancel, binRadius] using hsucc

/-- A periodic face. Lift the low bin by one period and reuse the linear gap. -/
theorem periodic_wrap_covered {n i j : ℕ} {h w p q : ℝ}
    (hh : 0 < h) (hw : 0 ≤ w) (hi : i ≤ j + n)
    (hp_hi : p < ((i + 1 : ℕ) : ℝ) * h)
    (hq_lo : (j : ℝ) * h ≤ q)
    (himg : q + (n : ℝ) * h - p ≤ w) :
    (j + n) - i ≤ binRadius w h := by
  have hq_lift : ((j + n : ℕ) : ℝ) * h ≤ q + (n : ℝ) * h := by
    have hcast : ((j + n : ℕ) : ℝ) = (j : ℝ) + (n : ℝ) := by
      simp [Nat.cast_add]
    rw [hcast]
    have : (j : ℝ) * h + (n : ℝ) * h ≤ q + (n : ℝ) * h := by
      have := add_le_add_right hq_lo ((n : ℝ) * h)
      simpa [add_mul, add_comm, add_left_comm, add_assoc] using this
    simpa [mul_add, add_mul, add_comm, add_left_comm, add_assoc] using this
  have hgap :=
    index_gap_of_ball (i := i) (j := j + n) (h := h) (w := w) (p := p)
      (q := q + (n : ℝ) * h) hh hw hp_hi hq_lift himg hi
  simpa using hgap

end Linkcell

import Mathlib.Tactic

namespace Linkcell

/-- `weight * parts * den ≤ total * num`, the test applied before an FM move. -/
def underBalance (weight total num den parts : ℕ) : Prop :=
  weight * parts * den ≤ total * num

/-- Numerator of the derived bound `1 + parts * wMax / total`. -/
def derivedNum (total wMax parts : ℕ) : ℕ :=
  total + parts * wMax

/-- One indivisible cell of weight `wMax` fits the derived bound. -/
theorem heavy_cell_fits (total wMax parts : ℕ) :
    underBalance wMax total (derivedNum total wMax parts) total parts := by
  unfold underBalance derivedNum
  have hsum : wMax * parts ≤ total + parts * wMax := by
    have : wMax * parts = parts * wMax := by ring
    rw [this]
    exact Nat.le_add_left (parts * wMax) total
  calc
    wMax * parts * total = total * (wMax * parts) := by ring
    _ ≤ total * (total + parts * wMax) := Nat.mul_le_mul_left _ hsum

/-- Part weights after moving `w` from `src` to `dst`. -/
def moved {k : ℕ} (wt : Fin k → ℕ) (src dst : Fin k) (w : ℕ) : Fin k → ℕ :=
  fun i => if i = src then wt src - w else if i = dst then wt dst + w else wt i

/-- An accepted Fiduccia-Mattheyses move keeps every part inside the bound.
The source must contain the moved weight, and the two updated part weights
must already have passed `underBalance`. Untouched parts are unchanged. -/
theorem fm_move_keeps {k : ℕ} {wt : Fin k → ℕ} {src dst : Fin k} {w total num den : ℕ}
    (hdiff : src ≠ dst) (hw : w ≤ wt src)
    (hsrc : underBalance (wt src - w) total num den k)
    (hdst : underBalance (wt dst + w) total num den k)
    (hold : ∀ i, i ≠ src → i ≠ dst → underBalance (wt i) total num den k) :
    ∀ i, underBalance (moved wt src dst w i) total num den k := by
  intro i
  have _hdiff : src ≠ dst := hdiff
  have _hw : w ≤ wt src := hw
  by_cases hs : i = src
  · have heq : moved wt src dst w i = wt src - w := by
      unfold moved
      rw [if_pos hs]
    rw [heq]
    exact hsrc
  · by_cases hd : i = dst
    · have heq : moved wt src dst w i = wt dst + w := by
        unfold moved
        rw [if_neg hs, if_pos hd]
      rw [heq]
      exact hdst
    · have heq : moved wt src dst w i = wt i := by
        unfold moved
        rw [if_neg hs, if_neg hd]
      rw [heq]
      exact hold i hs hd

/-- Each cell is sent to exactly one part. -/
theorem cell_has_one_part {n parts : ℕ} (assign : Fin n → Fin parts) (i : Fin n) :
    ∃! p : Fin parts, assign i = p := by
  refine ⟨assign i, rfl, ?_⟩
  intro p hp
  exact hp.symm

/-- The part preimages cover every cell and are pairwise disjoint. -/
theorem parts_cover_and_disjoint {n parts : ℕ} (assign : Fin n → Fin parts) :
    (∀ i, ∃ p, assign i = p) ∧
      (∀ p q : Fin parts, p ≠ q → ∀ i, assign i = p → assign i ≠ q) := by
  constructor
  · intro i
    exact ⟨assign i, rfl⟩
  · intro p q hpq i hip hiq
    exact hpq (hip.symm.trans hiq)

/-- Weight above a cap. -/
def excess (wt cap : ℕ) : ℕ := wt - min wt cap

theorem excess_under (wt cap : ℕ) (h : wt ≤ cap) : excess wt cap = 0 := by
  unfold excess
  rw [min_eq_left h]
  exact Nat.sub_self wt

theorem excess_over (wt cap : ℕ) (h : cap ≤ wt) : excess wt cap = wt - cap := by
  unfold excess
  rw [min_eq_right h]

/-- Moving a positive weight off an overweight part, into a part that stays
under the cap, lowers the total excess. The source must contain the weight. -/
theorem repair_lowers_excess (src dst w cap : ℕ) (hw : 0 < w) (hs : w ≤ src)
    (hover : cap < src) (hdst : dst + w ≤ cap) :
    excess (src - w) cap + excess (dst + w) cap < excess src cap + excess dst cap := by
  have hdst0 : excess dst cap = 0 :=
    excess_under dst cap (le_trans (Nat.le_add_right dst w) hdst)
  have hnew0 : excess (dst + w) cap = 0 := excess_under (dst + w) cap hdst
  have hold : excess src cap = src - cap := excess_over src cap (le_of_lt hover)
  by_cases hle : cap ≤ src - w
  · have hnew : excess (src - w) cap = src - w - cap := excess_over (src - w) cap hle
    have hlt : src - w - cap < src - cap := by omega
    have : excess (src - w) cap + excess (dst + w) cap < excess src cap + excess dst cap := by
      rw [hnew, hnew0, hold, hdst0]
      simpa using hlt
    exact this
  · have hunder : src - w ≤ cap := le_of_not_ge hle
    have hnew : excess (src - w) cap = 0 := excess_under (src - w) cap hunder
    have hlt : 0 < src - cap := by omega
    have : excess (src - w) cap + excess (dst + w) cap < excess src cap + excess dst cap := by
      rw [hnew, hnew0, hold, hdst0]
      simpa using hlt
    exact this

/-- `(2 * before + weight) * parts` fits in 128 bits for machine weights. -/
theorem owner_product_fits (before weight parts : ℕ) (hb : before < 2 ^ 64)
    (hw : weight < 2 ^ 64) (hp : parts < 2 ^ 32) :
    (2 * before + weight) * parts < 2 ^ 128 := by
  have h3 : (3 : ℕ) * 2 ^ 96 < 2 ^ 128 := by norm_num
  have hb' : before ≤ 2 ^ 64 - 1 := Nat.le_sub_one_of_lt hb
  have hw' : weight ≤ 2 ^ 64 - 1 := Nat.le_sub_one_of_lt hw
  have hp' : parts ≤ 2 ^ 32 - 1 := Nat.le_sub_one_of_lt hp
  have hsum : 2 * before + weight ≤ 3 * 2 ^ 64 - 3 := by omega
  have hprod : (2 * before + weight) * parts ≤ (3 * 2 ^ 64 - 3) * (2 ^ 32 - 1) :=
    Nat.mul_le_mul hsum hp'
  have hclosed : (3 * 2 ^ 64 - 3) * (2 ^ 32 - 1) < 3 * 2 ^ 96 := by omega
  exact lt_of_le_of_lt hprod (lt_trans hclosed h3)

end Linkcell

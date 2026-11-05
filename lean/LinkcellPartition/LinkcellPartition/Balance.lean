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

end Linkcell

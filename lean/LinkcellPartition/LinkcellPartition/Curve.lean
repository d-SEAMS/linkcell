import Mathlib.Data.Nat.Basic
import Mathlib.Tactic

namespace Linkcell

/-- Part index of a cell whose weight starts at `before`. -/
def owner (before weight total parts : ℕ) : ℕ :=
  min (parts - 1) ((2 * before + weight) * parts / (2 * total))

theorem owner_lt {before weight total parts : ℕ}
    (hp : 0 < parts) (_ht : 0 < total) :
    owner before weight total parts < parts := by
  have hsub : parts - 1 < parts := Nat.sub_lt hp (by decide)
  exact lt_of_le_of_lt (Nat.min_le_left _ _) hsub

/-- Prefix walk. A zero total uses weight 1 and `len` as the total. -/
def assignGo (total len parts : ℕ) : List ℕ → ℕ → List ℕ
  | [], _ => []
  | w :: rest, before =>
    let ww := if total = 0 then 1 else w
    let tot := if total = 0 then len else total
    owner before ww tot parts :: assignGo total len parts rest (before + ww)

/-- One owner per cell, in curve order. -/
def assign (ws : List ℕ) (parts : ℕ) : List ℕ :=
  assignGo ws.sum ws.length parts ws 0

theorem assignGo_length (total len parts : ℕ) (rest : List ℕ) (before : ℕ) :
    (assignGo total len parts rest before).length = rest.length := by
  induction rest generalizing before with
  | nil => simp [assignGo]
  | cons w rest ih =>
    simp [assignGo, ih]

theorem assign_length (ws : List ℕ) (parts : ℕ) :
    (assign ws parts).length = ws.length := by
  simp [assign, assignGo_length]

theorem assignGo_owners_lt {total len parts : ℕ} (hp : 0 < parts)
    (ht : 0 < (if total = 0 then len else total)) (rest : List ℕ) (before : ℕ) :
    ∀ o ∈ assignGo total len parts rest before, o < parts := by
  induction rest generalizing before with
  | nil => simp [assignGo]
  | cons w rest ih =>
    intro o ho
    simp [assignGo] at ho
    rcases ho with rfl | ho
    · exact owner_lt (before := before) (weight := if total = 0 then 1 else w)
        (total := if total = 0 then len else total) hp ht
    · exact ih (before + if total = 0 then 1 else w) o ho

theorem assign_owners_lt {ws : List ℕ} {parts : ℕ}
    (hp : 0 < parts) (hne : ws ≠ []) :
    ∀ o ∈ assign ws parts, o < parts := by
  have ht : 0 < (if ws.sum = 0 then ws.length else ws.sum) := by
    by_cases hz : ws.sum = 0
    · simpa [hz, List.length_pos_iff] using hne
    · simpa [hz] using Nat.pos_of_ne_zero hz
  simpa [assign] using
    assignGo_owners_lt (total := ws.sum) (len := ws.length) (parts := parts) hp ht ws 0

end Linkcell

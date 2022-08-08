Normal ⇐ {
  𝕊 rand‿⟨mu, sig⟩:
    q‿u‿v ← 3⥊0
    {
      u ↩ rand.Range 0
      v ↩ 1.7156 × (rand.Range 0) - 0.5
      x ← u - 0.449871
      y ← 0.386595 + |v
      q ↩ (x⋆2) + y × (y×0.19600) - x×0.25472
      𝕊⍟((q>0.27597) ∧ ((q>0.27846) ∨ ((v⋆2) > ¯4.0×(⋆⁼u)×u⋆2))) @
    } @
    mu+sig×v÷u
  ;
  𝕊 rand:
    Normal rand‿⟨0, 1⟩
  ;
  shape 𝕊 rand‿⟨mu, sig⟩:
    { Normal (rand‿⟨𝕩, sig⟩) }¨ shape⥊mu
  ;
  shape 𝕊 rand:
    shape Normal rand‿⟨0, 1⟩
}

InitializeParameters ⇐ {
  nx‿nh‿ny:
  rand ← •MakeRand 0
  parameters ← {
    w1 ⇐ nh‿nx Normal rand
    b1 ⇐ nh⥊0
    w2 ⇐ ny‿nh Normal rand
    b2 ⇐ ny⥊0
  }
}

InitializeParametersDeep ⇐ {
  𝕊 ns:  # output size of each layer: input, hiddens, output
  rand ← •MakeRand 0
  ws ← (Normal⟜rand)¨<˘⌽˘2↕ns
  bs ← ⥊⟜0¨ 1↓ns
  ws‿bs
}

⟨Dot⟩ ← •Import "../../../libs/numb.bqn"

LinearForward ⇐ {
  a‿w‿b:
  # a is m×n, n is number of examples
  z ← (w Dot a) + b
  z‿⟨a‿w‿b⟩
}

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
  w1 ← 0.01 × nh‿nx Normal rand
  b1 ← nh⥊0
  w2 ← 0.01 × ny‿nh Normal rand
  b2 ← ny⥊0
  parameters ← ⟨w1‿b1, w2‿b2⟩
}

InitializeParametersDeep ⇐ {
  𝕊 ns:  # output size of each layer: input, hiddens, output
  rand ← •MakeRand 0
  #ws ← 0.01 × (Normal⟜rand)¨<˘⌽˘2↕ns
  ws ← (Normal⟜rand¨<˘⌽˘2↕ns) ÷ √¯1↓ns
  bs ← ⥊⟜0¨ 1↓ns
  ws⋈¨bs
}

⟨Dot⟩ ← •Import "../../../libs/numb.bqn"
#Dot ← {
#  1≡⊑≠𝕨? ≍+´ (⥊𝕨) × <˘ 𝕩;
#  1≡1⊑≢𝕩? (≠𝕨)‿1⥊ (+´(⥊𝕩)⊸×)˘𝕨;
#  𝕨 +˝∘×⎉1‿∞ 𝕩
#}

LinearForward ⇐ {
  a‿w‿b:
  # a is m×n, n is number of examples
  z ← (w Dot a) + b
  cache ← a‿w‿b
  z‿cache
}

Sigmoid ⇐ {
  cache ← 𝕩
  a ← ÷1+⋆-𝕩
  a‿cache
}

ReLU ⇐ {
  cache ← 𝕩
  a ← 0⌈𝕩
  a‿cache
}

LinearActivationForward ⇐ {
  aPrev‿w‿b‿activation:
  # linearCache: a w b
  z‿linearCache ← LinearForward aPrev‿w‿b
  # activationCache: z
  a‿activationCache ← Activation z
  # cache: [a w b], z
  cache ← linearCache‿activationCache
  a‿cache
}

LModelForward ⇐ {
  x‿ps:
  a ← x
  cache ← @
  caches ← {
    w‿b:
    a‿cache ↩ LinearActivationForward a‿w‿b‿reLU
    cache
  }¨ ¯1↓ps
  w‿b ← ¯1⊑ps
  a‿cache ↩ LinearActivationForward a‿w‿b‿sigmoid
  # caches: [ [ [a w b], z ], [ [a w b], z ], ... ]
  caches ∾↩ ⋈cache
  a‿caches
}

ComputeCost ⇐ {
  al‿y:
  m ← 1⊑≢y
  cost ← -m÷˜+´⥊(y×⋆⁼al)+(¬y)×⋆⁼¬al
}

LinearBackward ⇐ {
  # cache: linearCache: a w b
  dz‿cache:
  aPrev‿w‿b ← cache
  m ← 1⊑≢aPrev
  dw ← m÷˜ dz Dot ⍉aPrev
  db ← m÷˜ ⥊+´˘dz
  daPrev ← (⍉w) Dot dz
  daPrev‿dw‿db
}

SigmoidBackward ⇐ {
  # cache: z
  da‿cache:
  z ← cache
  s ← ÷1+⋆-z
  dz ← da × s × ¬s
}

ReLUBackward ⇐ {
  # cache: z
  da‿cache:
  z ← cache
  dz ← (z>0) × da
}

LinearActivationBackward ⇐ {
  da‿cache‿activationBackward:
  linearCache‿activationCache ← cache
  dz ← ActivationBackward da‿activationCache
  daPrev‿dw‿db ← LinearBackward dz‿linearCache
  daPrev‿dw‿db
}

LModelBackward ⇐ {
  aL‿y‿caches:
  cache ← ¯1⊑caches
  daL ← - (y÷aL) - (¬y)÷¬aL
  da‿dw‿db ← LinearActivationBackward daL‿cache‿SigmoidBackward
  grads ← ⟨da‿dw‿db⟩
  grads ∾↩ {
    𝕊 cache:
    da‿dw‿db ↩ LinearActivationBackward da‿cache‿ReLUBackward
    da‿dw‿db
  }¨ ⌽¯1↓caches
  ⌽grads
}

UpdateParameters ⇐ {
  parameters‿grads‿learningRate:
  gradsNoA ← 1↓¨ grads
  parameters -↩ learningRate × gradsNoA
}

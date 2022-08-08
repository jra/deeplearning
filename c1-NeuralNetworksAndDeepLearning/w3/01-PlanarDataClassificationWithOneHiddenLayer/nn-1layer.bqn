# Normal Deviates by Ratio-of-Uniforms
# Numerical Recipes 7.3.9
Normal ⇐ {
  r 𝕊 mu‿sig:
  rand ← •MakeRand r
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
}

#rng ← •MakeRand 0

#n ← rng Normal 0‿1
#•Show n
#n ↩ rng Normal 0‿1
#•Show n
#ns ← {rng Normal 0‿1}¨100⥊0
#•Show ns

#n ← (rng.Range 0) Normal 0‿1
#•Show n
#n ↩ (rng.Range 0) Normal 0‿1
#•Show n
#ns ← {𝕩 Normal 0‿1}¨ 1000 rng.Range 0
#•Show ≠ns
#•Show ⌊´ns
#•Show ⌈´ns
#mean_ns ← (+´÷≠) ns
#•Out "mean_ns: "∾•Repr mean_ns
#var_ns ← (≠ns)÷˜+´2⋆˜ns-mean_ns
#•Out "var_ns: "∾•Repr var_ns

FmtNeg ⇐ {
  a ← •Fmt 𝕩
  (⊣+=×'-'⊸-)⟜'¯' a
}

#"normals.csv" •file.Lines ⟨"vals"⟩∾FmtNeg¨ns

LayerSizes ⇐ {
  x‿y:
  n_x ← ⊑≢x
  n_h ← 4
  n_y ← (1==y)⊑⟨⊑≢y, 1⟩
  n_x‿n_h‿n_y
}

InitializeParameters ⇐ {
  n_x‿n_h‿n_y:
  rng ← •MakeRand 2
  cache ← {
    w1 ⇐ 0.01 × {𝕩 Normal 0‿1}¨ n_h‿n_x rng.Range 0
    b1 ⇐ n_h‿1⥊0
    w2 ⇐ 0.01 × {𝕩 Normal 0‿1}¨ n_y‿n_h rng.Range 0
    b2 ⇐ n_y‿1⥊0
  }
}

Sigmoid ← { ÷1+⋆-𝕩 }
TanH ← { (1-˜⋆2×𝕩) ÷ 1+⋆2×𝕩 }
#TanH ← { •math.tanh }
#ReLU ← { 𝕩⌈0 }
#ReLULeaky ← { 𝕩⌈0.01×𝕩 }

⟨Dot⟩ ← •Import "../../../libs/numb.bqn"

ForwardPropagation ⇐ {
  x‿⟨w1, b1, w2, b2⟩:
  {
    # I'd use ⊏˘ or ≍⁼˘ to remove an extra trailing axis from a column vector.
    # Note that +˝∘×⎉1‿∞ doesn't require any special casing for lists. It treats lists as row vectors on the left, or column vectors on the right.
    # I think (m +˝∘×⎉1‿∞ x) + b should work whether x is a list or n×1 column, if b is a list.
    #  treat vectors as lists and rearrange my thinking/equations
    # Yeah. I tend to think it's the matrix product that's confusing. What it does is to pair the last axis of 𝕨 with the first axis of 𝕩. That shared axis gets reduced out, and any other axes (none in the case of a list) go into the result, in order.
    z1 ⇐ b1 ⥊⊸+ w1 Dot x  # TODO better way to broadcast?
    a1 ⇐ Tanh¨ z1
    z2 ⇐ b2 ⥊⊸+ w2 Dot a1
    a2 ⇐ Sigmoid¨ z2
  }
}

ComputeCost ⇐ {
  a2‿y‿⟨w1, b1, w2, b2⟩:
  m ← 1⊑≢y
  cost ← -m÷˜+´⥊(y×⋆⁼a2)+(¬y)×⋆⁼¬a2
}

BackwardPropagation ⇐ {
  ⟨w1,b1,w2,b2⟩‿⟨z1,a1,z2,a2⟩‿x‿y:
  m ← 1⊑≢y
  dz2 ← a2-y
  dw2 ← m÷˜ dz2 Dot ⍉a1
  db2 ← m÷˜ ≍ +´˘ dz2
  dz1 ← ((⍉w2) Dot dz2) × ¬a1⋆2
  dw1 ← m÷˜ dz1 Dot ⍉x
  db1 ← m÷˜ ⍉ ≍ +´˘ dz1
  grads ← {
    dw1 ⇐ dw1
    db1 ⇐ db1
    dw2 ⇐ dw2
    db2 ⇐ db2
  }
}

UpdateParameters ⇐ {
  ⟨w1,b1,w2,b2⟩‿⟨dw1,db1,dw2,db2⟩‿learning_rate:
  parameters ← {
    w1 ⇐ w1 - learning_rate × dw1
    b1 ⇐ b1 - learning_rate × db1
    w2 ⇐ w2 - learning_rate × dw2
    b2 ⇐ b2 - learning_rate × db2
  }
}

NNModel ⇐ {
  x‿y‿n_h‿learning_rate‿num_iterations‿print_cost:
  n_x‿·‿n_y ← LayerSizes x‿y
  cache‿cost‿grads ← @‿@‿@
  parameters ← InitializeParameters n_x‿n_h‿n_y
  i ← 0
  {𝕤
    cache ↩ ForwardPropagation x‿parameters
    cost ↩ ComputeCost cache.a2‿y‿parameters
    grads ↩ BackwardPropagation parameters‿cache‿x‿y
    parameters ↩ UpdateParameters parameters‿grads‿learning_rate
    {𝕤
      •Out "Costs after iteration "∾(•Repr i)∾": "∾•Repr cost
    }⍟(print_cost ∧ 0=1000|i) @
    i +↩ 1
  }⍟num_iterations @
  parameters
}

Predict ⇐ {
  parameters‿x:
  ⟨a2⟩ ← ForwardPropagation x‿parameters
  predictions ← a2 > 0.5
}

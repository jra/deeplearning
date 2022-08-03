# Normal Deviates by Ratio-of-Uniforms
# Numerical Recipes 7.3.9
Normal ← {
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

rng ← •MakeRand 0

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

FmtNeg ← {
  a ← •Fmt 𝕩
  (⊣+=×'-'⊸-)⟜'¯' a
}
#
#"normals.csv" •file.Lines ⟨"vals"⟩∾FmtNeg¨ns

LoadPlanarDataset ← {
  ·:
  rng ← •MakeRand 1
  m ← 400 # number of examples
  n ← ⌊m÷2 # number of points per class
  d ← 2 # dimensionality
  #x ← m‿D⥊0 # data matrix where each row is a single example
  #y ← m‿1⥊0 # labels vector (0 for red, 1 for blue)
  a ← 4 # maximum ray of the flower
  xy ← {
    t ← ((3.12×𝕩) + (3.12×(n-1)÷˜↕n)) + 0.2 × {𝕩 Normal 0‿1}¨ n rng.Range 0
    r ← (a×•math.Sin¨4×t) + 0.2 × {𝕩 Normal 0‿1}¨ n rng.Range 0
    x0 ← r×•math.Sin¨t
    x1 ← r×•math.Cos¨t
    y ← n⥊𝕩
    (⍉x0≍x1)‿y
  }¨0‿1
  x ← ∾⊑¨xy
  y ← ∾1⊸⊑¨xy
  ⍉¨x‿y
}

xy ← LoadPlanarDataset @
•Show ≢⊑xy
•Show ≢1⊑xy

ds ← ⍉(⊑xy)∾1⊑xy
"dataset.csv" •file.Lines ⟨"x0,x1,y"⟩∾{(FmtNeg 0⊑𝕩)∾","∾(FmtNeg 1⊑𝕩)∾","∾(FmtNeg 2⊑𝕩)}¨<˘ds

x ← ⊑xy
y ← 1⊑xy
shape_x ← ≢x
shape_y ← ≢y
m ← ≠y
•Out "The shape of x is: "∾•Repr shape_x
•Out "The shape of y is: "∾•Repr shape_y
•Out "I have m = "∾(•Repr m)∾" training examples!"

LayerSizes ← {
  x‿y:
  n_x ← ⊑≢x
  n_h ← 4
  n_y ← (1==y)⊑⟨⊑≢y, 1⟩
  n_x‿n_h‿n_y
}

LayerSizesTesCase ← {
  ·:
  rng ← •MakeRand 1
  x_assess ← 5‿3 rng.Range 0
  y_assess ← 2‿3 rng.Range 0
  x_assess‿y_assess
}

n_x‿n_h‿n_y ← 3⥊@

x_assess‿y_assess ← LayerSizesTesCase @
n_x‿n_h‿n_y ↩ LayerSizes x_assess‿y_assess
•Out "The size of the input layer is: n_x = "∾ •Repr n_x
•Out "The size of the hidden layer is: n_h = "∾ •Repr n_h
•Out "The size of the output layer is: n_y = "∾ •Repr n_y

InitializeParameters ← {
  n_x‿n_h‿n_y:
  rng ← •MakeRand 2

  w1 ← 0.01 × {𝕩 Normal 0‿1}¨ n_h‿n_x rng.Range 0
  b1 ← n_h‿1⥊0
  w2 ← 0.01 × {𝕩 Normal 0‿1}¨ n_y‿n_h rng.Range 0
  b2 ← n_y‿1⥊0

  •Show ≢w1
  •Show ≢b1
  •Show ≢w2
  •Show ≢b2

  #! n_h‿n_x=≢w1
  #! n_h‿1=≢b1
  #! n_y‿n_h=≢w2
  #! n_y‿1=≢b2

  w1‿b1‿w2‿b2
}

InitializeParametersTestCase ← {
  ·:
  n_x‿n_h‿n_y ← 2‿4‿1
  n_x‿n_h‿n_y
}

n_x‿n_h‿n_y ↩ InitializeParametersTestCase @

parameters ← InitializeParameters n_x‿n_h‿n_y
•Out "W1 = "∾•Repr 0⊑parameters
•Out "b1 = "∾•Repr 1⊑parameters
•Out "W2 = "∾•Repr 2⊑parameters
•Out "b2 = "∾•Repr 3⊑parameters

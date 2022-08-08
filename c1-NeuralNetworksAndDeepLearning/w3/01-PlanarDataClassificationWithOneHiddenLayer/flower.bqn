nn ← •Import "nn-1layer.bqn"
⟨Dot⟩ ← •Import "../../../libs/numb.bqn"

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
    t ← ((3.12×𝕩) + (3.12×(n-1)÷˜↕n)) + 0.2 × {𝕩 nn.Normal 0‿1}¨ n rng.Range 0
    r ← (a×•math.Sin¨4×t) + 0.2 × {𝕩 nn.Normal 0‿1}¨ n rng.Range 0
    x0 ← r×•math.Sin¨t
    x1 ← r×•math.Cos¨t
    y ← n⥊𝕩
    (⍉x0≍x1)‿y
  }¨0‿1
  x ← ∾⊑¨xy
  y ← ⍉≍∾1⊸⊑¨xy
  ⍉¨x‿y
}

x‿y ← LoadPlanarDataset @
#•Show ≢⊑xy
#•Show ≢1⊑xy

#ds ← ⍉(⊑xy)∾1⊑xy
ds ← (⍉x)∾˘⍉y
•Out "ds:"
•Show ≢ds
"dataset.csv" •file.Lines ⟨"x0,x1,y"⟩∾{(nn.FmtNeg 0⊑𝕩)∾","∾(nn.FmtNeg 1⊑𝕩)∾","∾(nn.FmtNeg 2⊑𝕩)}¨<˘ds

#x ← ⊑xy
#y ← 1⊑xy
shape_x ← ≢x
shape_y ← ≢y
m ← ≠y
•Out "The shape of x is: "∾•Repr shape_x
•Out "The shape of y is: "∾•Repr shape_y
•Out "I have m = "∾(•Repr m)∾" training examples!"

•Out "x:"
•Show ≢x
•Out "y:"
•Show ≢y

parameters ← nn.NNModel x‿y‿4‿1.2‿10000‿1
predictions ← nn.Predict parameters‿x
•Out "Accuracy: "∾•Repr 100× (1⊑≢y)÷˜ (y Dot ⍉predictions) + (¬y) Dot ⍉¬predictions

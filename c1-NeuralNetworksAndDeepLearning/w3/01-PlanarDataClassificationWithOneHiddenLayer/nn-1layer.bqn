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
"dataset.csv" •file.Lines ⟨"x0,x1,y"⟩∾{(FmtNeg 0⊑𝕩)∾","∾(FmtNeg 1⊑𝕩)∾","∾(FmtNeg 2⊑𝕩)}¨<˘ds

#x ← ⊑xy
#y ← 1⊑xy
shape_x ← ≢x
shape_y ← ≢y
m ← ≠y
•Out "The shape of x is: "∾•Repr shape_x
•Out "The shape of y is: "∾•Repr shape_y
•Out "I have m = "∾(•Repr m)∾" training examples!"

LayerSizes ⇐ {
  x‿y:
  n_x ← ⊑≢x
  n_h ← 4
  n_y ← (1==y)⊑⟨⊑≢y, 1⟩
  n_x‿n_h‿n_y
}

n_x‿n_h‿n_y ← 3⥊@
x_assess‿y_assess ← 2⥊@

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

  {
    w1⇐w1
    b1⇐b1
    w2⇐w2
    b2⇐b2
  }
}

InitializeParametersTestCase ← {
  ·:
  n_x‿n_h‿n_y ← 2‿4‿1
  n_x‿n_h‿n_y
}

n_x‿n_h‿n_y ↩ InitializeParametersTestCase @

parameters ← InitializeParameters n_x‿n_h‿n_y
#•Out "W1 = "∾•Repr 0⊑parameters
#•Out "b1 = "∾•Repr 1⊑parameters
#•Out "W2 = "∾•Repr 2⊑parameters
#•Out "b2 = "∾•Repr 3⊑parameters

•Out "W1 = "∾•Repr parameters.w1
•Out "b1 = "∾•Repr parameters.b1
•Out "W2 = "∾•Repr parameters.w2
•Out "b2 = "∾•Repr parameters.b2

Sigmoid ← { ÷1+⋆-𝕩 }
TanH ← { (1-˜⋆2×𝕩) ÷ 1+⋆2×𝕩 }
#TanH ← { •math.tanh }
#ReLU ← { 𝕩⌈0 }
#ReLULeaky ← { 𝕩⌈0.01×𝕩 }

⟨Dot⟩ ← •Import "../../w2/02-LogisticRegressionWithANeuralNetworkMindset/numb.bqn"
ForwardPropagation ← {
  x‿⟨w1, b1, w2, b2⟩:
  {
    #z1⇐(⥊b1) +⎉0‿1 w1 Dot x  # TODO better way to broadcast?
    z1⇐(⥊b1) + w1 Dot x  # TODO better way to broadcast?
    a1⇐Tanh¨ z1
    z2⇐(⥊b2) + w2 Dot a1
    a2⇐Sigmoid¨ z2
  }
}

ForwardPropagationTestCase ← {
  ·:
  #rng ← •MakeRand 1
  #x_assess ← {𝕩 Normal 0‿1}¨ 2‿3 rng.Range 0
  x_assess ← [[ 1.62434536, -0.61175641, -0.52817175]
              [-1.07296862,  0.86540763, -2.3015387 ]]
  #b1 ← {𝕩 Normal 0‿1}¨ 4‿1 rng.Range 0
  b1 ← [[ 1.74481176]
        [-0.7612069 ]
        [ 0.3190391 ]
        [-0.24937038]]
  b2 ← 1‿1⥊¯1.3
  parameters ← {
    w1⇐[[-0.00416758, -0.00056267]
        [-0.02136196,  0.01640271]
        [-0.01793436, -0.00841747]
        [ 0.00502881, -0.01245288]]
    b1⇐b1
    w2⇐[[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]
    b2⇐b2
  }
  x_assess‿parameters
}

x_assess‿parameters ↩ ForwardPropagationTestCase @

•Show x_assess
•Out "w1"
•Show parameters.w1
•Out "b1"
•Show parameters.b1
•Out "w2"
•Show parameters.w2
•Out "b2"
•Show parameters.b2

cache ← ForwardPropagation x_assess‿parameters
Mean ← { (+´⥊𝕩) ÷ (≠⥊𝕩) }  # TODO better way?
•Show (Mean cache.z1)‿(Mean cache.a1)‿(Mean cache.z2)‿(Mean cache.a2)

ComputeCost ← {
  a2‿y‿⟨w1, b1, w2, b2⟩:
  m ← 1⊑≢y
  -m÷˜+´⥊(y×⋆⁼a2)+(¬y)×⋆⁼¬a2
}

ComputeCostTestCase ← {
  ·:
  a2 ← [[0.5002307,  0.49985831, 0.50023963]]
  y_assess ← [[1, 0, 0]]
  parameters ← {
    w1 ⇐ [[-0.00416758, -0.00056267]
          [-0.02136196,  0.01640271]
          [-0.01793436, -0.00841747]
          [ 0.00502881, -0.01245288]]
    w2 ⇐ [[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]
    b1 ⇐ [[0]
          [0]
          [0]
          [0]]
    b2 ⇐ [[0]]
  }
  a2‿y_assess‿parameters
}

a2 ← @
a2‿y_assess‿parameters ↩ ComputeCostTestCase @
cost ← ComputeCost a2‿y_assess‿parameters
•Show cost

BackwardPropagation ← {
  ⟨w1,b1,w2,b2⟩‿⟨z1,a1,z2,a2⟩‿x‿y:
  m ← 1⊑≢y
  dz2 ← a2-y
  dw2 ← m÷˜ dz2 Dot ⍉a1
  #db2 ← m÷˜ +´ ⥊dz2
  #•Out "dz2:"
  #•Show dz2
  #•Show +˘dz2
  db2 ← m÷˜ ≍ +´˘ dz2
  dz1 ← ((⍉w2) Dot dz2) × ¬a1⋆2
  #•Out "dz1:"∾•Repr ≢dz1
  dw1 ← m÷˜ dz1 Dot ⍉x
  db1 ← m÷˜ ⍉ ≍ +´˘ dz1
  {
    dw1 ⇐ dw1
    db1 ⇐ db1
    dw2 ⇐ dw2
    db2 ⇐ db2
  }
}

BackwardPropagationTestCase ← {
  ·:
  parameters ← {
    w1 ⇐ [[-0.00416758, -0.00056267]
          [-0.02136196,  0.01640271]
          [-0.01793436, -0.00841747]
          [ 0.00502881, -0.01245288]]
    w2 ⇐ [[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]
    b1 ⇐ [[0]
          [0]
          [0]
          [0]]
    b2 ⇐ [[0]]
  }
  cache ← {
    a1 ⇐ [[-0.00616578,  0.0020626 ,  0.00349619]
          [-0.05225116,  0.02725659, -0.02646251]
          [-0.02009721,  0.0036869 ,  0.02883756]
          [ 0.02152675, -0.01385234,  0.02599885]]
    a2 ⇐ [[0.5002307 , 0.49985831, 0.50023963]]
    z1 ⇐ [[-0.00616586,  0.0020626 ,  0.0034962 ]
          [-0.05229879,  0.02726335, -0.02646869]
          [-0.02009991,  0.00368692,  0.02884556]
          [ 0.02153007, -0.01385322,  0.02600471]]
    z2 ⇐ [[ 0.00092281, -0.00056678,  0.00095853]]
  }
  x_assess ← [[ 1.62434536, -0.61175641, -0.52817175]
              [-1.07296862,  0.86540763, -2.3015387 ]]
  y_assess ← [[1, 0, 1]]
  parameters‿cache‿x_assess‿y_assess
}

parameters‿cache‿x_assess‿y_assess ↩ BackwardPropagationTestCase @

grads ← BackwardPropagation parameters‿cache‿x_assess‿y_assess

•Out "dw1: "∾•Repr grads.dw1
•Out "db1: "∾•Repr grads.db1
•Out "dw2: "∾•Repr grads.dw2
•Out "db2: "∾•Repr grads.db2

UpdateParameters ← {
  ⟨w1,b1,w2,b2⟩‿⟨dw1,db1,dw2,db2⟩‿learning_rate:
  #•Out "w1:"∾•Repr ≢w1
  #•Out "b1:"∾•Repr ≢b1
  #•Out "w2:"∾•Repr ≢w2
  #•Out "b2:"∾•Repr ≢b2
  #•Out "dw1:"∾•Repr ≢dw1
  #•Out "db1:"∾•Repr ≢db1
  #•Out "dw2:"∾•Repr ≢dw2
  #•Out "db2:"∾•Repr ≢db2
  w1 ↩ w1 - learning_rate × dw1
  b1 ↩ b1 - learning_rate × db1
  w2 ↩ w2 - learning_rate × dw2
  b2 ↩ b2 - learning_rate × db2
  {
    w1 ⇐ w1
    b1 ⇐ b1
    w2 ⇐ w2
    b2 ⇐ b2
  }
}

UpdateParametersTestCase ← {
  ·:
  parameters ← {
    w1 ⇐ [[-0.00643025,  0.01936718]
          [-0.02410458,  0.03978052]
          [-0.01653973, -0.02096177]
          [ 0.01046864, -0.05990141]]
    b1 ⇐ [[-1.02420756e¯06]
          [ 1.27373948e¯05]
          [ 8.32996807e¯07]
          [-3.20136836e¯06]]
    w2 ⇐ [[-0.01041081, -0.04463285,  0.01758031,  0.04747113]]
    b2 ⇐ [[0.00010457]]
  }
  grads ← {
    dw1 ⇐ [[ 0.00023322, -0.00205423]
           [ 0.00082222, -0.00700776]
           [-0.00031831,  0.0028636 ]
           [-0.00092857,  0.00809933]]
    dw2 ⇐ [[-1.75740039e¯05,  3.70231337e¯03, -1.25683095e¯03, -2.55715317e¯03]]
    db1 ⇐ [[ 1.05570087e¯07]
           [-3.81814487e¯06]
           [-1.90155145e¯07]
           [ 5.46467802e¯07]]
    db2 ⇐ [[-1.0892314e¯05]]
  }
  parameters‿grads
}

parameters‿grads ↩ UpdateParametersTestCase @
#•Show parameters
#•Show grads

parameters ↩ UpdateParameters parameters‿grads‿1.2

•Out "w1: "∾•Repr parameters.w1
•Out "b1: "∾•Repr parameters.b1
•Out "w2: "∾•Repr parameters.w2
•Out "b2: "∾•Repr parameters.b2

NNModel ← {
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

NNModelTestCase ← {𝕤
  x_assess ← [[ 1.62434536, -0.61175641, -0.52817175]
              [-1.07296862,  0.86540763, -2.3015387 ]]
  y_assess ← [[1, 0, 1]]
  x_assess‿y_assess
}

x_assess‿y_assess ↩ NNModelTestCase @

•Out "x_assess = "
•Show x_assess
•Out "y_assess = "
•Show y_assess

parameters ↩ NNModel x_assess‿y_assess‿4‿1.02‿10000‿1

•Out "w1 = "
•Show parameters.w1
•Out "b1 = "
•Show parameters.b1
•Out "w2 = "
•Show parameters.w2
•Out "b2 = "
•Show parameters.b2

Predict ← {
  parameters‿x:
  ⟨a2⟩ ← ForwardPropagation x‿parameters
  predictions ← a2 > 0.5
  #predictions
}

PredictTestCase ← {𝕤
  parameters ← {
    w1 ⇐ [[-0.00615039,  0.0169021 ]
          [-0.02311792,  0.03137121]
          [-0.0169217 , -0.01752545]
          [ 0.00935436, -0.05018221]]
    b1 ⇐ [[-8.97523455e¯07]
          [ 8.15562092e¯06]
          [ 6.04810633e¯07]
          [-2.54560700e¯06]]
    w2 ⇐ [[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]
    b2 ⇐ [[9.14954378e¯05]]
  }
  x_assess ← [[ 1.62434536, -0.61175641, -0.52817175]
              [-1.07296862,  0.86540763, -2.3015387 ]]
  parameters‿x_assess
}

parameters‿x_assess ↩ PredictTestCase @

predictions ← Predict parameters‿x_assess

•Out "predictions:"
•Show predictions

•Out "x:"
•Show ≢x
•Out "y:"
•Show ≢y

parameters ↩ NNModel x‿y‿4‿1.2‿10000‿1
predictions ↩ Predict parameters‿x
•Out "Accuracy: "∾•Repr 100× (1⊑≢y)÷˜ (y Dot ⍉predictions) + (¬y) Dot ⍉¬predictions

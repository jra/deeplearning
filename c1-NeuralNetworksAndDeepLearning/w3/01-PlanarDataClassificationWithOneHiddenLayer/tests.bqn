nn ← •Import "nn-1layer.bqn"

n_x‿n_h‿n_y ← 3⥊@
x_assess‿y_assess ← 2⥊@
parameters ← @

LayerSizesTesCase ← {
  ·:
  rng ← •MakeRand 1
  x_assess ← 5‿3 rng.Range 0
  y_assess ← 2‿3 rng.Range 0
  x_assess‿y_assess
}

x_assess‿y_assess ↩ LayerSizesTesCase @
n_x‿n_h‿n_y ↩ nn.LayerSizes x_assess‿y_assess
•Out "The size of the input layer is: n_x = "∾ •Repr n_x
•Out "The size of the hidden layer is: n_h = "∾ •Repr n_h
•Out "The size of the output layer is: n_y = "∾ •Repr n_y

InitializeParametersTestCase ← {
  ·:
  n_x‿n_h‿n_y ← 2‿4‿1
  n_x‿n_h‿n_y
}

n_x‿n_h‿n_y ↩ InitializeParametersTestCase @

parameters ↩ nn.InitializeParameters n_x‿n_h‿n_y
#•Out "W1 = "∾•Repr 0⊑parameters
#•Out "b1 = "∾•Repr 1⊑parameters
#•Out "W2 = "∾•Repr 2⊑parameters
#•Out "b2 = "∾•Repr 3⊑parameters

•Out "W1 = "∾•Repr parameters.w1
•Out "b1 = "∾•Repr parameters.b1
•Out "W2 = "∾•Repr parameters.w2
•Out "b2 = "∾•Repr parameters.b2

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

cache ← nn.ForwardPropagation x_assess‿parameters
Mean ← { (+´⥊𝕩) ÷ (≠⥊𝕩) }  # TODO better way?
•Show (Mean cache.z1)‿(Mean cache.a1)‿(Mean cache.z2)‿(Mean cache.a2)

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
cost ← nn.ComputeCost a2‿y_assess‿parameters
•Show cost

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

grads ← nn.BackwardPropagation parameters‿cache‿x_assess‿y_assess

•Out "dw1: "∾•Repr grads.dw1
•Out "db1: "∾•Repr grads.db1
•Out "dw2: "∾•Repr grads.dw2
•Out "db2: "∾•Repr grads.db2

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

parameters ↩ nn.UpdateParameters parameters‿grads‿1.2

•Out "w1: "∾•Repr parameters.w1
•Out "b1: "∾•Repr parameters.b1
•Out "w2: "∾•Repr parameters.w2
•Out "b2: "∾•Repr parameters.b2

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

parameters ↩ nn.NNModel x_assess‿y_assess‿4‿1.02‿10000‿1

•Out "w1 = "
•Show parameters.w1
•Out "b1 = "
•Show parameters.b1
•Out "w2 = "
•Show parameters.w2
•Out "b2 = "
•Show parameters.b2

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

predictions ← nn.Predict parameters‿x_assess

•Out "predictions:"
•Show predictions

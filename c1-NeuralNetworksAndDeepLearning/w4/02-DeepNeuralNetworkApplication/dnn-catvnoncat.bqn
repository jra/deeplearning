path_libreaddata ← "/Users/jra/prj/coursera/deeplearning/c1-NeuralNetworksAndDeepLearning/w2/02-LogisticRegressionWithANeuralNetworkMindset/libreaddata.dylib"
readTrainX ← path_libreaddata •FFI "a"‿"read_train_x"
readTrainY ← path_libreaddata •FFI "a"‿"read_train_y"
readTestX ← path_libreaddata •FFI "a"‿"read_test_x"
readTestY ← path_libreaddata •FFI "a"‿"read_test_y"
readTestListClasses ← path_libreaddata •FFI "a"‿"read_test_list_classes"

train_x_orig ← @ -˜ ReadTrainX ⟨⟩
train_y ← 1‿∘⥊ ReadTrainY ⟨⟩
test_x_orig ← @ -˜ ReadTestX ⟨⟩
test_y ← 1‿∘⥊ ReadTestY ⟨⟩
classes ← ReadTestListClasses ⟨⟩

index ← 25
#index ← 208
•Out "y = "∾(•Repr index⊑⊏train_y)∾", it's a '"∾((index⊑⊏train_y)⊑classes)∾"' picture."
ImageWrite ← {
  # write image to file:
  # ppm ex.
  # 4 8  4 columns, 8 rows
  # 255  max value
  # data read in rows
  rows‿cols‿ignore ← ≢𝕩  # ie 64 × 64 × 3
  header_ppm ← "P6
"∾(•Repr cols)∾" "∾(•Repr rows)∾"
255
"
  image_ppm ← @⊸+¨ ⥊𝕩
  bytes_ppm ← header_ppm ∾ image_ppm
  𝕨 •file.Bytes bytes_ppm
}
("img-train"∾(•Repr index)∾".ppm") ImageWrite index ⊏ train_x_orig

m_train ← ≠train_x_orig
m_test ← ≠test_x_orig
num_px ← 1⊑≢train_x_orig

•Out "Number of training examples: m_train = "∾•Repr m_train
•Out "Number of testing examples: m_test = "∾•Repr m_test
•Out "Height/Width of each image: num_px = "∾•Repr num_px
•Out "Each image is of size: "∾•Repr ≢⊏train_x_orig
•Out "train_x shape: "∾•Repr ≢train_x_orig
•Out "train_y shape: "∾•Repr ≢train_y
•Out "test_x shape: "∾•Repr ≢test_x_orig
•Out "test_y shape: "∾•Repr ≢test_y

train_x_flatten ← ⍉ (≠train_x_orig)‿∘ ⥊ train_x_orig
test_x_flatten ← ⍉ (≠test_x_orig)‿∘ ⥊ test_x_orig

train_x ← train_x_flatten ÷ 255
test_x ← test_x_flatten ÷ 255

ReportArray ← {
  nm‿ar:
  n ← ≠⥊ar
  mean ← (+´÷≠)⥊ar
  {𝕤
    •Show mean
    •Show mean - ar
    •Show ⋆⟜2 mean - ar
  }⍟(n<0) @
  stdv ← √ ÷⟜n +´ ⥊ ⋆⟜2 mean - ar
  •Out nm∾": shape: "∾(•Repr ≢ar)∾" mean: "∾(•Repr mean)∾" stdv: "∾(•Repr stdv)
}

ReportArray "train_x"‿train_x
ReportArray "train_y"‿train_y

n_x ← ⊑≢train_x  # 12288 num_px * num_px * 3
n_h ← 7
n_y ← 1
layers_dims ← n_x‿n_h‿n_y

nn ← •Import "../01-BuildingYourDeepNeuralNetworkStepByStep/nn.bqn"

TwoLayerModel ← {
  x‿y‿layers_dims‿learning_rate‿num_iterations‿print_cost:
  costs ← @
  m ← 1⊑≢x
  #parameters ← nn.InitializeParameters layers_dims
  parameters ← nn.InitializeParametersDeep layers_dims
  #w1i ← 0⊑0⊑parameters
  #w2i ← 0⊑1⊑parameters
  #ReportArray "w1i"‿w1i
  #ReportArray "w2i"‿w2i
  i ← 0
  {𝕤
    ⟨w1‿b1, w2‿b2⟩ ← parameters
    a1‿cache1 ← nn.LinearActivationForward x‿w1‿b1‿nn.reLU
    a2‿cache2 ← nn.LinearActivationForward a1‿w2‿b2‿nn.sigmoid
    cost ← nn.ComputeCost a2‿y
    da2 ← - (y÷a2) - (¬y)÷¬a2
    da1‿dw2‿db2 ← nn.LinearActivationBackward da2‿cache2‿nn.sigmoidBackward
    da0‿dw1‿db1 ← nn.LinearActivationBackward da1‿cache1‿nn.reluBackward
    grads ← ⟨⟨⟩‿dw1‿db1, ⟨⟩‿dw2‿db2⟩
    parameters ↩ nn.UpdateParameters parameters‿grads‿learning_rate
    {𝕤
      costs ∾↩ cost
      {𝕤, •Out "Cost after iteration "∾(•Repr i)∾": "∾(•Repr cost) }⍟print_cost @
    }⍟(0=100|i) @
    i +↩ 1
  }⍟num_iterations @
  parameters
}

#parameters ← TwoLayerModel train_x‿train_y‿layers_dims‿0.0075‿1‿1
parameters ← TwoLayerModel train_x‿train_y‿layers_dims‿0.0075‿2500‿1

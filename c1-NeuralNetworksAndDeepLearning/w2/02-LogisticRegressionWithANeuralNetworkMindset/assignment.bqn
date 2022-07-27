path_libreaddata ← "/Users/jra/prj/coursera/deeplearning/c1-NeuralNetworksAndDeepLearning/w2/02-LogisticRegressionWithANeuralNetworkMindset/libreaddata.dylib"
readTrainX ← path_libreaddata •FFI "a"‿"read_train_x"
readTrainY ← path_libreaddata •FFI "a"‿"read_train_y"
readTestX ← path_libreaddata •FFI "a"‿"read_test_x"
readTestY ← path_libreaddata •FFI "a"‿"read_test_y"
readTestListClasses ← path_libreaddata •FFI "a"‿"read_test_list_classes"

train_set_x_orig ← @ -˜ ReadTrainX ⟨⟩
train_set_y ← ReadTrainY ⟨⟩
test_set_x_orig ← @ -˜ ReadTestX ⟨⟩
test_set_y ← ReadTestY ⟨⟩
classes ← ReadTestListClasses ⟨⟩

index ← 10
•Out "y = "∾(•Repr index⊑train_set_y)∾", it's a '"∾((index⊑train_set_y)⊑classes)∾"' picture."
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
("img-train"∾(•Repr index)∾".ppm") ImageWrite index ⊏ train_set_x_orig

m_train ← ≠train_set_x_orig
m_test ← ≠test_set_x_orig
num_px ← 1⊑≢train_set_x_orig

•Out "Number of training examples: m_train = "∾•Repr m_train
•Out "Number of testing examples: m_test = "∾•Repr m_test
•Out "Height/Width of each image: num_px = "∾•Repr num_px
•Out "Each image is of size: "∾•Repr ≢⊏train_set_x_orig
•Out "train_set_x shape: "∾•Repr ≢train_set_x_orig
•Out "train_set_y shape: "∾•Repr ≢train_set_y
•Out "test_set_x shape: "∾•Repr ≢test_set_x_orig
•Out "test_set_y shape: "∾•Repr ≢test_set_y

train_set_x_flatten ← ⍉ (≠train_set_x_orig)‿∘ ⥊ train_set_x_orig
test_set_x_flatten ← ⍉ (≠test_set_x_orig)‿∘ ⥊ test_set_x_orig

•Out "train_set_x_flatten shape: "∾•Repr ≢train_set_x_flatten
•Out "train_set_y shape: "∾•Repr ≢train_set_y
•Out "test_set_x_flatten shape: "∾•Repr ≢test_set_x_flatten
•Out "test_set_y shape: "∾•Repr ≢test_set_y
•Out "sanity check after reshpaing: "∾•Repr 5↑⊏⎉1 train_set_x_flatten

train_set_x ← train_set_x_flatten ÷ 255
test_set_x ← test_set_x_flatten ÷ 255

Sigmoid ← { ÷1+⋆-𝕩 }

•Out "Sigmoid "∾(•Repr 0‿2)∾" = "∾•Repr Sigmoid 0‿2

InitializeWithZeros ← { (𝕩‿1⥊0)‿0 }

tdim ← 2
tw‿tb ← InitializeWithZeros tdim

•Out "tw = "∾•Repr tw
•Out "tb = "∾•Repr tb

numb ← •Import "numb.bqn"

Propagate ← {
  w‿b‿x‿y:
  m ← 1⊑≢y
  n ← ⊑≢w
  a ← Sigmoid b + (⍉w) numb.Dot x
  cost ← ⊑-÷⟜m+˝⍉(y×⋆⁼a)+(¬y)×⋆⁼¬a
  dw ← ÷⟜m x numb.Dot ⍉a-y
  db ← ⊑ ÷⟜m +˝⍉a-y
  ⟨dw‿db, cost⟩
}

w‿b‿x‿y ← ⟨∘‿1⥊1‿2, 2, >⟨1‿2‿¯1, 3‿4‿¯3.2⟩, 1‿∘⥊1‿0‿1⟩
grads‿cost ← Propagate w‿b‿x‿y

•Out "Propagate test:"
•Out "dw = "∾•Repr ⊑grads
•Out "db = "∾•Repr 1⊑grads
•Out "cost = "∾•Repr cost

Optimize ← {
  w‿b‿x‿y‿dw‿db‿costs‿num_iterations‿learning_rate‿print_cost:
  grads‿cost ← @‿@
  i ← 0
  {
    𝕤
    {
      grads‿cost ↩ Propagate w‿b‿x‿y
      dw‿db ↩ grads
      w -↩ learning_rate×dw
      b -↩ learning_rate×db
      {𝕤,costs ∾↩ 𝕩}⍟(0=100|i) cost
      {𝕤,•Out •Fmt 𝕩}⍟(print_cost ∧ 0=100|i) cost
      i +↩ 1
    }
  }⍟num_iterations @
  ⟨w, b⟩‿⟨dw, db⟩‿costs
}

dw‿db‿num_iterations‿learning_rate‿print_cost ← @‿@‿@‿@‿@
params ← @
costs ← ⟨⟩
w‿b‿x‿y‿dw‿db‿costs‿num_iterations‿learning_rate‿print_cost ↩ ⟨w, b, x, y, 0, 0, ⟨⟩, 100, 0.009, 0⟩
params‿grads‿costs ↩ Optimize w‿b‿x‿y‿dw‿db‿costs‿num_iterations‿learning_rate‿print_cost

•Out "Optimize test:"
•Out "w = "∾•Repr ⊏params
•Out "b = "∾•Repr 1⊏params
•Out "dw = "∾•Repr ⊑grads
•Out "db = "∾•Repr 1⊑grads

Predict ← {
  w‿b‿x:
  m ← 1⊑≢x
  y_prediction ← 1‿m⥊0
  w ↩ (⊑≢x)‿1⥊w
  a ← Sigmoid b + (⍉w) numb.Dot x
  y_prediction ↩ a>0.5
  y_prediction
}

w‿b‿x ↩ ⟨∘‿1⥊0.1124579‿0.23106775, ¯0.3, >⟨1.0‿¯1.1‿¯3.2, 1.2‿2.0‿0.1⟩⟩
•Out "predictions = "∾•Repr Predict w‿b‿x

Model ← {
  x_train‿y_train‿x_test‿y_test‿num_iterations‿learning_rate‿print_cost:
  w‿b ← InitializeWithZeros ⊑≢x_train
  dw‿db‿costs ← @‿@‿⟨⟩
  parameters‿grads ← @‿@
  parameters‿grads‿costs ↩ Optimize w‿b‿x_train‿y_train‿dw‿db‿costs‿num_iterations‿learning_rate‿print_cost

  # Retrieve parameters w and b from dictionary "parameters"
  w‿b ↩ parameters

  # Predict test/train set examples (≈ 2 lines of code)
  y_prediction_test ← Predict w‿b‿x_test
  y_prediction_train ← Predict w‿b‿x_train

  # Print train/test Errors
  •Out "train accuracy: "∾(•Repr ⊑100-100×(+´÷≠)˘|y_prediction_train-y_train)∾"%"
  •Out "test accuracy: "∾(•Repr ⊑100-100×(+´÷≠)˘|y_prediction_test-y_test)∾"%"

  ⟨
    costs
    y_prediction_test
    y_prediction_train
    w
    b
    learning_rate
    num_iterations
  ⟩
}

num_iterations‿learning_rate‿print_cost ↩ 2000‿0.005‿0
#num_iterations‿learning_rate‿print_cost ↩ 40‿0.005‿0
train_set_y ↩ 1‿∘⥊train_set_y
test_set_y ↩ 1‿∘⥊test_set_y
d ← Model train_set_x‿train_set_y‿test_set_x‿test_set_y‿num_iterations‿learning_rate‿print_cost
#•Out "b "∾•Fmt 4⊑d
#•Out "costs "∾•Fmt ⊑d
#•Out "costs "∾•Fmt ¯1⊑⊑d

index ↩ 25
("img-test"∾(•Repr index)∾".ppm") ImageWrite index ⊏ test_set_x_orig
•Show ≢test_set_y
•Show •Repr test_set_y
•Show ≢1⊑d
•Out "y = "∾(•Repr (0‿index)⊑test_set_y)∾", you predicted that it is a """∾(•Repr ((0‿index)⊑1⊑d)⊑classes)∾""" picture."

#costs_csv_lines ← "iter,cost"∾•Repr¨
costs ↩ ⊑d
•Show costs
•Show ≢costs
•Show ≠costs
iters ← 100×↕≠costs
•Show iters
•Show ≢iters
•Show ≠iters
•Show ⍉iters≍costs
ics ← ⍉iters≍costs
•Show {(•Repr 0⊑𝕩)∾","∾•Repr 1⊑𝕩}¨<˘ics


"costs.csv" •file.Lines ⟨"iters,cost"⟩∾{(•Repr 0⊑𝕩)∾","∾•Repr 1⊑𝕩}¨<˘ics
•SH "python"‿"chart_costs.py"

path_libmagickwand ← "/Users/jra/prj/coursera/deeplearning/c1-NeuralNetworksAndDeepLearning/w2/02-LogisticRegressionWithANeuralNetworkMindset/libmagickwand.dylib"
readResizeImage ← path_libmagickwand •FFI "&"‿"read_resize_image"‿"*i32:c8"‿"u32"‿"u32"‿"&f64"

Norm255 ← {
  mn ← ⌊´⥊𝕩
  mx ← ⌈´⥊𝕩
  0⌈¨255⌊¨⌊¨256×÷⟜(mx-mn)-⟜mn𝕩
}

modw ← 3⊑d
modb ← 4⊑d
•Out •Repr ≢modw
•Out •Repr modb

imgtemp ← 64‿64‿3⥊0

path_image ← "/Users/jra/Desktop/jra_photo.jpeg"∾@
hum1 ← Norm255 ReadResizeImage path_image‿64‿64‿imgtemp
("hum1.ppm") ImageWrite hum1
hy ← Sigmoid modb + (⍉modw) numb.Dot (64×64×3)‿1⥊hum1
•Out •Repr hy

path_image ↩ "/Users/jra/Desktop/cat1.jpeg"∾@
cat1 ← Norm255 ReadResizeImage path_image‿64‿64‿imgtemp
•Show •Repr 10↑⊏cat1
•Show •Repr ⌈´⥊cat1
•Show •Repr ⌊´⥊cat1
•Show •Repr (+´÷≠)⥊cat1
("cat1.ppm") ImageWrite cat1
c1y ← Sigmoid modb + (⍉modw) numb.Dot (64×64×3)‿1⥊cat1
•Out •Repr c1y

path_image ↩ "/Users/jra/Desktop/cat2.jpeg"∾@
cat2 ← Norm255 ReadResizeImage path_image‿64‿64‿imgtemp
("cat2.ppm") ImageWrite cat2
c2y ← Sigmoid modb + (⍉modw) numb.Dot (64×64×3)‿1⥊cat2
•Out •Repr c2y

learning_rates ← 10⋆-2+↕3
ds ← {Model train_set_x‿train_set_y‿test_set_x‿test_set_y‿1500‿𝕩‿0}¨learning_rates

iters ↩ ⋈100×↕≠⊑⊑ds
costs ↩ ⊑¨ds
•Show iters
•Show costs
ics ↩ ⍉>iters∾costs
•Show ≢ics
•Show ics
"costs_3lrs.csv" •file.Lines ⟨"iters,0.01,0.001,0.0001"⟩∾{(•Repr 0⊑𝕩)∾","∾(•Repr 1⊑𝕩)∾","∾(•Repr 2⊑𝕩)∾","∾(•Repr 3⊑𝕩)}¨<˘ics
•SH "python"‿"chart_costs_3lrs.py"

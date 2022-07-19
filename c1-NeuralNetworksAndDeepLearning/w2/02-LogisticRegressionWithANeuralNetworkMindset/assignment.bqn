path_libreaddata ← "libreaddata.dylib"

readTrainX ← path_libreaddata •FFI "a"‿"read_train_x"
readTrainY ← path_libreaddata •FFI "a"‿"read_train_y"
readTestX ← path_libreaddata •FFI "a"‿"read_test_x"
readTestY ← path_libreaddata •FFI "a"‿"read_test_y"

trainX ← @ -˜ ReadTrainX ⟨⟩
trainY ← ReadTrainY ⟨⟩
testX ← @ -˜ ReadTestX ⟨⟩
testY ← ReadTestY ⟨⟩

•Show ≢trainX
•Show 0‿0‿0‿0 ⊑ trainX
•Show 0‿0‿0‿1 ⊑ trainX
•Show 0‿0‿0‿2 ⊑ trainX

•Show ≢trainY
•Show 0 ⊑ trainY
•Show 1 ⊑ trainY
•Show 2 ⊑ trainY

•Show ≢testX
•Show 0‿0‿0‿0 ⊑ testX
•Show 0‿0‿0‿1 ⊑ testX
•Show 0‿0‿0‿2 ⊑ testX

•Show ≢testY
•Show 0 ⊑ testY
•Show 1 ⊑ testY
•Show 2 ⊑ testY

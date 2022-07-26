# basic bqn, general solution
#Dot ← +˝∘×⎉1‿∞

# optimized bqn
#Dot ← {
#  1≡⊑≠𝕨? ≍+´ (⥊𝕨) × <˘ 𝕩;
#  1≡1⊑≢𝕩? (≠𝕨)‿1⥊ (+´(⥊𝕩)⊸×)˘𝕨;
#  𝕨 +˝∘×⎉1‿∞ 𝕩
#}

cblasRowMajor←101
cblasColMajor←102

cblasNoTrans←111
cblasTrans←112
cblasConjTrans←113
cblasConjNoTrans←114
cblasTransOptions←⟨
  cblasNoTrans
  cblasTrans
⟩

#path_blas_lib ← "/Users/jra/.local/share/virtualenvs/deeplearning-TAU9TGrL/lib/python3.10/site-packages/numpy/.dylibs/libopenblas64_.0.dylib"  # no cblas
#path_blas_lib ← "/usr/local/lib/libgslcblas.dylib"  # gsl (not parallelized)
#path_blas_lib ← "/usr/local/Cellar/openblas/0.3.20/lib/libopenblasp-r0.3.20.dylib"  # brew package (good)
path_blas_lib ← "/Users/jra/pks/openblas/lib/libopenblas_haswellp-r0.3.17.dylib"  # compiled locally (best)

p ← @
#                                   lay tA  tB  m   n   k   α   A    lda B    ldb β   C    ldc
p ↩ (1-˜+`×¬)∘=⟜⊏⊸⊔ " & cblas_dgemm u32 u32 u32 i32 i32 i32 f64 *f64 i32 *f64 i32 f64 &f64 i32"
gemm ← path_blas_lib •FFI p
#                                   lay tA  m   n   α   A    lda x    incx β  y    incy
p ↩ (1-˜+`×¬)∘=⟜⊏⊸⊔ " & cblas_dgemv u32 u32 i32 i32 f64 *f64 i32 *f64 i32 f64 &f64 i32"
gemv ← path_blas_lib •FFI p

Cblas_gemm_ab ← {
  a‿b:
  m‿k←≢a ⋄ kx‿n←≢b ⋄ !k≡kx
  c←m‿n⥊0
  Gemm cblasRowMajor‿cblasNoTrans‿cblasNoTrans‿m‿n‿k‿1‿a‿k‿b‿n‿0‿c‿n
}

Cblas_gemv_axpby ← {
  trans‿a‿x‿beta‿y:
  ctrans ← trans⊑cblasTransOptions  # 0-no trans, 1-trans
  m‿n←≢a ⋄ nx←⊑≢x ⋄ ny←⊑≢y
  !nx=trans⊑n‿m ⋄ !ny=trans⊑m‿n
  incx ← 1
  incy ← 1
  alpha ← 1
  Gemv cblasRowMajor‿ctrans‿m‿n‿alpha‿a‿n‿x‿incx‿beta‿y‿incy
}

Cblas_gemv_ax ← {
  trans‿a‿x:
  ctrans ← trans⊑cblasTransOptions  # 0-no trans, 1-trans
  m‿n←≢a
  y ← (trans⊑m‿n)‿1⥊0
  nx←⊑≢x ⋄ ny←⊑≢y
  !nx=trans⊑n‿m ⋄ !ny=trans⊑m‿n
  incx ← 1
  incy ← 1
  alpha ← 1
  beta ← 0
  Gemv cblasRowMajor‿ctrans‿m‿n‿alpha‿a‿n‿x‿incx‿beta‿y‿incy
}

Dot ⇐ {
  1≡⊑≠𝕨? ⍉ Cblas_gemv_ax 1‿𝕩‿(⍉𝕨);
  1≡1⊑≢𝕩? Cblas_gemv_ax 0‿𝕨‿𝕩;
  Cblas_gemm_ab 𝕨‿𝕩
}

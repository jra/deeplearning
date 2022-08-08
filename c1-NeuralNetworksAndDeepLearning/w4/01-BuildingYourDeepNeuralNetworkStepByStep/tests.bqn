nn ← •Import "nn.bqn"

rand ← •MakeRand 0

n1 ← nn.Normal rand‿⟨0, 1⟩
•Show n1

n2 ← nn.Normal rand‿⟨0, 1⟩
•Show n2

n3 ← nn.Normal rand
•Show n3

n4 ← 3 nn.Normal rand‿⟨0, 1⟩
•Show n4

n5 ← 3 nn.Normal rand
•Show n5

n6 ← 2‿3 nn.Normal rand
•Show n6

ps ← nn.InitializeParameters 3‿2‿1
•Out "w1 = "∾•Repr ps.w1
•Out "b1 = "∾•Repr ps.b1
•Out "w2 = "∾•Repr ps.w2
•Out "b2 = "∾•Repr ps.b2

! 2‿3 ≡ ≢ps.w1
! ⟨2⟩ ≡ ≢ps.b1
! ∧´=⟜0¨⥊ps.b1
! 1‿2 ≡ ≢ps.w2
! ⟨1⟩ ≡ ≢ps.b2
! ∧´=⟜0¨⥊ps.b2

ws‿bs ← nn.InitializeParametersDeep 5‿4‿3

! 2=≠ws
! 2=≠bs
! 4‿5≡≢0⊑ws
! (≠=≠∘⍷)0⊑ws  # all unique
! (4⥊0)≡0⊑bs
! 3‿4≡≢1⊑ws
! (≠=≠∘⍷)1⊑ws  # all unique
! (3⥊0)≡1⊑bs

a ← [[ 1.62434536, -0.61175641]
     [-0.52817175, -1.07296862]
     [ 0.86540763, -2.3015387 ]]
w ← [[1.74481176, -0.7612069,   0.3190391]]
b ← ⟨-0.24937038⟩
z ← [[ 3.26295337, -1.23429987]]

zz‿⟨aa‿ww‿bb⟩ ← nn.LinearForward a‿w‿b

Close ← {
  d 𝕊 vs:
  (1=≠∘⍷) { (10⋆d)÷˜⌊𝕩×(10⋆d) }¨ vs
}

! (≢z)≡≢zz

! ∧´ ⥊ z { 7 Close 𝕨‿𝕩 }¨ zz

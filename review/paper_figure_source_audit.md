# 論文 Figure 1–7 と実装の差分監査

- **対象論文**: `review/Entropic_Partial_Optimal_Transport_and_Gromov__Wasserstein_Distances_between_Gaussian_Mixtrues.pdf`（Yachimura–Zou, Section 7 “Numerical Experiments”, pp. 29–36）
- **対象コード**: `docs/notebook/{figure1_2_3,figure4,figure5,PMGW_corrected}.ipynb`、`src/pgot/`、`src/computation/engine.cpp`、`vendor/PGW_Metric/lib/gromov.py`
- **リポジトリ**: `main`（issue #6 の P0 のうち Figure 4・Figure 5 分を適用した状態。submodule `vendor/PGW_Metric` = `8a9002e`）
- **実行環境**: Python 3.13.15 / POT 0.9.7 / NumPy 2.4.6 / SciPy 1.18.0 / scikit-learn 1.9.0 / Matplotlib 3.11.0
- **併読**: `review/code_audit_figures_1_7.pdf`（2026-08-03）。本書 §5 で各指摘と照合する。
- **対応 issue**: [#6 \[Paper reproducibility\] Figure 1–7 の生成コードを論文の定義・実験条件に揃える](https://github.com/yachimura-lab/EPGOT-PMGW/issues/6)。本書は同 issue が「詳細は `review/paper_figure_source_audit.md` に記録した」と参照する詳細記録である。

---

## 1. 結論サマリ

issue #6 の P0 により Figures 4–7 は論文の定義・実験条件に揃えた。残る主な差分は **Figure 2 の生成セルが存在しない**ことと、Figure 1・3 の体裁である。

| Fig | 論文の設定 | 実装の状態 | 判定 |
|---|---|---|---|
| 1 | λ=0.3, ε=0.01 | 数値は完全一致（mass=0.9000）。線の着色規則のみ掲載図と異なる | **B**（体裁のみ） |
| 2 | λ∈{0,0.125,0.25,0.375,0.5}, ε=0.01 | **生成セルなし**。ただし数値は現行コードで完全に再現する | **B**（描画欠落・数値は健全） |
| 3 | λ 7値 × ε∈{0.01,0.11,0.21} | cell 6 が一致。1 panel で Sinkhorn 非収束 | **B** |
| 4 | (4.18) の劣確率測度 | 質量 `Z_λ` を保持し全 panel 共通 density level。(4.17) は閉形式 | **B** |
| 5 | (a) λ=0.25 / (b) λ=0.5 | 既知 GMM + (3.1) の dummy-point EPOT。composite layout のみ未対応 | **B** |
| 6 | 半径 4・z=0、λ=0.01、barycentric→nearest | geometry・λ 換算・6/7 共有 fit・barycentric 対応を統一。PGW/pMGW とも `300/310` | **B** |
| 7 | barycentric projection map | (6.7) を中心化して解き、Figure 6 と同一 solve から生成 | **B** |

A = 論文記述と数値的に不一致（要対応）、B = 数値は妥当で体裁・欠落の問題。

---

## 2. penalty 規約の確定（最重要）

論文の penalty は (3.1) と (5.4) で **`−2λ`** の形で入る。

- (3.1): 拡張コスト `c̃_ij = c_ij − 2λ`（実–実ブロック）、拡張周辺 `ã=(a,1)`, `b̃=(b,1)`
- (5.4): 歪み項 `(|W₂(µi,µk)^q − W₂(νj,νl)^q|^p − 2λ) ω_ij ω_kl + 2λ`

リポジトリには **EPOT 系 3 実装 + PGW 系 1 実装** があり、λ の解釈が 1 つだけ異なる。

| 実装 | コスト変形 | 定式化 | 論文 λ との関係 | 使用箇所 |
|---|---|---|---|---|
| `pgot.entropic_partial_ot` | `M − 2*Lambda` | dummy node + `ot.sinkhorn`、拡張全体に entropy | **`Lambda = λ`**（(3.1) と厳密一致） | Fig 1, 2, 3, 5 |
| `cpp_engine::entropic_partial_ot`<br>(`engine.cpp`) | `M -= 2.0*Lambda` | 同上（Eigen 実装） | **`Lambda = λ`** | Fig 4 |
| `pgot.partial_wasserstein_lagrange_entropic` | `M − Lambda` | 不等式制約 capped scaling、実ブロックのみに entropy | **`Lambda = 2λ`** | **deprecated**。`forward/` の legacy notebook のみ |
| `lib.gromov.partial_gromov_ver1`<br>(`tensor_dot_param`) | `f1(r)=r²−2Λ` ⇒ 歪み `|C1−C2|²−2Λ` | Frank–Wolfe + dummy node | **`Lambda = λ`**（(5.4) と一致） | Fig 6(b)(d), 7(b) |

EPOT を要する図はすべて `entropic_partial_ot`（および同一規約の C++ 実装）を通すため、**notebook が渡す `Lambda` は常に論文の λ である**。規約の異なる `partial_wasserstein_lagrange_entropic` は deprecated で、経路上に残っていない（§3.5）。

規約の違いによる差（ε=0.03、GA/GB は §7.1 の厳密パラメータ）。`Lambda = 2λ` を渡したときだけ lagrange 版が (3.1) の mass に追随する:

| 論文 λ | (3.1) dummy-node の mass | lagrange(`Lambda`=λ) | lagrange(`Lambda`=2λ) |
|---|---|---|---|
| 0.125 | 0.3703 | 0.0661 | **0.5815** |
| 0.25 | 0.8876 | 0.5815 | **0.9022** |
| 0.5 | 0.9937 | 0.9022 | **1.0000** |

> **cost normalization**: 論文 (7.1) は `Ĉ_G = C_G / max(C_G)`。EPOT 系は `M / np.max(M)` で一致。GW 系は `C1, C2` を **両者の共通最大値** で割る（`pMGW2_coup` は `max(C1.max(), C2.max())`、point-level も `max(M1, M2)`）。よって λ は常に正規化スケール上の値であり、論文の「λ ∈ (0, 0.5] は (7.1) のスケールに対する相対値」という記述と整合する。

---

## 3. 図ごとの差分

### 3.1 Figure 1 — §7.1.1（λ=0.3, ε=0.01）

**数値は完全一致。** `entropic_partial_ot` で λ=0.3, ε=0.01 を解くと matched mass = **0.9000**、掲載図タイトルの `mass = 0.90` と一致する。描画は cell 5（`Lambda_list = np.arange(0.3, 0.325, 0.125)` → `[0.3]` の 1 panel）。

差分は体裁のみ:

| 項目 | 論文掲載図 | cell 5 |
|---|---|---|
| 線の色 | coupling weight を viridis で共通着色 | `tab20` で「成分対ごと」に色分け |
| colorbar | 横 colorbar（0.00–0.40） | なし |
| 凡例・ラベル | なし | `w_{1-i,2-j}=…` 凡例、`1-0`/`2-1` 成分ラベル |

- cell 4 は同じ計算を `ot.sinkhorn` で **インラインに重複実装**している（`entropic_partial_ot` と同一処理）。図は出力せず、検算用と思われる。
- cell 3 が保存する `GMMs.eps`（GMM 単体図）は論文に存在しない補助図。

### 3.2 Figure 2 — §7.1.2（λ∈{0,0.125,0.25,0.375,0.5}, ε=0.01）

**この 1×5 を生成するセルは存在しない。** 現存する sweep は cell 6（3×7）、cell 7（6×7）、cell 8（5×5）で、いずれも λ の刻みが論文と異なる（cell 8 は λ∈{0.1,…,0.5} で λ=0 を欠く）。

ただし **数値そのものは現行コードで完全に再現する**:

| λ | 0 | 0.125 | 0.25 | 0.375 | 0.5 |
|---|---|---|---|---|---|
| 論文 Figure 2 の mass | 0.00 | 0.42 | 0.90 | 0.95 | 1.00 |
| `entropic_partial_ot` 実測 | 0.0000 | 0.4196 | 0.9000 | 0.9489 | 0.9994 |

したがって Figure 2 は**アルゴリズム上の不一致ではなく、描画セルの欠落**である。cell 5 の描画ロジックを λ リストでループさせれば再現できる。

### 3.3 Figure 3 — §7.1.3（λ 7値 × ε 3値）

cell 6 が生成元であり、**パラメータは論文と厳密一致**する。

配置は `plt.subplots(len(reg_list), len(Lambda_list))` = **3 行（ε）× 7 列（λ）**。

- `np.arange(0.125, 0.5625, 0.0625)` = {0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5} ✓
- `np.arange(0.01, 0.31, 0.1)` = {0.01, 0.11, 0.21} ✓
- viridis + 横 colorbar も掲載図と一致 ✓

残る問題:

1. **Sinkhorn 非収束**: 保存出力に `Sinkhorn did not converge` が記録されている（ε=0.11 の λ=0.3125 付近）。確定図をこの出力のまま使うべきではない。
2. **colorbar の上限が意味的に不整合**: `norm = Normalize(vmin=0, vmax=np.max(M))` の `M` は**正規化済みコスト行列**であり `np.max(M) == 1.0`。つまり colorbar の [0,1] は coupling weight の尺度ではなく、コスト行列の最大値がたまたま 1 であることの副産物。実測の weight 最大は約 0.45。共通最大値または理論上限 0.5 を明示的に指定すべき。
3. **出力名の衝突**: cell 7 と cell 8 がともに `fig_lambda_epsilon_sweep.eps` に保存し、cell 8 が cell 7 を上書きする。いずれも論文外。
4. cell 7 は `M_1` を計算しながら solver には cell 6 由来の `M` を渡している（`M_1` は未使用）。セル実行順への暗黙依存。

### 3.4 Figure 4 — §7.1.4（displacement interpolation (4.18)）

**パラメータは一致**: `configs` = (Λ=0.25, ε=0.03), (Λ=0.25, ε=0.1), (Λ=0.5, ε=0.03) ✓、`t_list = np.linspace(0,1,5)` = {0, 0.25, 0.5, 0.75, 1.0} ✓。

notebook は `entropic_partial_displacement_interpolation` で (4.18) を評価する。coupling entry は**閾値も再正規化もかけず**そのまま重みに使うので、描かれる密度の積分は各行の `Z_λ` に一致する（数値積分で確認、誤差 <2e-4）:

| 行 | (ε, λ) | `Z_λ` |
|---|---|---|
| 1 | (0.03, 0.25) | 0.8875742 |
| 2 | (0.10, 0.25) | 0.6807441 |
| 3 | (0.03, 0.50) | 0.9937141 |

密度 level は全 15 panel で共通（`4.001e-04 .. 4.001e-01`）。**再正規化の除去と共通 level は両方必要**である。`display_gmm` が panel ごとに `z_max` を決めていた旧実装では、劣確率測度を渡しても質量差は見えなかった。

component 補間は (4.17) の閉形式
`µ^t_kl = ((1−t)Id + t T_kl)_# µ^0_k`（平均 `(1−t)m_0k + t m_1l`、共分散 `B Σ_0k Bᵀ`、`B = (1−t)I + t A_kl`）
を直接使う。反復 W₂ barycenter（20 iter）との差は共分散で 3.1e-07 であり、閉形式の方が厳密。

> **`cpp_engine` への依存を解消した。** 旧実装は `cpp_engine` の `GaussianW2` / `GaussianBarycenterW2` / `entropic_partial_ot` を使っていた。`.so` は gitignore されるためビルドしないと Figure 4 を再実行できず、本監査でも当初は再実行できなかった。現在は `pgot` の実装のみを使う。
>
> なお両実装は完全には一致しない。`GaussianW2` の差は 5.7e-14 だが、**EPOT は (λ=0.5, ε=0.03) で coupling が最大 7.6e-04、matched mass が 0.9929587（C++）と 0.9937141（Python）で食い違う**。C++ 側の Sinkhorn が `‖u − u_prev‖_∞ < 1e-7` で打ち切るのに対し POT は周辺分布の残差で判定するためで、C++ 版の方が収束が甘い。Python 版に統一したことで Figure 4 の第 3 行と Figure 5(b) が同じ `Z_λ = 0.9937141` を報告するようになった。

### 3.5 Figure 5 — §7.1.5（barycentric projection map (6.2)）

**一致している点**: 各 GMM から 2000 サンプル ✓、ε=0.03 ✓、`ts = np.arange(0.2, 1.01, 0.2)` = {0.2, 0.4, 0.6, 0.8, 1.0} ✓。

#### (1) solver と λ は (3.1) に一致している

notebook は §7.1 の**既知の GMM `GA` / `GB` をそのまま** `entropic_partial_barycentric_map` に渡す。sample は (6.2) の評価点と散布図の重ね描きにのみ使い、再 fit は行わない。component parameters と cost matrix は両 λ で同一である。solver は `entropic_partial_ot`、すなわち (3.1) の dummy-point EPOT で、`Lambda` は論文の λ をそのまま受け取る:

| panel | 表示 λ | solver への入力 | matched mass |
|---|---|---|---|
| Figure 5(a) | 0.25 | 0.25 | **0.8875742** |
| Figure 5(b) | 0.5 | 0.5 | **0.9937141** |

notebook は `log=True` で受け取った matched mass を出力するため、掲載値・solver 入力・出力の三者が突き合わせられる。

> **`1834ea2` 以前の状態**: `compute_T_X_to_Z_C` は `partial_wasserstein_lagrange_entropic`（`M − Lambda`、実ブロックのみ entropy、capped scaling）を呼んでおり、これは (3.1) とは別問題である。penalty scale が半分になるため cell 5 は `Lambda = 2 * Lambda` で補正していたが cell 4 は補正しておらず、**Figure 5(a) は実際には λ=0.125 相当（mass 0.5917881、論文値 0.8847946）** を描いていた。solver 差し替えと `2 * Lambda` 廃止は、片方だけ行うと (b) が 1.0000 → 0.8989 と論文値から離れるため同一コミットで行った。
>
> 是正による写像点の移動は平均 0.10・最大 0.17（データの広がりは約 20）にとどまり、**図の外観はほぼ変わらない**。`T_b` が行正規化した coupling にしか依存せず、その形が 2.3e-2 しか変わらないためである。是正の効果は外観ではなく数値の正しさにある。

#### (2) 残る差分

- **`compute_T_X_to_Z`（もう一方の公開関数）は (6.2) ではない**。分母が `np.sum(a * p)`、すなわち**全混合密度 `Σ_k a_k p_k(x)`** であり、(6.2) の matched density `Σ_kl ω_kl p_k(x)` ではない。これは (6.1) の balanced 版の分母。`figure5.ipynb` は使っていないが `__init__.py` から公開されている。docstring に注意書きを入れてあるが、是正または非公開化が要る。
- cell 6 は cell 5 の `Z`, `Lambda`, `xmin/xmax` に依存する。セル実行順への依存。
- 掲載図の (a)/(b) 2 段組を保存するセルがない（§4）。

### 3.6 Figures 6–7 — §7.2（point-cloud matching）

#### (1) geometry を論文どおりに再構成した

`PMGW_corrected.ipynb` は半径 4 の ring から source/target を**独立に** 300 点ずつ、別分布（球内一様）の noise を 10 点生成する。実測:

| 量 | 論文 | 実測 |
|---|---|---|
| source ring 半径 | 4 | 4.005（z = −0.010, sd 0.191） |
| target ring 半径 | 4（別の高さ） | 4.007（z = 10.002） |
| noise | 10 点 | 中心 (9.97, 0.03, 10.06)、半径 0.3 |

論文が明示していない **target ring の高さ `h = 10` と noise の分布**（中心 `(2.5·radius, 0, h)`、半径 0.3 の球内一様）は notebook 冒頭で定数として固定し、論文にも記載する必要がある。データ種別ごとに独立した `default_rng` を使う。

旧実装（`42ee1ad`）は source を 2.5 倍、target を 2 倍して平行移動していたため半径が 9.987 / 7.990、target ring は source 標本の affine 像、source の z 分散は厳密に 0 だった。**再構成により z 分散が 0 でなくなり、共分散が正定値になった**ので、論文 §4 の仮定も満たすようになった。

#### (2) penalty 換算規約

point-level と component-level は**異なる定数でコストを正規化する**ため、同じ数値の λ が同じ意味を持たない。そこで λ を、その表現における **balanced 最適解の平均歪み** `D_match` を単位として測る:

```
Lambda = lambda_tilde × gw_mean_distortion(C1, C2, G_balanced)
```

`lambda_tilde` は無次元で両手法が共有する。論文の λ = 0.01 を point-level の値として固定すると `lambda_tilde = 0.9199` となり、

| 表現 | `D_match` | solver への `Lambda` |
|---|---|---|
| point-level（二乗距離/max） | 0.010871 | **0.010000**（= 論文値） |
| component-level（W₂²/max, 6/7 fit） | 0.014403 | **0.013249** |

この換算で **PGW・pMGW とも matched mass が厳密に `300/310 = 0.9677419`** になる。4 種の seed で確認済み。

`D_match` は square loss の恒等式で `O(n²)` で計算でき、`gw_mean_distortion` / `partial_gw_penalty` として公開している。

#### (3) 比較条件の統一

- source 6 / target 7 成分を**一度だけ** fit し、`components, weights, pairwise W2 matrices` を balanced/partial の両 solver へ渡す（`MGW2_coup` / `pMGW2_coup` の `mu=`, `nu=`, `C1=`, `C2=`）
- balanced MGW の alignment を **full mixture mean で中心化**してから解く。(6.7) は中心化した成分上で定義されており、balanced では (6.6) の matched mean が全混合平均に一致する
- 手法ごとに **1 回だけ solve** し、Figure 7 は raw barycentric map、Figure 6 は同じ map の nearest-neighbor 版として作る（`return_both=True`）
- raw GW/PGW も `argmax` ではなく **barycentric projection + nearest-target** に統一。partial では質量 0 の source 行を描かない

結果:

| 手法 | matched mass | 描画した source 点 | noise へ着地 |
|---|---|---|---|
| GW2 | 1.0000000 | 300/300 | **10** |
| PGW2 | 0.9677419 | 294/300 | 0 |
| MGW2 | 1.0000000 | 300/300 | 0 |
| pMGW2 | 0.9677419 | 300/300 | 0 |

balanced GW だけが noise へ 10 本の対応を引き、partial GW は 6 点を未マッチとして描かない。論文の主張どおりである。

> **論文の記述と合わない点**: 論文は balanced *mixture* GW についても「outlier が transport plan に影響し spurious correspondence を生む」と述べるが、**MGW2 は noise へ 1 本も対応を引かない**。7 成分 fit では noise が独立成分になり、barycentric 平均がその寄与を薄めるためである。MGW2 の劣化は noise への対応ではなく **ring 対応の交差**として現れる。§7.2 の記述はこの観察に合わせて直す必要がある。

#### (4) alignment の発散（[#15](https://github.com/yachimura-lab/EPGOT-PMGW/issues/15)）

Stiefel 上の projected gradient descent が既定ステップ幅で目的関数を悪化させていた。**Figures 6–7 の実データでも発生**しており、balanced MGW で 1.829721 → 19.239575、partial pMGW で 0.006050 → 14.348159 と、初期値より大幅に悪い `P` が使われていた。

各ステップを **目的関数が減るまで backtracking** するよう修正した。あわせて `proj_gradient_descent` が記録していた loss を、共分散項のみから (6.7) の完全な目的関数（`alignment_objective`）へ直した。修正後は両者とも単調で、この data では初期値 `P0 = proj_stiefel(Σ γ_kl m_k m_lᵀ)` が既に最適であり `P0` がそのまま返る。

#### (5) 残る差分

- `ot.gromov_wasserstein(..., epsilon=5e-3)` の削除（P1）。POT 0.9.7 では `**kwargs` に吸収され効果は厳密に 0
- 計算量の主張（P1）。coupling 次元は 300×310 → 6×7 に縮むが、component 側は GMM fit のコストを伴い N=300 では end-to-end の高速化を示せない

## 4. ノートブック ↔ 掲載図 の対応

| 掲載図 | ノートブック | セル | 保存名 |
|---|---|---|---|
| Figure 1 | `figure1_2_3.ipynb` | 5 | `fig_single_lambda_colored_wlabel.eps` |
| Figure 2 | — | **なし** | — |
| Figure 3 | `figure1_2_3.ipynb` | 6 | `fig_lambda_reg_sweep_optimized.eps` |
| Figure 4 | `figure4.ipynb` | 4 | `comparison_high_res.{eps,pdf}` |
| Figure 5(a) | `figure5.ipynb` | 4 | `point_reg_0.03_lambda_0.25.eps`（実際は λ≈0.125） |
| Figure 5(b) | `figure5.ipynb` | 5+6 | `point_reg_0.03_lambda_0.5.eps` |
| Figure 6(a) | `PMGW_corrected.ipynb` | 14 | `gw_coupling.eps` |
| Figure 6(b) | `PMGW_corrected.ipynb` | 15 | `pgw_coupling.eps` |
| Figure 6(c) | `PMGW_corrected.ipynb` | 12 | `mgw_coupling.eps` |
| Figure 6(d) | `PMGW_corrected.ipynb` | 10 | `pmgw_coupling.eps` |
| Figure 7(a) | `PMGW_corrected.ipynb` | 13 | `mgw_bary.eps` |
| Figure 7(b) | `PMGW_corrected.ipynb` | 11 | `pmgw_bary.eps` |

（論文外の出力: `GMMs.eps`, `fig_lambda_epsilon_sweep.eps`）

**掲載図の composite layout を直接保存するセルが存在しない。** 論文の Figure 5 は (a)/(b) の 2 段組、Figure 6 は (a)–(d) の 4 panel、Figure 7 は (a)/(b) の 2 panel だが、ノートブックはいずれも **1 panel（または 1 行）ずつ別ファイルに保存**しており、掲載図の組版は notebook の外で行われている。したがって「保存出力 → 掲載図」の対応がファイル単位で追跡できない。

---

## 5. `code_audit_figures_1_7.pdf` の指摘との照合

### 一覧表の指摘（同 PDF p.1）

| # | 前回の指摘 | 本監査の判定 | 補足 |
|---|---|---|---|
| 1 | Fig 1: 数値一致・線色/凡例が掲載図と異なる | ✅ **支持** | mass=0.9000 を実測確認 |
| 2 | Fig 2: 1×5 sweep の生成セルなし | ✅ **支持（軽減）** | 描画欠落のみ。数値は 5 点とも完全再現（§3.2） |
| 3 | Fig 3: 概ね一致・1 panel で Sinkhorn 非収束 | ✅ **支持** | colorbar 上限の由来を特定（§3.3-2） |
| 4 | Fig 4: 総和 1 へ再正規化し (4.18) を描いていない | ✅ **支持 → 解消** | 再正規化と panel ごとの level 調整を両方撤去（§3.4） |
| 5 | Fig 5: λ=0.5 と表示しつつ solver に 2λ=1.0 | ❌ **panel の帰属が誤り**（解消済み） | 齟齬は (b) ではなく **(a)** 側にあった。`1834ea2` で solver を (3.1) に統一し両 panel を再生成（§3.5-1） |
| 6 | Fig 6: データ・λ・成分数・point 対応が不一致 | ✅ **支持（強化）** | λ=0.01 が効かないのは geometry 由来。論文どおりの半径 4 に直せば PGW は λ=0.01 で `300/310`（§3.6-2） |
| 7 | Fig 6 と 7 が別 solve で同一 coupling 由来と保証されない | ⚠️ **格下げ** | 実測で完全一致。P0 → 保守性の改善項目（§3.6-7） |

### 本文の個別指摘

| 箇所 | 前回の指摘 | 判定 |
|---|---|---|
| §1 Fig 1 | tab20 着色・凡例・成分ラベルの除去、共通 Normalize + viridis + 横 colorbar | ✅ 支持 |
| §1 Fig 1 | 見出し「Figure 2 – the coupling at a single λ」を Figure 1 に直す | ⚠️ **現行版に該当なし**（markdown は revert 済み） |
| §1 Fig 2 | λ∈{0,0.125,0.25,0.375,0.5}, ε=0.01 の専用セルを追加 | ✅ 支持 |
| §1 Fig 3 | 非収束 panel の再計算、colorbar の明示、cell 13/15 の整理 | ✅ 支持（現行 cell 7/8） |
| §1 冒頭 | markdown の entropy 式が (3.1) と異なる | ⚠️ **現行版に該当なし**。なお `entropic_partial_ot` の実装は (3.1) どおり拡張行列全体に entropy を課しており正しい |
| §2 Fig 4 | `W[i,j] > 1e-5` の閾値は (4.18) にない | ✅ **解消**。閾値を撤去し 6 対すべてを保持（§3.4） |
| §2 Fig 4 | 反復 barycenter ではなく (4.17) の閉形式を使う | ✅ **解消**（§3.4） |
| §3 Fig 5 | `Lambda=Lambda` に直して Fig 5(b) を再生成 | ❌ **単独では逆効果**。倍化のみ外すと (b) は 1.0000→0.8989 と論文値 0.9932 から離れる。`1834ea2` では solver を `entropic_partial_ot` に差し替えたうえで倍化を外し、(a)(b) 両方を再生成した（§3.5-1） |
| §3 Fig 5 | λ ごとに GMM を fit し直す構造 | ✅ **解消**。notebook は既知 GMM を直接使い、両 λ で同一の component parameters / cost を共有する（§3.5-1） |
| §4.1 | source 半径 10 / z 分散 0、target 半径 8・高さ 25、noise (20,0,25)・半径 0.6、target は affine 変換 | ✅ **全項目を実測で確認**（§3.6-1） |
| §4.2 | λ_PGW=0.04, λ_pMGW=0.10 が論文の 0.01 と不一致 | ✅ **支持（強化）**。両者とも係数規約は (5.4) と同一だが正規化スケールが異なる。λ=0.01 が効かないのは geometry 由来（§3.6-2） |
| §4.2 | balanced 6/6 vs partial 6/7 で条件が不公平 | ✅ 支持 |
| §4.3 | Fig 6(a),(b) は argmax であり barycentric projection ではない | ✅ 支持。描画閾値も 2 panel で不一致 |
| §4.4 | Fig 6 と 7 を 1 回の solve から作る | ⚠️ 格下げ（上記 #7） |
| §4.5 | `epsilon=5e-3` は標準 GW の entropy 正則化指定ではない | ✅ **支持（確定）**。`**kwargs` に吸収され効果は厳密に 0 |
| §4.6 | 小規模例で end-to-end 高速化は示されていない | ✅ **支持（定量化）**（§3.6-5） |

### 共通の再現性修正（同 PDF §5）

| # | 前回の指摘 | 現状 |
|---|---|---|
| 1 | solver source を同梱し commit hash を含める | ✅ **解消**。`src/pgot`, `src/computation`, submodule `8a9002e` を確認。ただし fresh clone では `vendor/PGW_Metric` が空で `import pgot` が失敗する（README/CLAUDE.md に sparse checkout 手順あり） |
| 2 | 環境を lock file に固定 | ✅ 概ね解消（`uv.lock`、POT 0.9.7）。`partial_gromov_ver1` の deprecation warning は未対応 |
| 3 | fit と solve を分離し比較内で再利用 | ❌ 未対応（`pMGW2_coup` / `compute_T_X_to_Z_C` が内部で fit）。ただし seed 固定により実害は出ていない。EPOT solver 自体は `entropic_partial_ot` に一本化済み |
| 4 | 乱数を局所化 | ⚠️ 部分対応。`set_reproducible()` が package RNG + global `np.random` + `random` + torch を seed するが、`PMGW_corrected.ipynb` は global `np.random` を直接使いセル実行順に依存 |
| 5 | 1 figure = 1 生成関数 = 1 出力名 | ❌ 未対応（`fig_lambda_epsilon_sweep.eps` が cell 7/8 で衝突） |
| 6 | EPS の透明度を避ける | ❌ 未対応。4 ノートブックすべてで transparency warning |

---

## 6. 修正の優先順位

issue #6 の方針「**原則として論文を正とし、source code / notebook / docs を修正する**」に従う。以下は issue の P0/P1 タスクに対応する。

### 解消済み

- **Figure 5 の solver と λ** — `compute_T_X_to_Z_C` を `entropic_partial_ot` に差し替え、同時に cell 5 の `Lambda = 2 * Lambda` を廃止して (a)(b) を再生成。`partial_wasserstein_lagrange_entropic` は deprecate。あわせて (6.2) の分子だけに掛かっていた `gamma>1e-10` の閾値を除去。
- **Figure 4 の (4.18) 描画** — 再正規化と `W>1e-5` 閾値を撤去して質量 `Z_λ` を保持し、全 panel 共通 density level に変更。(4.17) は閉形式に置き換え、`cpp_engine` 依存を解消（§3.4）。
- **Figures 6–7** — geometry を半径 4 の独立標本へ再構成、λ 換算規約を `D_match` 基準で固定、6/7 成分の fit と cost を共有、balanced alignment を中心化、全手法を barycentric projection + nearest に統一、手法ごとに 1 回だけ solve（§3.6）。あわせて alignment の発散を backtracking で解消（[#15](https://github.com/yachimura-lab/EPGOT-PMGW/issues/15)）。
- **Figure 5 の fit 共有** — §7.1.5 どおり既知 GMM を使う `entropic_partial_barycentric_map` を追加し、notebook をそれに切り替え。両 λ が同一の component parameters / cost を共有する。`compute_T_X_to_Z_C` は fit する薄い wrapper として残した（§3.5-1）。

### P0（数学的意味・比較条件）


### P1（Figure と主張）

1. **Figure 2** — 1×5 の描画セルを追加（数値は現行コードで再現済み、§3.2）。**Figure 1** — weight の連続 colormap + 横 colorbar に戻し、label/legend を外す。
2. **Figure 3** — 非収束 panel の再計算と residual の保存、colorbar 上限の明示、cell 7/8 の出力名衝突の解消（§3.3）。
3. **point-level GW** — 無効な `epsilon=5e-3` を削除（図の再生成は不要、§3.6-5）。
4. **composite layout** — Figure 5 の (a)/(b) 2 段組を notebook から保存する（Figures 6–7 は対応済み、§4）。
5. **計算量の主張** — coupling dimension 削減に限定するか、N を増やした solver-only benchmark を追加（§3.6-5）。

### その他

6. `compute_T_X_to_Z` の分母が (6.2) ではなく全混合密度である点を是正するか、公開 API から外す（§3.5-2）。
7. 再生成により掲載 PDF と外観が変わるため、issue #6 の方針 7・8 に従い差分を [#8](https://github.com/yachimura-lab/EPGOT-PMGW/issues/8) に記録する。

---

## 7. 本監査の限界

- **`cpp_engine` の数値挙動は Figure 4 の範囲でのみ検証した。** `GaussianW2`（差 5.7e-14）、`GaussianBarycenterW2`（(4.17) 閉形式との差 3.1e-07）、`entropic_partial_ot`（(λ=0.5, ε=0.03) で coupling 最大 7.6e-04 の差）を比較している。legacy の `src/algorithm/*.py` が使う経路は未検証。
- 掲載図との比較は**論文 PDF から抽出したテキスト・軸目盛り・caption**に基づく。図の画素単位の照合は行っていない。
- Figure 1/3 の「線の色が掲載図と異なる」は、前回監査の記述と論文本文（"The width and color of each line indicate the corresponding coupling weight"）に依拠している。
- **§3.6-2 の「論文どおりの geometry」は本書による再構成である。** 論文は target ring の高さと noise の分布を明示していないため、高さ h ∈ {8, 10, 12}、noise は中心 `(2.5·radius, 0, h)`・半径 0.3 の球内一様として検証した。matched mass `300/310` は h に依らず同一だったが、noise の配置を大きく変えれば λ の閾値は動きうる。実装時には論文側にも高さと noise 分布を明記する必要がある。
- 実測値はすべて `set_reproducible()`（seed=42）下で `42ee1ad` / POT 0.9.7 において取得したもの。

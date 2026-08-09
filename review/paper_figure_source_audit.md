# 論文 Figure 1–7 と実装の差分監査

- **対象論文**: `review/Entropic_Partial_Optimal_Transport_and_Gromov__Wasserstein_Distances_between_Gaussian_Mixtrues.pdf`（Yachimura–Zou, Section 7 “Numerical Experiments”, pp. 29–36）
- **対象コード**: `docs/notebook/{figure1_2_3,figure4,figure5,PMGW_corrected}.ipynb`、`src/pgot/`、`src/computation/engine.cpp`、`vendor/PGW_Metric/lib/gromov.py`
- **リポジトリ**: `42ee1ad`（submodule `vendor/PGW_Metric` = `8a9002e`）
- **実行環境**: Python 3.13.15 / POT 0.9.7 / NumPy 2.4.6 / SciPy 1.18.0 / scikit-learn 1.9.0 / Matplotlib 3.11.0
- **併読**: `review/code_audit_figures_1_7.pdf`（2026-08-03）。本書 §5 で各指摘と照合する。
- **対応 issue**: [#6 \[Paper reproducibility\] Figure 1–7 の生成コードを論文の定義・実験条件に揃える](https://github.com/yachimura-lab/EPGOT-PMGW/issues/6)。本書は同 issue が「詳細は `review/paper_figure_source_audit.md` に記録した」と参照する詳細記録である。

---

## 1. 結論サマリ

保存済み出力は概ね掲載図に対応するが、**Figure 2 の生成セルが存在せず、Figure 5(a) と Figure 6–7 は論文記述と数値的に不一致**である。

| Fig | 論文の設定 | 実装の状態 | 判定 |
|---|---|---|---|
| 1 | λ=0.3, ε=0.01 | 数値は完全一致（mass=0.9000）。線の着色規則のみ掲載図と異なる | **B**（体裁のみ） |
| 2 | λ∈{0,0.125,0.25,0.375,0.5}, ε=0.01 | **生成セルなし**。ただし数値は現行コードで完全に再現する | **B**（描画欠落・数値は健全） |
| 3 | λ 7値 × ε∈{0.01,0.11,0.21} | cell 6 が一致。1 panel で Sinkhorn 非収束 | **B** |
| 4 | (4.18) の劣確率測度 | 総和 1 へ再正規化 **＋** panel ごとに density level 自動調整 | **A**（描いている対象が違う） |
| 5 | (a) λ=0.25 / (b) λ=0.5 | **齟齬は (a) にある**（論文値との差 0.293、(b) は 0.007）。さらに Fig 1–4 と別 solver | **A** |
| 6 | 半径 4・z=0、λ=0.01、barycentric→nearest | データ・λ・成分数・点対応がすべて不一致 | **A** |
| 7 | barycentric projection map | (a) の alignment が (6.7) と異なる | **A** |

A = 論文記述と数値的に不一致（要対応）、B = 数値は妥当で体裁・欠落の問題。

---

## 2. penalty 規約の確定（最重要）

論文の penalty は (3.1) と (5.4) で **`−2λ`** の形で入る。

- (3.1): 拡張コスト `c̃_ij = c_ij − 2λ`（実–実ブロック）、拡張周辺 `ã=(a,1)`, `b̃=(b,1)`
- (5.4): 歪み項 `(|W₂(µi,µk)^q − W₂(νj,νl)^q|^p − 2λ) ω_ij ω_kl + 2λ`

リポジトリには **EPOT 系 3 実装 + PGW 系 1 実装** があり、λ の解釈が 1 つだけ異なる。

| 実装 | コスト変形 | 定式化 | 論文 λ との関係 | 使用箇所 |
|---|---|---|---|---|
| `pgot.entropic_partial_ot`<br>(`partial_ot.py:37`) | `M − 2*Lambda` | dummy node + `ot.sinkhorn`、拡張全体に entropy | **`Lambda = λ`**（(3.1) と厳密一致） | Fig 1, 2, 3 |
| `cpp_engine::entropic_partial_ot`<br>(`engine.cpp`) | `M -= 2.0*Lambda` | 同上（Eigen 実装） | **`Lambda = λ`** | Fig 4 |
| `pgot.partial_wasserstein_lagrange_entropic`<br>(`partial_ot.py:113`) | `M − Lambda` | 不等式制約 capped scaling、実ブロックのみに entropy | **`Lambda = 2λ`** | Fig 5 |
| `lib.gromov.partial_gromov_ver1`<br>(`tensor_dot_param`) | `f1(r)=r²−2Λ` ⇒ 歪み `|C1−C2|²−2Λ` | Frank–Wolfe + dummy node | **`Lambda = λ`**（(5.4) と一致） | Fig 6(b)(d), 7(b) |

**したがって現行コードにおける `figure5.ipynb` の `Lambda = 2 * Lambda` は誤りではなく、論文 λ へ換算するための正しい変換である。**（§3.5、§5 の指摘 5 を参照。）ただし solver を `entropic_partial_ot` に統一すれば倍化は二重適用になるため、**その差し替えとセットで廃止する**のが正しい修正である（§6 P0-1）。

数値による裏付け（ε=0.03、GA/GB は §7.1 の厳密パラメータ）:

| 論文 λ | (3.1) dummy-node の mass | lagrange(`Lambda`=λ) | lagrange(`Lambda`=2λ) |
|---|---|---|---|
| 0.125 | 0.3703 | 0.0661 | **0.5815** |
| 0.25 | 0.8876 | 0.5815 | **0.9022** |
| 0.5 | 0.9937 | 0.9022 | **1.0000** |

`Lambda = 2λ` を渡したときだけ (3.1) の mass に追随する。

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

**パラメータは一致**: `configs` = (Λ=0.25, ε=0.03), (Λ=0.25, ε=0.1), (Λ=0.5, ε=0.03) ✓、`t_list = np.linspace(0,1,5)` = {0, 0.25, 0.5, 0.75, 1.0} ✓。`cpp_engine::entropic_partial_ot` は Python 版と同一規約（`M -= 2.0*Lambda` + dummy node）なので Λ はそのまま論文 λ である。

**描いている対象が (4.18) ではない。** cell 3 の末尾:

```python
return GaussianMixture(np.array(new_w) / np.sum(new_w), new_m, new_c)
```

(4.18) の `µ^t_{ε,λ} = Σ_kl ω^{ε,λ}_kl µ^t_kl` は総質量 `Z_λ < 1` の**劣確率測度**だが、ここで総和 1 に再正規化しているため実際に描かれるのは `µ^t_{ε,λ} / Z_λ` である。

**再正規化を外すだけでは不十分である。** `display_gmm`（cell 2）が

```python
z_max = max(Z.max(), 1e-8)
levels = np.logspace(np.log10(z_max * 1e-3), np.log10(z_max), 7)
```

と **panel ごとに density level を自前で決めている**ため、仮に劣確率測度を渡しても質量差は可視化されない。**再正規化の除去と全 panel 共通 level の指定は両方必要**である。

その他:

- `if W[i, j] > 1e-5` の打ち切りは (4.18) にない。成分対は 6 個しかないので閾値は不要。
- 2 Gaussian の補間に `GaussianBarycenterW2(..., 20)`（反復 W₂ barycenter, 20 iter）を使っている。(4.17) は `µ^t_kl = ((1−t)Id + t T_kl)_# µ^0_k` という**閉形式**であり、そちらの方が厳密かつ再現的。

### 3.5 Figure 5 — §7.1.5（barycentric projection map (6.2)）

**一致している点**: 各 GMM から 2000 サンプル ✓、ε=0.03 ✓、`ts = np.arange(0.2, 1.01, 0.2)` = {0.2, 0.4, 0.6, 0.8, 1.0} ✓。

#### (1) λ の不一致は **(a)** にある

§2 の通り `partial_wasserstein_lagrange_entropic` は `M − Lambda` なので、論文 λ を渡すには 2 倍が必要。

| セル | コード | solver が受ける値 | **実際の論文 λ** | 出力 mass | caption/ファイル名 |
|---|---|---|---|---|---|
| cell 4 | `Lambda=Lambda`（=0.25） | 0.25 | **0.125** | 0.5918 | λ=0.25 と主張 ❌ |
| cell 5 | `Lambda=2*Lambda`（=0.5→1.0） | 1.0 | **0.5** | 1.0000 | λ=0.5 ✅ |

`figure5.ipynb` と同じ条件（2000 サンプル → 2/3 成分 fit、ε=0.03）で測ると:

| 論文 λ | dummy EPOT (3.1) | lagrange(`Lambda`=λ) | lagrange(`Lambda`=2λ) |
|---|---|---|---|
| 0.25 | 0.8847946 | **0.5917881** | 0.8988555 |
| 0.5 | 0.9932484 | 0.8988555 | **1.0000000** |

太字が notebook の保存出力と一致する値である（cell 4 = 0.5917881、cell 5 = 1.0000000）。論文値との差は:

- **cell 5（=Figure 5(b)）**: 1.0000000 vs 0.9932484 → **0.0068**
- **cell 4（=Figure 5(a)）**: 0.5917881 vs 0.8847946 → **0.2930**

**齟齬は (a) にある。** (b) の `2 * Lambda` は現行 solver に対する正しい換算である。

> **修正方針（issue #6 の P0 に従う）**: `compute_T_X_to_Z_C` の solver を `entropic_partial_ot`（`M − 2λ` + dummy node）に差し替え、`partial_wasserstein_lagrange_entropic` は rename/deprecate する。**その差し替えを行った時点で cell 5 の `Lambda = 2 * Lambda` は二重適用になるため廃止する**（issue の P0 タスク「`Lambda=2 * Lambda` を廃止」はこの意味で正しい）。両 panel とも素の λ を渡し、**(a) (b) の両方を再生成する**。
>
> 現行 solver を残したまま `2*Lambda` だけを外すと (b) が 1.0000 → 0.8989 と論文値から離れるため、**solver 差し替えと `2*Lambda` 廃止は必ずセットで行うこと**。

#### (2) Figure 5 だけ別の EPOT solver を使っている

§7.1.5 は `T^{ε,λ}_b` を (6.2) で定義し、`ω^{ε,λ}` は **(4.4)=(3.1) の minimizer の実–実ブロック**としている。しかし `figure5.ipynb` が呼ぶ `compute_T_X_to_Z_C` は `partial_wasserstein_lagrange_entropic`、すなわち

- dummy node ではなく **不等式制約（γ1≤a, γᵀ1≤b）への capped scaling**
- entropy を**拡張行列全体ではなく実ブロックのみ**に課す

という別問題を解いている。λ を換算しても両者の解は一致しない（§2 の表: λ=0.25 で 0.8876 vs 0.9022）。**Figure 5 は (4.4) から生成されていない。** 論文どおりにするなら `entropic_partial_ot` を使うか、§7.1.5 に使用した定式化を明記する必要がある。

#### (3) その他

- `partial_wasserstein_lagrange_entropic` に `print(gamma)` / `print(np.sum(gamma))` のデバッグ出力が残っており、ノートブックの保存出力に coupling 行列が混入している（`partial_ot.py:128-129`）。
- `compute_T_X_to_Z_C` は分子で `gamma[k,l] > 1e-10` の対のみ加算するが、**分母は全対を加算**する（`partial_ot.py:313, 333`）。(6.2) からの微小なずれ。数値的には無視できる。
- **`compute_T_X_to_Z`（もう一方の公開関数）は (6.2) ではない**。分母が `np.sum(a * p)`、すなわち**全混合密度 `Σ_k a_k p_k(x)`** であり、(6.2) の matched density `Σ_kl ω_kl p_k(x)` ではない。これは (6.1) の balanced 版の分母。`figure5.ipynb` は使っていないが、`__init__.py` から公開され docstring も "the original Figure 5 …" とあるため誤用の危険がある。
- λ ごとに GMM を fit し直すが、`random_state=DEFAULT_SEED` 固定かつ `X, Y` 同一なので **fit 結果は完全に同一**。前回監査が懸念した比較条件の不公平は現行コードでは生じない（冗長計算のみ）。
- cell 6 は cell 5 の `Z`, `Lambda`, `xmin/xmax` に依存する。セル実行順への依存。

### 3.6 Figures 6–7 — §7.2（point-cloud matching）

#### (1) データ生成が論文記述と一致しない

論文: 「source P1 は半径 4 の円周上（z=0）に平均を置く 6 成分 GMM から 300 点。target P2 は対応する 6 成分 GMM を**別の高さ**に置いて 300 点、+ noise 10 点」。

`PMGW_corrected.ipynb` cell 2–7 を再実行した実測値:

| 量 | 論文 | 実測 |
|---|---|---|
| source ring 半径 | 4 | **9.987** |
| source z | 0 | 0（**分散が厳密に 0**） |
| target ring 半径 | 4（別の高さ） | **7.990** |
| target ring 高さ | — | **25.03** |
| noise 中心 | — | **(20.11, −0.01, 24.95)** |
| noise 半径 | — | **0.6**（設計値。実標本の中心からの最大距離は 0.53） |

原因は cell 7 の `P1 = 2.5*P1_3d` と `P23 = 2*np.vstack([P2, P3]) + (0,0,5)`。加えて:

- **target ring は独立標本ではない**。`P23[:300, :2] == 0.8 * P1[:, :2]` が **1.8e-15** の精度で成立する（同一標本 `P_1` の affine 変換）。
- cell 3 が生成する 2D 標本 `P1_2d` は**捨てられており**、cell 4/7 では行数だけが再利用される。
- **source の z 分散が厳密に 0** のため共分散が z 方向に特異。6 成分 full-covariance GMM の fit が通るのは sklearn の `reg_covar=1e-6` のおかげであり、論文 §4 の「全共分散が正定値」という仮定を満たさない。

> 掲載 Figure 6 の軸範囲（x: −10…20、y: −15…15、z: 0…30）は**この scale 済みデータと整合する**。つまり掲載図は論文本文が記述する geometry ではなく、このパイプラインから生成されている。geometry を論文どおりに直せば**掲載図の外観は必ず変わる**ため、再生成と差分記録が要る。

#### (2) λ が論文と不一致で、成立するかは geometry に依存する

論文は partial 系で λ=0.01 と記す。実装は:

- point-level PGW（cell 15）: `gromov.partial_gromov_ver1(..., Lambda=0.04)`
- pMGW（cell 10/11）: `pMGW2_coup(..., Lambda=0.10)` → 同じ `partial_gromov_ver1` へ

§2 の通り `partial_gromov_ver1` の `Lambda` は**すでに論文 (5.4) の λ そのもの**なので、規約の違いでは説明できない。

**λ=0.01 が成立するかは geometry に依存する。** 現行の scale 済み geometry と、論文どおりに直した geometry（半径 4 の ring から source 300 点、高さ h の対応 ring から**独立に** 300 点、別分布の noise 10 点）で matched mass を比較した:

| geometry | 手法 | λ=0.01 | λ = コードの値 |
|---|---|---|---|
| 現行（半径 10 / 8、affine 変換） | point-level PGW | **0.0000** | 0.9677（λ=0.04） |
| 現行 | pMGW | `ValueError: Z_lambda = 0.0` | 0.9677（λ=0.10） |
| 論文どおり（半径 4、独立標本） | point-level PGW | **0.9677419** ✅ | 0.9677419（λ=0.04） |
| 論文どおり | pMGW | **0.0000000** ❌ | 0.9677419（λ=0.10） |

- **point-level PGW** は、geometry を論文に戻せば λ=0.01 で `300/310 = 0.9677419` を輸送する（h = 8 / 10 / 12 のいずれでも同じ）。**geometry と penalty は一体で直す必要がある。**
- **pMGW では、geometry を直しても λ=0.01 では matched mass が 0 のままである。** 閾値を走査すると:

| 正規化スケール | λ=0.005 | 0.01 | 0.02 | 0.04 以上 |
|---|---|---|---|---|
| point-level（二乗距離/max） | 0.9677419 | 0.9677419 | 0.9677419 | 0.9677419 |
| component-level（W₂²/max, 6/7 fit） | — | **0.0000000** | 0.9677419 | 0.9677419 |

閾値を超えれば両者とも厳密に `300/310` で頭打ちになる（target 側の 300 点分の質量が上限）ので、**λ の正確な値ではなく「閾値を超えているか」だけが効く**。閾値が 2〜4 倍ずれるのは正規化スケールが違うためで、point-level は二乗ユークリッド距離/最大値、component-level は W₂²/最大値である。

**したがって、論文の単一の λ=0.01 を両 solver にそのまま書くことはできない。** 論文の λ から各 solver input への換算規約を数式で固定し、論文とコードの双方に記載する必要がある。

#### (3) GMM 成分数が非対称

- balanced MGW（cell 12/13）: `MGW2_coup(n_components=6)` → **source 6 / target 6**
- partial pMGW（cell 10/11）: **source 6 / target 7**

Figure 6(c) と 6(d) の差には「partial 化」と「target 成分数 6→7」の両方が混入する。

#### (4) balanced 側の alignment が (6.7) と異なる

論文 (6.7) は**中心化した成分**上で Stiefel 最適化を行う（(6.6) の matched means で中心化、balanced なら全混合平均に一致）。

- `pMGW2_coup` は `matched_statistics` → `gmm_transform(mu, b=-m0)` で中心化してから最適化する ✅
- `MGW2_coup` は **中心化しない `mu`, `nu` のまま** `proj_gradient_descent` を回し、中心化は後段の `T_mean` 内でのみ行う ❌（`mgw.py:144-151`, `gaussian.py:185-196`）

実測すると alignment 行列は有意に異なる:

```
||P_uncentered − P_centered||_F = 0.7618
diag(P_uncentered) = [0.9917, 0.9921, 0.9996]
diag(P_centered)   = [0.8569, 0.9818, 0.8702]
```

したがって **Figure 6(c) / 7(a) は (6.7) を実装していない**。これは (3) とは独立した第 2 の比較条件の非対称性である。

#### (5) Figure 6(a),(b) は barycentric projection ではない

論文は Figure 6 を「barycentric projection map が誘導する nearest-point correspondence」と説明するが、point-level の 2 panel は coupling 行の argmax を直接描く。

```python
j = np.argmax(G[i, :])          # cell 14 (GW), cell 15 (PGW)
```

これは maximum-coupling assignment であり barycentric projection とは一般に異なる。実際に barycentric projection + `NearestNeighbors` を使っているのは (c), (d) のみ。

また 2 panel で描画閾値が異なる: GW は `G[i,j] > 1e-5`、PGW は `G_partial[i,j] > 0.1/n1`（= 3.33e-4）。

#### (6) point-level GW の `epsilon` は無効

```python
G = ot.gromov_wasserstein(C1, C23, p, q, 'square_loss', epsilon=5e-3)   # cell 14
```

POT 0.9.7 の `ot.gromov_wasserstein` のシグネチャに `epsilon` は存在せず、`**kwargs` に吸収される。実測で

```
max|G(epsilon=5e-3) − G(no epsilon)| = 0.000e+00
```

**完全に無効**。結果として cell 14 は (2.5) の非正則化 balanced GW を計算しており、これは論文の意図と一致する。誤解を招く引数を削除するだけでよい（図の再生成は不要）。

#### (7) Figure 6 と 7 は別 solve だが、結果は一致する

`points=True` / `points=False` で `pMGW2_coup` / `MGW2_coup` を 2 回呼ぶため、GMM fit・coupling・alignment が再計算される。ただし `random_state=DEFAULT_SEED` 固定かつ両 solver が決定的なため、**実測では完全に一致する**:

```
pMGW: idx(別々) == idx(1回solve)? True ; max|map差| = 0.00e+00
MGW : idx(別々) == idx(1回solve)? True ; max|map差| = 0.00e+00
```

**Figure 6(c)(d) と 7(a)(b) は同一の map から作られている。** 両関数は `return_both=True` を既に備えているので、それに切り替えるのは実行時間と堅牢性の改善であって、図の正しさの修正ではない。

#### (8) 計算量の主張

coupling 次元は 300×310 → 6×7 に縮小し、**solver 単体では大幅に速い**。しかし component 側は GMM fit のコストを伴う:

| 段階 | 時間 |
|---|---|
| point-level GW solver (300×310) | 101.9 ms |
| point-level PGW solver (300×310) | 92.4 ms |
| component pMGW solver (6×7) | **0.4 ms**（約 230×） |
| component 側が必要とする GMM fit | 96.6 ms |
| **component 経路 合計** | **97.0 ms**（vs point-level PGW 92.4 ms） |

N=300 のこの例では **end-to-end の高速化は示せていない**。「coupling problem の次元を削減する」という主張に限定するか、N を増やした solver-only の scaling benchmark を追加するのが妥当。

---

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
| 4 | Fig 4: 総和 1 へ再正規化し (4.18) を描いていない | ✅ **支持（強化）** | 再正規化に加え panel ごとの level 自動調整も要修正（§3.4） |
| 5 | Fig 5: λ=0.5 と表示しつつ solver に 2λ=1.0 | ❌ **要修正** | 現行コードでは `2*Lambda` が**正しい換算**。不一致は **(a)** 側（§3.5-1） |
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
| §2 Fig 4 | `W[i,j] > 1e-5` の閾値は (4.18) にない | ✅ 支持 |
| §2 Fig 4 | 反復 barycenter ではなく (4.17) の閉形式を使う | ✅ 支持 |
| §3 Fig 5 | `Lambda=Lambda` に直して Fig 5(b) を再生成 | ❌ **単独では逆効果**。現行 solver のまま倍化を外すと (b) は 1.0000→0.8989 と論文値 0.9932 から離れる。solver を `entropic_partial_ot` に差し替えたうえで倍化を外し、(a)(b) 両方を再生成する（§6 P0-1） |
| §3 Fig 5 | λ ごとに GMM を fit し直す構造 | ⚠️ **事実だが影響なし**。`random_state` 固定で fit は同一（冗長計算のみ） |
| §4.1 | source 半径 10 / z 分散 0、target 半径 8・高さ 25、noise (20,0,25)・半径 0.6、target は affine 変換 | ✅ **全項目を実測で確認**（§3.6-1） |
| §4.2 | λ_PGW=0.04, λ_pMGW=0.10 が論文の 0.01 と不一致 | ✅ **支持（強化）**。両者とも係数規約は (5.4) と同一だが正規化スケールが異なる。λ=0.01 が効かないのは geometry 由来（§3.6-2） |
| §4.2 | balanced 6/6 vs partial 6/7 で条件が不公平 | ✅ 支持 |
| §4.3 | Fig 6(a),(b) は argmax であり barycentric projection ではない | ✅ 支持。描画閾値も 2 panel で不一致 |
| §4.4 | Fig 6 と 7 を 1 回の solve から作る | ⚠️ 格下げ（上記 #7） |
| §4.5 | `epsilon=5e-3` は標準 GW の entropy 正則化指定ではない | ✅ **支持（確定）**。`**kwargs` に吸収され効果は厳密に 0 |
| §4.6 | 小規模例で end-to-end 高速化は示されていない | ✅ **支持（定量化）**（§3.6-8） |

### 共通の再現性修正（同 PDF §5）

| # | 前回の指摘 | 現状 |
|---|---|---|
| 1 | solver source を同梱し commit hash を含める | ✅ **解消**。`src/pgot`, `src/computation`, submodule `8a9002e` を確認。ただし fresh clone では `vendor/PGW_Metric` が空で `import pgot` が失敗する（README/CLAUDE.md に sparse checkout 手順あり） |
| 2 | 環境を lock file に固定 | ✅ 概ね解消（`uv.lock`、POT 0.9.7）。`partial_gromov_ver1` の deprecation warning は未対応 |
| 3 | fit と solve を分離し比較内で再利用 | ❌ 未対応（`pMGW2_coup` / `compute_T_X_to_Z_C` が内部で fit）。ただし seed 固定により実害は出ていない |
| 4 | 乱数を局所化 | ⚠️ 部分対応。`set_reproducible()` が package RNG + global `np.random` + `random` + torch を seed するが、`PMGW_corrected.ipynb` は global `np.random` を直接使いセル実行順に依存 |
| 5 | 1 figure = 1 生成関数 = 1 出力名 | ❌ 未対応（`fig_lambda_epsilon_sweep.eps` が cell 7/8 で衝突） |
| 6 | EPS の透明度を避ける | ❌ 未対応。4 ノートブックすべてで transparency warning |

---

## 6. 修正の優先順位

issue #6 の方針「**原則として論文を正とし、source code / notebook / docs を修正する**」に従う。以下は issue の P0/P1 タスクに対応する。

### P0（数学的意味・比較条件）

1. **Figure 5** — `compute_T_X_to_Z_C` の solver を `entropic_partial_ot` に差し替え、**同時に** cell 5 の `Lambda = 2 * Lambda` を廃止する（片方だけ行わないこと、§3.5-1）。(a) (b) とも再生成。`partial_wasserstein_lagrange_entropic` は rename/deprecate。
2. **Figures 6–7 の geometry** — 半径 4 の ring から source/target を**独立に** 300 点ずつ、別分布の noise 10 点。これにより point-level PGW は論文どおり λ=0.01 で `300/310` を達成する（§3.6-2）。
3. **penalty 換算規約の明示** — pMGW は component-level の正規化スケールでは λ=0.01 で matched mass 0 になるため、**論文の λ から各 solver input への換算を数式で固定**し、論文側にもその換算を記載する。閾値を超えれば両者とも `300/310` で頭打ちになる（§3.6-2）。
4. **Figures 6–7 の比較条件** — source 6 / target 7 成分を 1 回だけ fit して MGW/pMGW で共有。balanced MGW の alignment を full-mean centered mixtures で計算（§3.6-4）。raw GW/PGW を barycentric projection + nearest-target に統一し、unmatched row は描かない（§3.6-5）。手法ごとに 1 回だけ solve し、Figure 7 = raw map、Figure 6 = その nearest-neighbor 版とする（§3.6-7）。
5. **Figure 4** — 再正規化と `W>1e-5` 閾値を外して質量 `Z_λ` を保持し、**かつ**全 panel 共通 density level を使う（§3.4）。Gaussian 補間は (4.17) の閉形式に置き換える。

### P1（Figure と主張）

6. **Figure 2** — 1×5 の描画セルを追加（数値は現行コードで再現済み、§3.2）。**Figure 1** — weight の連続 colormap + 横 colorbar に戻し、label/legend を外す。
7. **Figure 3** — 非収束 panel の再計算と residual の保存、colorbar 上限の明示、cell 7/8 の出力名衝突の解消（§3.3）。
8. **point-level GW** — 無効な `epsilon=5e-3` を削除（図の再生成は不要、§3.6-6）。
9. **composite layout** — Figures 5–7 の掲載図と同じ組版を notebook から直接保存する（§4）。
10. **計算量の主張** — coupling dimension 削減に限定するか、N を増やした solver-only benchmark を追加（§3.6-8）。

### その他

11. `compute_T_X_to_Z` の分母が (6.2) ではなく全混合密度である点を是正するか、公開 API から外す（§3.5-3）。
12. 再生成により掲載 PDF と外観が変わるため、issue #6 の方針 7・8 に従い差分を [#8](https://github.com/yachimura-lab/EPGOT-PMGW/issues/8) に記録する。

---

## 7. 本監査の限界

- **Figure 4 は再実行できていない**。`cpp_engine` の `.so` は gitignore されており、本監査ではビルドしていない。`engine.cpp` のソース読解により penalty 規約が Python 版と一致することは確認済みだが、`GaussianBarycenterW2` の数値挙動は未検証。
- 掲載図との比較は**論文 PDF から抽出したテキスト・軸目盛り・caption**に基づく。図の画素単位の照合は行っていない。
- Figure 1/3 の「線の色が掲載図と異なる」は、前回監査の記述と論文本文（"The width and color of each line indicate the corresponding coupling weight"）に依拠している。
- **§3.6-2 の「論文どおりの geometry」は本書による再構成である。** 論文は target ring の高さと noise の分布を明示していないため、高さ h ∈ {8, 10, 12}、noise は中心 `(2.5·radius, 0, h)`・半径 0.3 の球内一様として検証した。matched mass `300/310` は h に依らず同一だったが、noise の配置を大きく変えれば λ の閾値は動きうる。実装時には論文側にも高さと noise 分布を明記する必要がある。
- 実測値はすべて `set_reproducible()`（seed=42）下で `42ee1ad` / POT 0.9.7 において取得したもの。

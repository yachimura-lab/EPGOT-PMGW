# 論文 Figure 1–7 と実装の最終監査

## 1. 対象と結論

- 対象: 論文 Section 7 の Figures 1–7
- canonical code: `src/pgot/paper_figures.py`
- canonical notebooks: `docs/notebook/{figure1_2_3,figure4,figure5,PMGW_corrected}.ipynb`
- parameter manifest: `docs/notebook/figure_manifest.json`
- provenance: `docs/notebook/metadata/figure{1..7}.json`
- 比較画像: `review/figures/{90c1098,after-p0,after-issue6}/`
- vendored PGW solver: `vendor/PGW_Metric` commit `8a9002e14d6b17decf35e603e9a9e42aa8465e04`

issue #6 のコード・notebook・docs・tests に関する受け入れ条件は完了した。Figures 1–7 は、論文パラメータ、solver 入力、入力 data/GMM、coupling mass/residual、runtime、source revision、solver/API、coupling/output hash まで sidecar から追跡できる。

論文本文の編集可能 source は repository にないため、Figures 6–7 の未記載条件・penalty 換算・balanced MGW の観測修正は [#18](https://github.com/yachimura-lab/EPGOT-PMGW/issues/18) に分離した。

## 2. penalty 規約

### 2.1 Figures 1–5: EPOT (3.1)

canonical primitive は `pgot.entropic_partial_ot` のみである。実–実 block の cost は `M - 2*Lambda`、extended marginals は `(a, 1)` / `(b, 1)`、entropy は dummy を含む extended coupling 全体へ掛かる。

したがって API の `Lambda` は論文の `lambda` と同じであり、2 倍・1/2 倍の換算はない。

旧 `partial_wasserstein_lagrange_entropic` は `M - Lambda` と capped scaling を使う別問題なので deprecated のまま legacy 再現用に残した。canonical notebook は参照しない。

### 2.2 Figures 6–7: PGW/pMGW (5.4)

PGW と pMGW の係数規約自体はともに `-2*Lambda` だが、point cost と component W2 cost の normalization scale が異なる。共通の無次元値 `lambda_tilde` を使い、各表現の balanced optimum に対して

```text
Lambda_solver
  = lambda_tilde
  * gw_mean_distortion(C1, C2, balanced_coupling)
```

とする。論文の point-level `lambda=0.01` を固定すると canonical run は次になる。

| 表現 | solver input | matched mass |
|---|---:|---:|
| point-level PGW | 0.0100000000 | 0.9677419355 |
| component-level pMGW | 0.0132491708 | 0.9677419355 |

## 3. Figure ごとの最終状態

| Fig | canonical output | 最終状態 |
|---|---|---|
| 1 | `outputs/figure1.pdf` | `lambda=0.3`, `epsilon=0.01`。coupling weight の continuous colormap + horizontal colorbar。component label / categorical pair colors / legend は除去 |
| 2 | `outputs/figure2.pdf` | `epsilon=0.01`, `lambda=[0,0.125,0.25,0.375,0.5]` の正確な 1×5 sweep |
| 3 | `outputs/figure3.pdf` | 3 epsilon rows × 7 lambda columns。colorbar は全 coupling の最大 weight（約 0.45）を明示。全 panel の residual と停止条件を保存 |
| 4 | `outputs/figure4.pdf` | (4.18) の sub-probability measure。threshold/再正規化なし、全 panel 共通 density level、(4.17) の閉形式 |
| 5 | `outputs/figure5.pdf` | 既知 GMM、dummy-point EPOT、matched-density map (6.2)、2×5 composite。両 lambda で同じ GMM/cost |
| 6 | `outputs/figure6.pdf` | 半径 4 の独立 ring + noise 10 点。全4手法を barycentric projection + nearest target に統一した 2×2 composite |
| 7 | `outputs/figure7.pdf` | Figure 6 と同じ MGW2/pMGW2 solve の raw maps。1×2 composite |

PDF は notebook 実行時に `docs/notebook/outputs/` へ生成される（repository の `*.pdf` ignore 規約に従い commit しない）。同じ Figure は notebook の embedded PNG と `review/figures/after-issue6/` に保存されている。

### 3.1 Figure 1

- paper parameter: `lambda=0.3`, `epsilon=0.01`
- matched mass: `0.9000`
- coupling の全正値を shared `Normalize` + `viridis` で表示
- horizontal colorbar の label は `coupling weight omega_ij^(epsilon,lambda)`
- 掲載図にない component label、pair legend、categorical colors はない

### 3.2 Figure 2

canonical 1×5 grid を新設した。高精度 solve 後の mass（4 桁丸め）は次のとおり。

| lambda | 0 | 0.125 | 0.25 | 0.375 | 0.5 |
|---:|---:|---:|---:|---:|---:|
| mass | 0.0000 | 0.4196 | 0.9000 | 0.9489 | 1.0000 |

`lambda=0.5` の旧 `0.9994` は 1000 iteration で停止した未収束値だった。厳密な extended marginal へ refinement すると `0.9999755` で、4 桁表示は `1.0000` になる。

### 3.3 Figure 3

旧 Figure 3 は `epsilon=0.01, lambda=0.5` で Sinkhorn が非収束だった。dummy cost と real cost が近いと matrix scaling の収束が極端に遅くなるため、canonical `entropic_partial_ot` は次を行う。

1. POT Sinkhorn を実行し、generic warning ではなく extended row/column residual を直接計測する
2. residual が宣言 tolerance を超えた小規模問題だけ、同じ strictly-convex entropic primal を SciPy SLSQP で refinement する
3. primary/fallback iteration、最終 residual、runtime、solver 名を返す

全21 panel は warning なしで `max(row_residual, column_residual) <= 1e-9` を満たす。colorbar の上限は cost maximum 1.0 ではなく、全 panel の coupling weight maximum である。

### 3.4 Figure 4

coupling は各 `(lambda, epsilon)` row につき1回だけ solve し、5つの `t` で再利用する。重みは全6 entry をそのまま使い、threshold と和1への再正規化はない。

| row | (lambda, epsilon) | Z_lambda |
|---:|---:|---:|
| 1 | (0.25, 0.03) | 0.8875742 |
| 2 | (0.25, 0.10) | 0.6807441 |
| 3 | (0.50, 0.03) | 0.9937141 |

回帰 test は density の数値積分が coupling sum と 1e-3 以内で一致することを確認する。全15 panel が同じ interpolation density levels を使う。Gaussian interpolation は反復 barycenter でなく論文 (4.17) の閉形式である。

### 3.5 Figure 5

Section 7.1 の既知 GMM を直接使い、2000 samples は map の評価点・scatter 表示だけに使う。source/target は用途別 local RNG（seed 52/53）で生成する。

| panel | paper lambda | solver input | epsilon | Z_lambda |
|---|---:|---:|---:|---:|
| (a) | 0.25 | 0.25 | 0.03 | 0.8875742 |
| (b) | 0.50 | 0.50 | 0.03 | 0.9937141 |

各 lambda は1回だけ solve し、map と5つの interpolation time が同じ coupling を使う。`compute_T_X_to_Z` の旧 full-mixture denominator は廃止し、`compute_T_X_to_Z_C` と同じ matched density (6.2) に是正した。

### 3.6 Figures 6–7

data generation、fit/cost、solve、plot を別関数へ分けた。

- source/target ring: 半径 4 の GMM から300点ずつ独立 sample
- 実測平均半径: source `4.0052`, target `4.0068`
- source z standard deviation: 約 `0.19`（共分散は正定値）
- target height: `h=10`
- noise: 中心 `(10,0,10)`、半径0.3の球内一様、10点
- RNG: source/target/noise = seed 42/43/44
- GMM fit: source 6 / target 7 components、random state 42
- MGW/pMGW は同じ fitted parameters と pairwise W2 matrices を使用
- balanced MGW は full means、partial MGW は matched means で center
- point-level balanced GW に無効な `epsilon` 引数はない

| method | mass | source rows drawn | target noise landings |
|---|---:|---:|---:|
| GW2 | 1.0000000 | 300/300 | 10 |
| PGW2 | 0.9677419 | 294/300 | 0 |
| MGW2 | 1.0000000 | 300/300 | 0 |
| pMGW2 | 0.9677419 | 300/300 | 0 |

`solve_figure67` は4 method を各1回だけ solve する。Figure 7 は `map_mgw/map_pmgw`、Figure 6 はそれらへの nearest-target assignment を読む。Figure 6/7 sidecar の MGW2/pMGW2 `solve_id` は byte-for-byte 同一である。

coupling dimension は `300×310 -> 6×7` に減る。これは solver problem size の主張であり、N=300 における GMM fit 込み end-to-end 高速化の主張ではない。

## 4. provenance contract

`docs/notebook/metadata/figureN.json` は各 panel について次を持つ。

- paper parameter / solver input / normalization scale
- input point cloud、known/fitted GMM parameters の shape・dtype・SHA-256
- source/target/fit seed
- solver/API、iteration/fallback、runtime
- coupling shape・dtype・SHA-256、matched mass、row/column residual
- stable solve ID
- repository commit、PGW_Metric submodule commit
- Python/POT/NumPy/SciPy/scikit-learn/Matplotlib version
- final PDF path/bytes/SHA-256

parameter の静的 single source は `FIGURE_PARAMETERS` と `figure_manifest.json` である。出力名は `figure1.pdf` ... `figure7.pdf` で一意で、旧 `fig_lambda_epsilon_sweep.eps` の上書きはない。

## 5. 環境 lock

- Python: 3.13.15 (`.python-version`, `requires-python`)
- NumPy: 2.4.6
- SciPy: 1.18.0
- POT: 0.9.7
- scikit-learn: 1.9.0
- Matplotlib: 3.11.0
- pybind11: 3.0.4
- CMake minimum: 3.22.2
- Eigen: 3.3.9 exact
- uv: 0.12.3
- Quarto: 1.10.18 + tarball SHA-256 verification

direct dependencies は `pyproject.toml` で exact pin、transitive dependencies は `uv.lock` で固定する。Netlify は `latest` Quarto を取得せず、同じ固定 release と `uv sync --frozen` を使う。

## 6. 回帰 test と clean build

標準 `unittest` の9 tests を追加した。

- Figure 5 EPOT mass / residual
- Figure 4 weight sum と density integral
- `compute_T_X_to_Z` matched-density parity
- Figures 1–3 parameter / residual / plot layout
- Figures 6–7 geometry independence / radius / noise count
- PGW/pMGW mass `300/310` と penalty input
- Figure 6/7 shared solve contract
- MGW coupling translation invariance

検証 command:

```bash
MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba \
  uv run python -m unittest discover -s tests -v

MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba \
  uv run python tools/execute_notebooks.py

QUARTO_PYTHON="$PWD/.venv/bin/python" quarto render docs
```

結果:

- tests: 9/9 success
- notebook: 4/4 clean-kernel success
- embedded stderr: transparency / deprecated / nonconvergence warning なし
- Quarto 1.10.18: 5 pages render success

## 7. before / after

- baseline `90c1098`: `review/figures/90c1098/`
- P0: `review/figures/after-p0/`
- issue #6 complete: `review/figures/after-issue6/`

`after-issue6/manifest.json` は canonical Figure 1–7 の notebook/cell/output/SHA-256 を持つ。主な外観変化は次のとおり。

- Figure 1: categorical colors/legend/labels → continuous coupling colorbar
- Figure 2: 欠落 → exact 1×5 grid
- Figure 3: cost scale colorbar → coupling scale、非収束値 → high-accuracy value
- Figure 4: probability-normalized appearance → `Z_lambda` を保持する shared density scale
- Figure 5: separate rows → paper 2×5 composite（mass correction は外観より数値に効く）
- Figures 6–7: 半径約10/8の affine geometry → 半径4の独立 samples、同一 solve 由来の composite

issue #8 の比較本文は final branch/PR の commit URL で `after-issue6` を参照する。

## 8. 分離した技術課題

- [#14](https://github.com/yachimura-lab/EPGOT-PMGW/issues/14): legacy `cpp_engine` と Python EPOT の収束判定不一致
- [#15](https://github.com/yachimura-lab/EPGOT-PMGW/issues/15): Stiefel PGD の発散（canonical pipeline では backtracking により解消済み）
- [#18](https://github.com/yachimura-lab/EPGOT-PMGW/issues/18): 論文 §7.2 の未記載条件・penalty 換算・balanced MGW の記述修正

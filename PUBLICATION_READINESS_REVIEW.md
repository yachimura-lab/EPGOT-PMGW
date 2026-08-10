# 論文公開前の記述・成果物監査

監査日: 2026-08-10
対象: `main` (`3ef1e72`)

この文書は、論文公開前に削除・変更・追加を検討すべき記述と成果物をまとめたものです。

## 最優先

### 1. Figures 6–7 の lambda の説明を実装に合わせる（対応済み）

- [x] `docs/index.qmd` の変換式と component input `0.013249` を削除する。
- [x] point/component の両solverへ `lambda=0.01` をそのまま渡す説明に置き換える。

変更前の `docs/index.qmd` には次の説明がありました。

> Point and mixture costs use different normalization constants. A shared
> dimensionless value is therefore converted separately.

実装・manifest・sidecarに合わせ、現在は次の条件を記載しています。

- point solver input: `0.01`
- component solver input: `0.01`
- PGW matched mass: `300/350`
- pMGW matched mass: `300/350`

推奨記述:

> Point and mixture costs are each divided by their own pair maximum. The
> paper's lambda=0.01 is passed unchanged to both partial solvers on their
> respective normalized cost matrices. In the canonical run, PGW and pMGW
> both transport 300/350 mass.

対象:

- `docs/index.qmd`
- `tools/build_notebooks.py`
- `docs/notebook/figure_manifest.json`
- `docs/notebook/metadata/figure6.json`
- `docs/notebook/metadata/figure7.json`

### 2. 最終論文とのFigure番号を照合する

- [ ] GW/noise実験が最終原稿のFigure 5かFigures 6–7かを確定する。
- [ ] README、公開サイト、ノートブック名、manifestを同じ番号へ統一する。

今回の作業依頼ではGW/noise実験を「Figure 5」と呼んでいましたが、現在のリポジトリではFigures 6–7として扱っています。

同時に、最終原稿と以下を照合してください。

- noise points: `50`
- source points: `300`
- target points: `350`
- transported mass: `300/350`
- point-level coupling dimension: `300×350`
- mixture-level coupling dimension: `6×7`
- 式番号: `(3.1)`, `(4.17)`, `(4.18)`, `(6.2)`, `(6.8)–(6.9)`
- Section番号: `7.1.1–7.1.5`, `7.2`

対象:

- `README.md`
- `docs/index.qmd`
- `tools/build_notebooks.py`
- `docs/notebook/*.ipynb`
- `docs/notebook/figure_manifest.json`

### 3. 「すべてのFigureがPDF」という記述を変更する

- [ ] `unique PDF` / `Every figure has one unique PDF output` を削除または変更する。
- [ ] 成果物形式を統一するか、形式に依存しない記述へ変更する。

現在のmanifestは次の構成です。

- Figures 1–5: PDF
- Figures 6–7: PNG

形式を混在させる場合の推奨記述:

> Each figure has one canonical output and an accompanying JSON sidecar.

対象:

- `README.md`
- `docs/index.qmd`
- `docs/notebook/figure_manifest.json`

### 4. manifestが参照する成果物を公開対象に含める

- [ ] Figures 1–5 のPDFを追跡するか、manifestから存在しない出力への参照を削除する。
- [ ] `outputs/figure6.png` と `outputs/figure7.png` を公開成果物として追跡するか方針を明記する。
- [ ] sidecarのoutput hashが公開されたファイルと一致することを確認する。

監査時点では、manifestが参照するFigures 1–5のPDFは存在せず、Figures 6–7のPNGは未追跡です。fresh cloneでmanifestの全出力を検証できる状態にする必要があります。

### 5. provenanceをcleanな状態で再生成する

- [ ] sidecarの `dirty` をすべて `false` にする。
- [ ] sidecarのsource commitを公開対象コミットに更新する。
- [ ] 全ノートブック、sidecar、Figure成果物をcleanなコミットから再生成する。

現在のsidecarはすべて `dirty: true` で、Figures 1–5とFigures 6–7がそれぞれ古いsource commitを記録しています。

対象:

- `docs/notebook/metadata/figure1.json`
- `docs/notebook/metadata/figure2.json`
- `docs/notebook/metadata/figure3.json`
- `docs/notebook/metadata/figure4.json`
- `docs/notebook/metadata/figure5.json`
- `docs/notebook/metadata/figure6.json`
- `docs/notebook/metadata/figure7.json`

### 6. `generated_at = 2000-01-01` の意味を修正する

- [ ] 固定された再現ビルド時刻を実際の生成日時として表示しない。
- [ ] `generated_at` を `source_date_epoch` に変更するか、実際の生成日時を別途記録する。
- [ ] schema versionを更新する。

現在の `generated_at: "2000-01-01T00:00:00+00:00"` は、Matplotlib成果物の再現性確保に使う `SOURCE_DATE_EPOCH` であり、実際の生成日時ではありません。

対象:

- `src/pgot/metadata.py`
- `docs/notebook/metadata/figure*.json`

## 公開品質として変更推奨

### 7. プロジェクト名と冒頭説明を統一する

- [ ] README、公開サイト、Quartoのタイトルを正式な論文タイトルへ統一する。
- [ ] `EPGOT/pMGW` と `EPGOT-pMGW` の表記を統一する。
- [ ] READMEのNetlify URLを説明付きMarkdownリンクにする。

現在、次のタイトルが混在しています。

- `EPGOT-pMGW — Entropic Partial Optimal Transport and Partial Gromov--Wasserstein Distance between Gaussian Mixtures`
- `Entropic Partial Optimal Transport and Gromov–Wasserstein`
- `Entropic Partial OT`

README冒頭の推奨例:

> This repository accompanies the EPGOT-pMGW paper. It provides
> implementations of entropic partial optimal transport between Gaussian
> mixtures and partial mixture Gromov–Wasserstein matching, together with
> executed notebooks that reproduce Figures 1–7.

対象:

- `README.md`
- `docs/index.qmd`
- `docs/_quarto.yml`

### 8. `pyproject.toml` の仮説明を変更する

- [ ] `description = "Add your description here"` を正式な説明へ変更する。
- [ ] 公開パッケージとして配布する場合はauthors、license、URLsも追加する。

対象:

- `pyproject.toml`

### 9. 未使用のテンプレート `main.py` を削除する

- [ ] 使用予定がなければ `main.py` を削除する。

現在は次の出力だけを行うテンプレートです。

```text
Hello from epgot-pmgw!
```

パッケージやCLIから参照されていません。

### 10. 内部レビュー・Issue番号の記述を削除または一般化する

- [ ] `tools/extract_review_figures.py` が不要なら削除する。
- [ ] 残す場合は `issue #8`、`after-issue6`、`review/` 依存を一般的な名前へ変更する。
- [ ] テスト内の `Issue #21` を、Issue番号に依存しない設計理由へ変更する。

対象:

- `tools/extract_review_figures.py`
- `tests/test_notebook_specification.py`
- `tests/paper_setup.py`

### 11. source of truthの説明を統一する

- [ ] 論文とノートブックのどちらがauthorityかを一貫して説明する。

現在は次の記述が混在しています。

- manifest / 公開サイト: paper equations and Section 7 conditions are the source of truth
- tests: notebooks are the authority

推奨する整理:

> The paper defines the experiment. The generated notebook source is the
> executable specification and must mirror the paper exactly.

対象:

- `docs/index.qmd`
- `docs/notebook/figure_manifest.json`
- `tests/test_notebook_specification.py`
- `tests/paper_setup.py`

### 12. `results published with it` の表現を確認する

- [ ] 対応する既公開成果がない場合は `historical results` または `earlier experimental results` に変更する。
- [ ] 実際に既公開成果がある場合は、対象バージョンや文献を明示する。

対象:

- `DEVELOPMENT.md`
- `src/pgot/legacy.py`

## 削除せず、論文・公開ページにも明記すべき制約

### 13. Partial GWの初期値依存性

次の説明は削除せず、論文または公開READMEからも確認できるようにしてください。

- Partial GWは非凸である。
- point/componentの両方でbalanced couplingを初期値として使う。
- component problemはsolverのdefault startでは空結合付近に停滞する。

対象:

- `DEVELOPMENT.md`
- `tools/build_notebooks.py`
- `src/pgot/paper_figures.py`
- 論文の実験条件

### 14. `d > d'` が未実装であること

現在の `partial_mgw_barycentric_map` は、論文実験で使う `d=d'=3` には対応していますが、一般の `d>d'` は未実装です。

- [ ] 論文が一般次元への対応を主張していないことを確認する。
- [ ] READMEまたは公開ドキュメントにlimitationsとして明記する。

対象:

- `DEVELOPMENT.md`
- `src/pgot/partial_mgw.py`
- 論文の定理・実装説明

## 追加推奨

- [ ] `LICENSE` を追加する。
- [ ] `CITATION.cff` またはBibTeX citationを追加する。
- [ ] READMEに論文URL、著者、arXiv/DOI、引用方法を追加する。
- [ ] `pyproject.toml` にauthors、license、project URLsを追加する。
- [ ] 公開直前にNetlifyとGitHubのリンクを手動確認する。

## 既存のpublication-readinessブランチについて

未統合の `agent/publication-readiness` ブランチには、次の修正案があります。

- 公開向けREADME・サイト冒頭文
- Figures 6–7のlambda説明
- `generated_at` から `source_date_epoch` への変更
- Figure成果物の追跡

ただし、このブランチにはnoise pointsが10点だった時点の記述や、Figures 6–7をPDFとして扱う変更も含まれます。現在の50点・PNG仕様へ調整せず、そのままマージしないでください。

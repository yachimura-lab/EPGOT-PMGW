# 論文公開前の記述・成果物監査

監査日: 2026-08-10
対象: `main` (`3ef1e72`)

この文書は、論文公開前に削除・変更・追加を検討すべき記述と成果物をまとめたものです。

## 公開品質として変更推奨

### 8. `pyproject.toml` の仮説明を変更する

- [ ] `description = "Add your description here"` を正式な説明へ変更する。
- [ ] 公開パッケージとして配布する場合はauthors、license、URLsも追加する。

対象:

- `pyproject.toml`

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

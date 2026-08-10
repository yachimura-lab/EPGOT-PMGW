# 論文公開前の記述・成果物監査

監査日: 2026-08-10
対象: `main` (`3ef1e72`)

この文書は、論文公開前に削除・変更・追加を検討すべき記述と成果物をまとめたものです。

## 公開品質として変更推奨

### 8. `pyproject.toml` の仮説明を変更する

- [ ] `description = "Add your description here"` を正式な説明へ変更する。
- [ ] 公開パッケージとして配布する場合はauthors、URLsも追加する。

対象:

- `pyproject.toml`

### 12. `results published with it` の表現を確認する

- [ ] 対応する既公開成果がない場合は `historical results` または `earlier experimental results` に変更する。
- [ ] 実際に既公開成果がある場合は、対象バージョンや文献を明示する。

対象:

- `DEVELOPMENT.md`
- `src/pgot/legacy.py`

## 追加推奨

- [ ] `CITATION.cff` またはBibTeX citationを追加する。
- [ ] READMEに論文URL、著者、arXiv/DOI、引用方法を追加する。
- [ ] `pyproject.toml` にauthors、project URLsを追加する。
- [ ] 公開直前にNetlifyとGitHubのリンクを手動確認する。

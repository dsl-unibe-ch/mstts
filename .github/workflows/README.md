# Release workflow

`publish.yml` builds and uploads `mstts` to PyPI on every pushed tag, using
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC — no API
token stored in the repo). The version is taken from the tag, so you never edit
`pyproject.toml`'s version by hand.

## One-time PyPI setup

Trusted Publishing needs the publisher registered on PyPI before the first run.

1. Go to https://pypi.org/manage/account/publishing/ (or, if the project already
   exists, its *Settings → Publishing* page).
2. Add a **GitHub** pending publisher with:
   - **PyPI Project Name:** `mstts`
   - **Owner:** `dsl-unibe-ch`
   - **Repository name:** `mstts`
   - **Workflow name:** `publish.yml`
   - **Environment name:** `pypi`  *(must match `environment:` in the workflow)*

> The project name `mstts` must be available/owned by you on PyPI. If the name is
> taken, pick another and update `[project].name` in `pyproject.toml`.

## Cutting a release

```bash
git tag v0.2.0
git push origin v0.2.0
```

The tag `v0.2.0` publishes version `0.2.0` (a leading `v` is stripped). PyPI
rejects re-uploading an existing version, so each release needs a new tag.

Only tags matching `v*` trigger a publish, so non-release tags are ignored.

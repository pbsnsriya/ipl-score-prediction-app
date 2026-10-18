# Contributing

How changes to ipl-score-prediction-app are named and versioned. The rules follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/),
[Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html) and [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

## Commit messages

```
<type>(<scope>): <description>        at most 72 characters, no full stop
```

Types: `feat` (something new, minor version), `fix` (a bug fixed, patch), `docs`, `refactor`, `test`, `build`, `ci`, `chore`, `style`, `perf`.
A `!` after the type marks a breaking change. Examples: `feat(report): add the confusion matrix`, `fix(app): handle an empty upload`, `docs: describe the dataset`.
One logical change per commit. Commits before version 0.1.0 were written before this convention and are not rewritten.

## Branches

`master` is the stable line. Larger changes go on a short-lived branch named `<type>/<short-description>` and are merged back with a pull request. Never force-push `master`.

## Versions

`MAJOR.MINOR.PATCH`, as in Semantic Versioning, written as a git tag with a `v` in front (`v0.1.0`). Major version 0 means the project is still changing; 1.0.0 means it is finished enough to rely on.

| Change | Version |
| --- | --- |
| Removes or changes something people rely on | minor before 1.0.0, major from 1.0.0 |
| Adds something new | minor |
| Fixes a mistake | patch |

## The changelog

[CHANGELOG.md](CHANGELOG.md) says in plain words what changed in each version. Add a line under `## [Unreleased]` for every change worth knowing about.

## Releasing

1. Move the `Unreleased` lines under a new heading `## [0.2.0] - YYYY-MM-DD` in `CHANGELOG.md` and commit it as `chore(release): v0.2.0`.
2. Tag it: `git tag -a v0.2.0 -m "ipl-score-prediction-app 0.2.0"`, then `git push && git push origin v0.2.0`.
3. Create the GitHub Release from the tag, with the changelog section as its text.

A released version is never edited: a mistake is fixed in the next version.

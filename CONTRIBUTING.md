# Contributing

MetaHuman Maker is developed by TREE Industries and contributors. Improvements to the original code and documentation are welcome under the [MIT license](LICENSE).

## Reporting a bug

Use the bug-report form and include versions, the exact action, expected/actual results and a small redistributable reproduction. For fit or material problems, include meaningful before/after images. Remove credentials, private paths and identifying data. Do not attach licensed templates, DNA, weights or customer scans.

Use [SECURITY.md](SECURITY.md) for vulnerabilities. For setup problems, check [installation](docs/installation.md) and [troubleshooting](docs/troubleshooting.md) first.

## Proposing a change

Open an issue for substantial solver, workflow or interface changes so the user-visible problem and acceptance scope are clear. Smaller fixes and documentation improvements can go straight to a pull request.

1. Fork and create a focused branch.
2. Follow the [development setup](docs/development.md) and run the lightweight source checks.
3. For geometry/material changes, provide one meaningful complete workflow with versions, inputs, before/after evidence and limits.
4. Describe the concrete problem, resulting behavior and validation in the PR. Include screenshots for visible UI changes.

Preserve artist inputs, accepted paint, native topology/UV identities, cancellation/resume and map provenance. Avoid character-specific fixes that silently alter unrelated anatomy. Do not claim state-of-the-art quality or universal production acceptance without supporting evidence.

Keep contributions redistributable and exclude generated character projects, binaries, model weights and unrelated experiments. Worker dependencies belong in separate environments, not Blender/Unreal's embedded Python. See [known limitations](docs/known-issues.md) for areas that need broader validation.

Please follow the [code of conduct](CODE_OF_CONDUCT.md).

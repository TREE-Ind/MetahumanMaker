# Preparing a release

This is the maintainer checklist. Creating/publishing a remote repository or a release is a separate action from local preparation.

## Source and documentation

- Confirm the add-on version in `create_metahuman/__init__.py`, the bridge version in its `.uplugin`, and the installer names in the guides.
- Keep the public README in `release/README.md` synchronized with the workspace README. The source tool prefers the public copy and writes it as the checkout's `README.md`.
- Check the quick start, screenshots, documented operator examples and current known limitations.
- Run the offline checks and generate a fresh source stage/archive. Inspect its inventory and Git diff. Confirm that only documentation screenshots, not texture atlases or character files, are included.

## Release assets

Prepare these alongside a SHA256 list:

| Asset | Purpose |
| --- | --- |
| `metahuman_maker-0.346.0.zip` | Blender installer. |
| `MetaHumanMakerBridge-UE5.8-0.29.0.zip` | Windows native bridge for the tested engine build. |
| `metahuman_maker-source-0.346.0.zip` | Audited source archive and manifest. |
| `TREE_Industries_MetaHuman_Maker_Launch.mp4` | TREE-branded launch film; upload as a release/media asset, not ordinary source history. |
| `SHA256SUMS.txt` | Content identities for the prepared assets. |

Install/enable the packaged add-on in an isolated Blender profile and verify its registration/cleanup. Check the bridge's actual engine build and source identity. A broad smoke check supplements the source checks; do not rerun every expensive character experiment for a documentation release.

## GitHub setup

1. Create the intended repository under the authorized account/organization and push only the reviewed source stage.
2. Set a description and topics such as `blender-addon`, `metahuman`, `unreal-engine`, `character-creation`, `blender-mcp` and `generative-textures`.
3. Enable Issues, Discussions if wanted, and private vulnerability reporting. Require source checks before merging when repository permissions/settings allow it.
4. Tag the tested source commit and attach the prepared installers, audited source ZIP, checksums and launch film. Use release notes that distinguish the add-on and bridge versions and state the tested environment.
5. After the media has a real hosted URL, add that permalink to the README. Do not publish a guessed video URL or a badge for checks that have not run.

Retain required third-party notices. The launch film uses Scott Buckley's **Emergent** under CC BY 4.0; copy its prepared music credit into the public video/release description. See [media notes](media/README.md).

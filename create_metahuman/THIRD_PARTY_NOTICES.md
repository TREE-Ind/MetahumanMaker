# License scope and third-party components

The MIT license in LICENSE covers original MetaHuman Maker code and documentation,
including the original Blender add-on, worker orchestration, web-viewer integration,
and Unreal editor bridge source. Commercial use, modification and redistribution
are permitted under that license. It does not relicense dependencies or user assets.

## Included third-party files

* Three.js 0.186.1: MIT. Original notice:
  `create_metahuman/web_vendor/three/LICENSE`. The pinned file inventory and
  source URLs are in `create_metahuman/web_vendor/manifest.json`.
* The groom Alembic helper uses Alembic and Imath. Their notices are retained in
  `create_metahuman/native/LICENSE-Alembic.txt` and `LICENSE-Imath.txt`.
  The source-release archive contains our writer source, not the compiled helper
  or Unreal SDK libraries. The installable Blender ZIP contains the Windows
  helper and those notices. Rebuild instructions accompany the writer.

## Separately obtained dependencies and content

Blender, Unreal Engine, MetaHuman plugins/Core Data and their SDKs keep their own
licenses. Obtain them from their respective vendors. Their code, SDK libraries,
MetaHuman character data, DNA and textures are not part of this source archive.

Optional Python packages, GNM/MediaPipe data, restoration models and diffusion
models keep their upstream terms. In particular, the PiSA-SR author adapter and
the Stable Diffusion base weights have separate terms; the project's MIT license
does not grant rights to those weights. Provisioning tools record upstream
sources and local identities. No model weights or Python environments are shipped
in the source release.

Studio environments are read from the local Blender installation when used.
Exported environments retain their source notices. Characters, input meshes,
paint, audio, grooms and generated output remain subject to their applicable
asset and service terms; no rights to third-party assets are granted by LICENSE.


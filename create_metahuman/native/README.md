# Windows groom Alembic writer

`cm_groom_alembic.exe` is built from `groom_alembic_writer.cpp` with Alembic 1.8.9
and Imath 3.2.2. Their license notices are included. No Unreal Engine code or
licensed character data is linked into this helper. The compiler uses the
Microsoft dynamic C++ runtime installed with Unreal/Visual Studio.

Rebuild from the repository with `python tools/build_groom_alembic.py` using the
installed UE 5.8 third-party SDK libraries and Visual Studio 2022 C++ tools.
The distributed helper is Windows x64. Blender does not need Unreal running
to export. The temporary binary input and staged output are removed afterward.

The writer accepts only the versioned binary stream produced by
`groom_alembic.py`. It emits one static linear-curve Ogawa archive in Unreal
centimeters, Z up, with uniform group/guide/root-UV attributes and vertex
widths/optional colors. It does not export simulation or animation caches.

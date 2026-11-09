"""Disposable background Blender entry point for one native geometry stage."""
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from create_metahuman.native_web_geometry import convert

request=json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_bytes())
convert(request['source'],Path(request['job'])/'geometry',face_component=request['face_component'],
        facial_mode='NATIVE_SAMPLED' if request.get('expression_bank') else 'DIRECT',legacy_component_frame=request['legacy_component_frame'],expression_bank=request.get('expression_bank'))

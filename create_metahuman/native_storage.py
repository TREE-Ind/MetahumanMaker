"""Prediction of the native commit's relative-key write and float32 evaluation.

Operation order follows Blender 4.3.2 key_evaluate_relative / rel_flerp:
https://github.com/blender/blender/blob/v4.3.2/source/blender/blenkernel/intern/key.cc
This is not accepted as authority until the native host verifies the prediction.
"""
import json
import numpy as np


def snapshot(context,obj,key_name):
    """Snapshot supported storage; unsupported stacks retain native-only checks."""
    from . import mesh_ops as m
    keys=obj.data.shape_keys
    if keys and (not keys.use_relative or obj.show_only_shape_key or any(k.vertex_group for k in keys.key_blocks)):return {}
    blocks=list(keys.key_blocks) if keys else []
    coords=np.empty((max(len(blocks),1),len(obj.data.vertices),3),dtype=np.float32)
    if blocks:
        for i,key in enumerate(blocks):key.data.foreach_get('co',coords[i].ravel())
        names=[key.name for key in blocks];values=np.array([key.value for key in blocks],dtype=np.float32);muted=np.array([key.mute for key in blocks]);relative=np.array([blocks.index(key.relative_key) for key in blocks],dtype=np.int64)
    else:
        obj.data.vertices.foreach_get('co',coords[0].ravel());names=['Basis'];values=np.zeros(1,dtype=np.float32);muted=np.zeros(1,dtype=bool);relative=np.zeros(1,dtype=np.int64)
    if key_name in names and names.index(key_name)==0:return {}
    data=dict(coords=coords,values=values,muted=muted,relative=relative,before=m.evaluated_coordinates(context,obj),matrix=np.asarray(obj.matrix_world),inverse=np.asarray(obj.matrix_world.inverted()))
    try:Predictor(data,names,key_name)
    except (ValueError,AssertionError):return {}
    data.update(names=np.frombuffer(json.dumps(names).encode(),dtype=np.uint8).copy(),key_name=np.frombuffer(key_name.encode(),dtype=np.uint8).copy())
    return {'native_storage_'+key:value for key,value in data.items()}


def from_snapshot(data):
    if 'native_storage_coords' not in data:return None
    values={key.removeprefix('native_storage_'):value for key,value in data.items() if key.startswith('native_storage_')}
    return Predictor(values,json.loads(bytes(values['names']).decode()),bytes(values['key_name']).decode())


class Predictor:
    def __init__(self,data,names,key_name):
        self.data=data;self.before=data['before'];self.coords=data['coords'];self.relative=data['relative'];self.values=data['values'];self.muted=data['muted'];self.old=names.index(key_name) if key_name in names else None
        self.index=len(names) if self.old is None else self.old
        self.baseline=self.coords[0 if self.old is None else self.relative[self.old]].astype(float)
        self.value=float(self.values[self.old]) if self.old is not None and self.values[self.old]!=0 else 1.
        self.old_delta=np.zeros_like(self.before) if self.old is None else (self.coords[self.old].astype(float)-self.baseline)*float(self.values[self.old])
        self.inverse=data['inverse'];self.matrix=data['matrix'];self.visible=self.local(self.before)
        self.world_from_local=np.linalg.inv(self.inverse[:3,:3])
        np.testing.assert_array_equal(self.evaluate(None),self.before)

    def local(self,points):return points@self.inverse[:3,:3].T+self.inverse[:3,3]

    def encode(self,planned):
        changed=np.any(planned!=self.before,axis=1)
        stored=(self.baseline+(self.local(planned)-self.visible+self.old_delta)/self.value).astype(np.float32)
        stored[~changed]=self.coords[self.old][~changed] if self.old is not None and self.values[self.old]!=0 else self.baseline[~changed]
        return stored

    def plan(self,stored,changed):
        local=self.visible-self.old_delta+(stored.astype(float)-self.baseline)*self.value
        result=(local-self.inverse[:3,3])@self.world_from_local.T
        result[~changed]=self.before[~changed]
        np.testing.assert_array_equal(self.encode(result),stored)
        return result

    def evaluate(self,stored):
        output=self.coords[0].copy()
        for i in range(1,len(self.coords)):
            value=np.float32(self.value) if stored is not None and i==self.old else self.values[i]
            if self.muted[i] or value==0:continue
            points=stored if stored is not None and i==self.old else self.coords[i]
            ref=stored if stored is not None and self.relative[i]==self.old else self.coords[self.relative[i]]
            output+=(points-ref)*value
        if stored is not None and self.old is None:output+=(stored-self.coords[0])*np.float32(self.value)
        return output.astype(float)@self.matrix[:3,:3].T+self.matrix[:3,3]

    def __call__(self,planned):return self.evaluate(self.encode(planned))

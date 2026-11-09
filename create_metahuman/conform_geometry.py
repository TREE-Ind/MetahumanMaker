"""Resolve native vertex order locally before Unreal's ambiguous UV lookup."""
from types import SimpleNamespace
import numpy as np
from . import expressions as e, geometry, mesh_ops as m


def prepare(context,s,f,*,include_correspondence=False):
    """Use the installed archetype library, independently of the preview identity."""
    reference=SimpleNamespace(expression_reference='',model_dir=f.model_dir)
    if not e.reference_path(reference).is_file():
        raise ValueError('Prepare the local native expression reference in Refine before delivering eyes and teeth; it supplies their exact vertex correspondence')
    data=e.bank(reference)
    meshes={v['name']:v for v in data['meshes']}
    head=meshes['head_lod0_mesh'];head_map=e.uv_mapping(s.template,head)
    rotation,scale,translation=geometry.similarity(np.asarray(head['neutral'])[head_map]*[.01,-.01,.01],m.coordinates(s.template))
    subjects={'head':(s.result,head,head_map)}
    anatomy={role:getattr(f,role) for role in ('eye_left','eye_right','teeth')}
    reassigned=False
    if anatomy['eye_left'] and anatomy['eye_right']:
        # Library object labels sometimes mean screen-left rather than the
        # native anatomical side. Compare the eye-pair direction in the fitted
        # head frame; absolute X signs fail on rotated / translated characters.
        left=np.asarray(meshes['eyeLeft_lod0_mesh']['neutral']).mean(0)
        right=np.asarray(meshes['eyeRight_lod0_mesh']['neutral']).mean(0)
        native_axis=((left-right)*[.01,-.01,.01])@rotation
        current_axis=(m.evaluated_coordinates(context,anatomy['eye_left']).mean(0)
                      -m.evaluated_coordinates(context,anatomy['eye_right']).mean(0))
        alignment=float(current_axis@native_axis)
        if abs(alignment)<np.linalg.norm(current_axis)*np.linalg.norm(native_axis)*.1:
            raise ValueError('Eye assignments do not span the native eye axis; check eye placement before conforming')
        if alignment<0:
            anatomy['eye_left'],anatomy['eye_right']=anatomy['eye_right'],anatomy['eye_left']
            reassigned=True
    for role,name in [('eye_left','eyeLeft_lod0_mesh'),('eye_right','eyeRight_lod0_mesh'),('teeth','teeth_lod0_mesh')]:
        obj=anatomy[role]
        if not obj:continue
        mesh=meshes[name]
        rest=(np.asarray(mesh['neutral'])*[.01,-.01,.01])@rotation*scale+translation
        # No deformation-equivalence aliases: distinct native anatomy vertices
        # must retain their own correspondence even if a bad fitted rig merged them.
        mapping=e.uv_mapping(obj,mesh,reference_points=rest)
        subjects[role]=(obj,mesh,mapping)
    output={};correspondence={}
    unit=context.scene.unit_settings.scale_length*100
    for role,(obj,mesh,mapping) in subjects.items():
        current=m.evaluated_coordinates(context,obj)
        if len(current)!=len(mapping):raise ValueError('Anatomy topology changed before canonical conform: '+obj.name)
        count=len(mesh['neutral']);counts=np.bincount(mapping,minlength=count)
        if len(counts)!=count or np.any(counts==0):raise ValueError('Native anatomy correspondence is incomplete: '+obj.name)
        positions=np.zeros((count,3));np.add.at(positions,mapping,current)
        positions/=counts[:,None]
        if np.linalg.norm(current-positions[mapping],axis=1).max(initial=0)*context.scene.unit_settings.scale_length>1e-5:
            raise ValueError('Restore split-vertex continuity before canonical conform: '+obj.name)
        output[role]=(positions*[1,-1,1]*unit).tolist()
        if include_correspondence:correspondence[role]=np.asarray(mapping,dtype=int).tolist()
    result={'method':'LOCAL_UV_CORRESPONDENCE','vertices':output,
            'anatomy_assignment':{role:obj.name for role,obj in anatomy.items() if obj},
            'eye_labels_normalized':reassigned}
    if any(obj and obj.get('cm_ocular_volume') for role,obj in anatomy.items() if role.startswith('eye_')):
        from . import ocular_profile
        import hashlib
        front=np.asarray({'X':(1,0,0),'-X':(-1,0,0),'Y':(0,1,0),'-Y':(0,-1,0)}[f.front_axis],dtype=float)
        basis=np.column_stack((np.cross([0.,0.,1.],front),-front,[0.,0.,1.]));native_basis=np.diag([1.,-1.,1.])@basis
        volumes={}
        for role in ('eye_left','eye_right'):
            obj=anatomy[role]
            if not obj or not obj.get(ocular_profile.PROPERTY):continue
            obj.data.calc_loop_triangles();tri=np.asarray([t.vertices[:] for t in obj.data.loop_triangles],dtype=np.int32)
            frame=ocular_profile.frame(obj[ocular_profile.PROPERTY],m.evaluated_coordinates(context,obj),tri,np.asarray(obj.matrix_world),basis)
            parameters=frame['parameters'];parameters['center']=(np.asarray(parameters['center'])*unit).tolist();parameters['radii']=(np.asarray(parameters['radii'])*unit).tolist()
            volumes[role]=dict(schema=1,coordinate_space='CANONICAL_CONFORM_CM',basis=native_basis.tolist(),parameters=parameters,
                canonical_positions_sha256=hashlib.sha256(np.asarray(output[role],dtype='<f4').tobytes()).hexdigest(),
                hidden_volume_inferred=True,native_motion_calibration_required=True)
        result['ocular_volumes']=volumes
    if include_correspondence:result['vertex_correspondence']=correspondence
    return result

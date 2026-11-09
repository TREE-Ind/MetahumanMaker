"""Create editable, surface-bound brow strands from an artist-painted UV mask."""
import numpy as np


def from_mask(context, head, density, *, color=(.04,.018,.01), count=1800,
              length=.004, width=.000065, root_offset=.00012, anatomical_region='BROWS',
              seed=42, name='PaintedEyebrows'):
    from . import mesh_ops as m, grooming
    density=np.asarray(density,dtype=float)
    if density.ndim!=2 or not np.isfinite(density).all() or np.any((density<0)|(density>1)):
        raise ValueError('Supply a finite top-down UV density mask between zero and one')
    if not 1<=count<=100000 or not 0<length<=.03 or not 0<width<=.001 or not 0<=root_offset<=.003:
        raise ValueError('Use a bounded brow count, length, strand width and root offset')
    m.check_mesh(head,'Brow attachment head')
    head.data.calc_loop_triangles()
    points=m.evaluated_coordinates(context,head)
    triangles=np.array([t.vertices[:] for t in head.data.loop_triangles])
    uv=np.array([[head.data.uv_layers.active.data[i].uv[:] for i in t.loops]
                 for t in head.data.loop_triangles])
    vertices=points[triangles]
    region=None
    if anatomical_region is not None:
        from . import groom_regions
        regions,_=groom_regions.masks(context,head,points)
        if anatomical_region not in regions:raise ValueError('Choose a supported anatomical growth region')
        region=regions[anatomical_region].astype(float)[triangles]
    cross=np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0])
    area=np.linalg.norm(cross,axis=1)
    def lookup(coords):
        x=np.clip((coords[...,0]*(density.shape[1]-1)).astype(int),0,density.shape[1]-1)
        y=np.clip(((1-coords[...,1])*(density.shape[0]-1)).astype(int),0,density.shape[0]-1)
        return density[y,x]
    # Vertex-only sampling misses narrow painted brows crossing broad polygons.
    support=np.maximum(lookup(uv).max(axis=1),lookup(uv.mean(axis=1)))
    for a in np.linspace(.1,.9,9):
        for b in np.linspace(.05,1-a,8):
            support=np.maximum(support,lookup(uv[:,0]*a+uv[:,1]*b+uv[:,2]*(1-a-b)))
    probability=area*(support>0)
    if region is not None:probability*=region.max(axis=1)>0
    if probability.sum()<=0:raise ValueError('The painted brow mask does not intersect the head UVs')
    rng=np.random.default_rng(seed);accepted=[]
    for _ in range(100):
        chosen=rng.choice(len(area),max(count*4,4096),p=probability/probability.sum())
        r=rng.random((len(chosen),2));q=np.sqrt(r[:,0])
        bary=np.column_stack((1-q,q*(1-r[:,1]),q*r[:,1]))
        roots_uv=np.einsum('ni,nij->nj',bary,uv[chosen])
        keep=rng.random(len(chosen))<lookup(roots_uv)
        if region is not None:keep &= np.sum(bary*region[chosen],axis=1)>.5
        accepted.extend(zip(chosen[keep],bary[keep],roots_uv[keep]))
        if len(accepted)>=count:break
    if len(accepted)<count:raise ValueError('Painted density is too sparse; enlarge the selected brow region')
    unit=context.scene.unit_settings.scale_length*100;curves=[];root_normals=[]
    color=np.asarray(color,float)
    for index,(face,bary,root_uv) in enumerate(accepted[:count]):
        root=bary@vertices[face];normal=cross[face]/area[face]
        # UV outward sweep, rising at the inner brow and flattening at the tail.
        side=1 if root_uv[0]>.5 else -1
        rising=np.clip(1-abs(root_uv[0]-.5)/.18,0,1)
        duv=uv[face,1:]-uv[face,0]
        try:tangent=np.linalg.solve(duv,vertices[face,1:]-vertices[face,0])
        except np.linalg.LinAlgError:continue
        direction=np.array([side,.2+.9*rising])@tangent
        direction-=normal*np.dot(direction,normal);direction/=max(np.linalg.norm(direction),1e-12)
        t=np.linspace(0,1,7);strand_length=length*rng.uniform(.65,1.25)
        path=root+t[:,None]*direction*strand_length+normal*(np.sin(t*np.pi)*.00035)[:,None]
        shade=np.clip(color*rng.uniform(.8,1.2),0,1)
        curves.append(dict(points=(path*[unit,-unit,unit]).tolist(),root_uv=root_uv.tolist(),
            widths=(width*unit*np.maximum(.05,1-t**1.8)).tolist(),
            colors=np.tile(shade,(7,1)).tolist(),group=0,guide=int(index%20==0)))
        root_normals.append(normal)
    groom=grooming.import_curves(context,{'curves':curves},name)
    try:
        grooming.attach(context,groom,head)
        # Native attachment snaps roots to the surface. Apply clearance after
        # attaching, retaining those surface UVs for subsequent deformation.
        if root_offset:
            positions=np.empty((len(groom.data.points),3),np.float32)
            groom.data.position_data.foreach_get('vector',positions.ravel())
            inverse=np.asarray(groom.matrix_world.inverted())[:3,:3]
            offsets=np.asarray(root_normals)@inverse.T*root_offset
            positions+=np.repeat(offsets,7,axis=0)
            groom.data.position_data.foreach_set('vector',positions.ravel())
            groom.data.update_tag();context.view_layer.update()
    except Exception:
        import bpy
        data=groom.data;bpy.data.objects.remove(groom,do_unlink=True)
        if not data.users:bpy.data.hair_curves.remove(data)
        raise
    grooming.preview_material(groom,(1,1,1),.48,strand_color=True)
    groom['cm_groom_method']='PAINTED_UV_DENSITY';groom['cm_groom_role']='Eyebrows'
    groom['cm_groom_root_offset']=root_offset
    return groom

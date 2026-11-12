"""Private albedo repair buffers, sparse stroke history and atomic publication."""
from pathlib import Path
import uuid
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree
from . import mesh_ops as m, skin_projection

TILE=64
TARGET='cm_repair_target'
REVISION='cm_repair_revision'
_REVISIONS={}


def pixels(image):
    count=len(image.pixels)
    width,height=image.size
    if width<1 or height<1 or count!=width*height*4:
        raise ValueError('The albedo image has no readable RGBA pixels')
    result=np.empty(count,dtype=np.float32);image.pixels.foreach_get(result)
    return result.reshape(height,width,4)


def map_image(obj,kind='ALBEDO'):
    if kind not in {'ALBEDO','ROUGHNESS'}:raise ValueError('Choose color or roughness for surface paint')
    images={node.image.as_pointer():node.image for mat in obj.data.materials if mat and mat.use_nodes
            for node in mat.node_tree.nodes if node.type=='TEX_IMAGE' and node.image and
            (node.get('cm_map_kind')==kind or kind=='ALBEDO' and node.name=='CM_Albedo')}
    if len(images)!=1:raise ValueError('Prepare one '+kind.lower()+' image for this skin part before Smart Paint')
    return next(iter(images.values()))


def albedo(obj):return map_image(obj,'ALBEDO')


def part(context):
    s,f=context.scene.cm_settings,context.scene.cm_features
    body=s.part=='BODY' or s.part=='FULL_BODY' and f.texture_part=='BODY'
    return ('BODY' if body else 'HEAD',f.body_result if s.part=='FULL_BODY' and body else s.result)


def reference(context,role,obj):
    s,f=context.scene.cm_settings,context.scene.cm_features
    explicit=f.reference_skin_body if role=='BODY' else f.reference_skin
    if explicit:return explicit
    from . import photo_skin
    photo=photo_skin.reference(obj,albedo(obj),role)
    if photo:return photo
    for root in (f.character_applied_skin_image,f.generated_local_skin_image):
        if root:
            image=root.get('cm_body_skin_reference' if role=='BODY' else 'cm_native_skin_reference')
            if image and image.get('cm_native_uv_layout')==skin_projection.uv_layout_signature(obj):return image
    template=f.body_template if s.part=='FULL_BODY' and role=='BODY' else s.template
    if template and skin_projection.uv_layout_signature(template)==skin_projection.uv_layout_signature(obj):
        try:return albedo(template)
        except ValueError:pass
    return None


def resample(image,width,height):
    return resample_values(pixels(image)[...,:3],width,height)


def resample_values(source,width,height,box=None):
    xstart,xend,ystart,yend=box or (0,width,0,height)
    if source.shape[:2]==(height,width):return source[ystart:yend,xstart:xend].copy()
    sh,sw=source.shape[:2]
    x=np.clip((np.arange(xstart,xend)+.5)*sw/width-.5,0,sw-1)
    y=np.clip((np.arange(ystart,yend)+.5)*sh/height-.5,0,sh-1)
    x0=x.astype(int);x1=np.minimum(x0+1,sw-1);u=(x-x0).astype(np.float32)
    result=np.empty((yend-ystart,xend-xstart,source.shape[2]),dtype=np.float32)
    for row in range(0,yend-ystart,128):
        yy=y[row:row+128];y0=yy.astype(int);y1=np.minimum(y0+1,sh-1)
        v=(yy-y0).astype(np.float32)[:,None,None]
        top=source[y0[:,None],x0]*(1-u[None,:,None])+source[y0[:,None],x1]*u[None,:,None]
        bottom=source[y1[:,None],x0]*(1-u[None,:,None])+source[y1[:,None],x1]*u[None,:,None]
        result[row:row+128]=top*(1-v)+bottom*v
    return result


def atomic_save(image,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    staging=path.parent/('.'+path.stem+'-repair-'+uuid.uuid4().hex[:8]+'.png')
    original_path,original_format=image.filepath_raw,image.file_format
    try:
        # Undo/Redo synchronization must not create or remove image IDs: doing
        # so during undo_post discards Blender's redo history.
        packed=bytes(image.packed_file.data) if image.packed_file and not image.is_dirty else None
        if packed and packed.startswith(b'\x89PNG\r\n\x1a\n'):
            # Packed clean PNGs already represent this exact history state.
            # Re-encoding changes file hashes despite identical pixel values.
            staging.write_bytes(packed)
        else:
            image.file_format='PNG'
            image.save(filepath=str(staging))
        staging.replace(path)
    finally:
        image.filepath_raw=original_path
        image.file_format=original_format
        staging.unlink(missing_ok=True)


def trim_inactive_image_cache(context,keep=()):
    """Release reloadable texture buffers outside every visible viewport.

    Retain dirty/generated images, visible materials, image editors and the
    current paint inputs. Image IDs, packed files and disk assets stay intact.
    Avoid querying image.size: that can decode an otherwise unloaded atlas.
    """
    retained={im.as_pointer() for im in keep if im}
    seen=set()
    def images(tree):
        if not tree or tree.as_pointer() in seen:return
        seen.add(tree.as_pointer())
        for node in tree.nodes:
            if node.type=='TEX_IMAGE' and node.image:retained.add(node.image.as_pointer())
            if node.type=='GROUP':images(node.node_tree)
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            space=area.spaces.active
            if area.type=='IMAGE_EDITOR' and space.image:retained.add(space.image.as_pointer())
            if area.type!='VIEW_3D':continue
            for obj in window.view_layer.objects:
                if obj.visible_get(view_layer=window.view_layer,viewport=space):
                    for slot in obj.material_slots:
                        if slot.material and slot.material.use_nodes:images(slot.material.node_tree)
    released=[]
    for image in bpy.data.images:
        if image.as_pointer() in retained or image.source!='FILE' or image.is_dirty:continue
        if not image.packed_file and not Path(bpy.path.abspath(image.filepath)).is_file():continue
        try:
            image.gl_free();image.buffers_free();released.append(image.name)
        except RuntimeError:continue
    return released


class PaintSession:
    def __init__(self,context,push_undo=False,controls=None,*,channel='ALBEDO',reference_image=None):
        from . import skin_reconstruction
        if skin_reconstruction.running():
            raise ValueError('Apply or discard the skin reconstruction before Smart Paint')
        from . import bake_jobs,detail_jobs,wrinkle_jobs,native_body,sculpt_session
        self.scene=context.scene;self.s=self.scene.cm_settings;self.f=self.scene.cm_features
        self.controls=controls if controls is not None else self.f
        self.selection_only=bool(getattr(self.controls,'mask_only',False))
        if self.f.expression_active:
            raise ValueError('Return to Neutral before Smart Paint so repaired skin remains on the accepted character')
        if context.mode!='OBJECT' or self.f.body_editing or self.f.full_sculpt or self.f.restore_editing or sculpt_session.pending():
            raise ValueError('Finish the active edit before Smart Paint')
        if any(job.running() for job in (bake_jobs,detail_jobs,wrinkle_jobs)) or native_body._JOB:
            raise ValueError('Wait for the current body or map operation before Smart Paint')
        self.role,self.obj=part(context);m.check_mesh(self.obj,'Skin part')
        if not self.obj.data.uv_layers.active:raise ValueError('Smart Paint needs the skin UV layout')
        self.uv_layout=skin_projection.uv_layout_signature(self.obj)
        self.channel=channel;self.source=map_image(self.obj,channel);self.reference_source=self.source if self.selection_only else reference_image or (reference(context,self.role,self.obj) if channel=='ALBEDO' else None)
        if not self.reference_source:raise ValueError('Load a matching reference skin for this part first')
        if self.source.source=='TILED' or self.reference_source.source=='TILED':
            raise ValueError('Use a flat head or body albedo image for Smart Paint')
        self.width,self.height=map(int,self.source.size)
        if min(self.width,self.height)<1 or max(self.width,self.height)>8192:
            raise ValueError('Smart Paint needs a readable albedo map up to 8K')
        from . import resource_budget
        count=self.width*self.height;ref_count=0 if self.selection_only else int(self.reference_source.size[0])*int(self.reference_source.size[1])
        required=count*64+ref_count*28+256*1024**2
        status=resource_budget.memory()
        self.trimmed_images=[]
        if status and status['available_commit_bytes']<required:
            self.trimmed_images=trim_inactive_image_cache(context,(self.source,self.reference_source))
        resource_budget.require('Smart Paint',required)
        resource_budget.require_disk('Smart Paint',Path(bpy.path.abspath(self.s.project_dir)),count*16+256*1024**2)
        self.snapshot=pixels(self.source)
        self.world=m.evaluated_coordinates(context,self.obj)
        if len(self.world)!=len(self.obj.data.vertices):raise ValueError('Paint the base skin before topology-changing modifiers')
        self.obj.data.calc_loop_triangles()
        self.triangles={}
        for triangle in self.obj.data.loop_triangles:
            self.triangles.setdefault(triangle.polygon_index,[]).append((triangle.vertices[:],triangle.loops[:]))
        self.surface_triangles=list(self.obj.data.loop_triangles)
        self.surface_polygons={}
        self.surface_neighbors=[set() for _ in self.surface_triangles]
        edges={}
        for index,triangle in enumerate(self.surface_triangles):
            self.surface_polygons.setdefault(triangle.polygon_index,[]).append(index)
            ids=triangle.vertices[:]
            for a,b in zip(ids,(*ids[1:],ids[0])):
                key=tuple(sorted((a,b)))
                for other in edges.get(key,()):
                    self.surface_neighbors[index].add(other);self.surface_neighbors[other].add(index)
                edges.setdefault(key,[]).append(index)
        self.bvh=BVHTree.FromPolygons(self.world,[p.vertices[:] for p in self.obj.data.polygons])
        self.center=(self.world.min(0)+self.world.max(0))*.5
        from . import paint_reference,skin_detail_color
        self.raw_reference=None if self.selection_only else pixels(self.reference_source)[...,:3].copy()
        self.reference=None if self.selection_only else paint_reference.ReferencePlane(self.raw_reference,self.snapshot,self.width,self.height,
            lambda:self.settings,skin_projection.linear,skin_projection.encoded,skin_detail_color.match_detail)
        self.pixels=self.snapshot.copy();self.mask=np.zeros((self.height,self.width),dtype=np.float32)
        self.touched=set()
        self.history=[];self.redo=[];self.stroke=None;self.settings=None;self.dirty=False;self.full_dirty=False;self.closed=False
        self.preview_image=None
        self.preview_pixels=None;self.preview_dirty=set()
        self.root=Path(bpy.path.abspath(self.s.project_dir)).resolve();self.name=self.s.character_name
        if not self.name or any(c in self.name for c in '/\\:'):raise ValueError('Choose a valid character name before Smart Paint')
        self.path=self.root/'Maps'/(self.name+('_Body' if self.role=='BODY' and self.s.part=='FULL_BODY' else '')+'_'+channel+'.png')
        self.materials=list(self.obj.data.materials);self.active_material=self.obj.active_material_index
        self.advanced=self.f.advanced_ui
        if not self.selection_only:
            self.source.pack()
            self.source[TARGET]=str(self.path)
            if not self.source.get(REVISION):self.source[REVISION]=uuid.uuid4().hex
            _REVISIONS[str(self.path)]=self.source[REVISION]
        if push_undo and not bpy.app.background:bpy.ops.ed.undo_push(message='Before Smart Paint')
        self.image=bpy.data.images.new(self.name+('_SkinRepair' if channel=='ALBEDO' else '_RoughnessRepair'),width=self.width,height=self.height,
                                       alpha=True,float_buffer=self.source.is_float)
        self.copies=[]
        self.image.colorspace_settings.name=self.source.colorspace_settings.name
        self.image.alpha_mode=self.source.alpha_mode
        for key in ('cm_photo_reference','cm_photo_projection_mask','cm_photo_previous_albedo'):
            if key in self.source:self.image[key]=self.source[key]
        if self.source.get('cm_photo_reference'):
            self.image['cm_photo_clean_reference']=self.reference_source
            self.image['cm_photo_clean_uv']=skin_projection.uv_layout_signature(self.obj)
        self.image.pixels.foreach_set(self.pixels.ravel());self.image.update()
        if max(self.width,self.height)>4096:
            self.preview_pixels=resample_values(self.snapshot,max(1,(self.width+3)//4),max(1,(self.height+3)//4))
            h,w=self.preview_pixels.shape[:2]
            self.preview_image=bpy.data.images.new(self.name+'_StrokePreview',width=w,height=h,alpha=True,float_buffer=self.source.is_float)
            self.preview_image.colorspace_settings.name=self.source.colorspace_settings.name
            self.preview_image.pixels.foreach_set(self.preview_pixels.ravel());self.preview_image.update()
        for index,original in enumerate(self.materials):
            if not original:continue
            copy=original.copy();self.copies.append(copy)
            if copy.use_nodes:
                for node in copy.node_tree.nodes:
                    if node.type=='TEX_IMAGE' and node.image==self.source:
                        node.image=self.image
                        if node.get('cm_map_kind')==channel or channel=='ALBEDO' and node.name=='CM_Albedo':copy.node_tree.nodes.active=node
            self.obj.data.materials[index]=copy
        self.f.advanced_ui=False
        self.correct()

    def correct(self):
        preserve=getattr(self.controls,'paint_preserve_color',False)
        radius=getattr(self.controls,'paint_detail_radius',8)
        values=(self.controls.paint_exposure,self.controls.paint_gamma,self.controls.paint_saturation,preserve,radius)
        if values==self.settings:return
        if preserve:
            if self.source.colorspace_settings.name!='sRGB' or self.reference_source.colorspace_settings.name!='sRGB':
                raise ValueError('Keep Local Skin Color requires sRGB albedo and reference images')
        self.settings=values
        if self.reference is not None:self.reference.clear()
        for tx,ty in self.touched:
            self.compose((tx*TILE,min((tx+1)*TILE,self.width),ty*TILE,min((ty+1)*TILE,self.height)))

    def compose(self,box):
        x0,x1,y0,y1=box;amount=self.mask[y0:y1,x0:x1,None]
        before=self.snapshot[y0:y1,x0:x1,:3]
        if self.selection_only:
            self.pixels[y0:y1,x0:x1,:3]=before+(np.array([.12,.8,.32],dtype=np.float32)-before)*amount*.65
        else:self.pixels[y0:y1,x0:x1,:3]=before+(self.reference.patch(box)-before)*amount
        self.touched.update((tx,ty) for ty in range(y0//TILE,(y1-1)//TILE+1) for tx in range(x0//TILE,(x1-1)//TILE+1))
        self.preview_dirty.update((tx,ty) for ty in range(y0//TILE,(y1-1)//TILE+1) for tx in range(x0//TILE,(x1-1)//TILE+1))
        self.dirty=True;self.full_dirty=True

    def begin(self):
        self.end();self.stroke={}

    def remember(self,x0,x1,y0,y1):
        if self.stroke is None:self.begin()
        for ty in range(y0//TILE,(y1-1)//TILE+1):
            for tx in range(x0//TILE,(x1-1)//TILE+1):
                key=(tx,ty)
                if key not in self.stroke:self.stroke[key]=self.mask[ty*TILE:(ty+1)*TILE,tx*TILE:(tx+1)*TILE].copy()

    def stamp(self,uv,erase=False):
        center=(np.asarray(uv)%1)*[self.width,self.height]
        radius=self.controls.paint_radius
        for dx in (-self.width,0,self.width):
            for dy in (-self.height,0,self.height):
                cx,cy=center+[dx,dy]
                x0,x1=max(0,int(np.floor(cx-radius))),min(self.width,int(np.ceil(cx+radius+1)))
                y0,y1=max(0,int(np.floor(cy-radius))),min(self.height,int(np.ceil(cy+radius+1)))
                if x0>=x1 or y0>=y1:continue
                y,x=np.mgrid[y0:y1,x0:x1]
                alpha=np.clip(1-np.sqrt((x-cx)**2+(y-cy)**2)/radius,0,1).astype(np.float32)**2*self.controls.paint_opacity
                self.remember(x0,x1,y0,y1)
                patch=self.mask[y0:y1,x0:x1]
                patch[:]=patch*(1-alpha) if erase else patch+(1-patch)*alpha
                self.compose((x0,x1,y0,y1))

    def apply_mask(self,weights,*,erase=False,expected_uv=None):
        """Apply an explicit atlas mask as one undoable Smart Paint stroke.

        The caller owns anatomical selection; masks do not infer damaged skin.
        Values use the same lower-left image origin as Blender pixel buffers.
        """
        if self.closed:raise ValueError('Start a new Smart Paint session before applying a mask')
        if (skin_projection.uv_layout_signature(self.obj)!=self.uv_layout
                or expected_uv is not None and expected_uv!=self.uv_layout):
            raise ValueError('The repair mask belongs to a different or changed UV layout')
        weights=np.asarray(weights,dtype=np.float32)
        if weights.shape!=self.mask.shape or not np.isfinite(weights).all() or np.any((weights<0)|(weights>1)):
            raise ValueError('Smart Paint mask needs matching atlas dimensions and finite weights from 0 to 1')
        y,x=np.nonzero(weights>0)
        if not len(x):return False
        x0,x1=int(x.min()),int(x.max())+1;y0,y1=int(y.min()),int(y.max())+1
        self.begin();self.remember(x0,x1,y0,y1)
        patch=self.mask[y0:y1,x0:x1];amount=weights[y0:y1,x0:x1]
        patch[:]=patch*(1-amount) if erase else patch+(1-patch)*amount
        self.compose((x0,x1,y0,y1));self.end();return True

    def stamp_surface(self,point,polygon,erase=False):
        """Paint a connected sphere/skin intersection, including split UV seams.

        Radius is converted from texels using the hit triangle's area density.
        Nearby disconnected surfaces and UV-only neighbors are never visited.
        Shared atlas texels necessarily remain shared by their consumers.
        """
        from mathutils import Vector
        from mathutils.geometry import closest_point_on_tri
        from .photo_color_core import raster
        if self.closed:raise ValueError('Start a new Smart Paint session before painting')
        if skin_projection.uv_layout_signature(self.obj)!=self.uv_layout:
            raise ValueError('The skin UV layout changed during Smart Paint')
        hit=self.uv_at(point,polygon)
        if hit is None:return False
        _,triangle,triangle_uv=hit
        texel=(triangle_uv[1:]-triangle_uv[0])*[self.width,self.height]
        texel_area=abs(float(np.linalg.det(texel)))
        world_area=float(np.linalg.norm(np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0])))
        if texel_area<1e-12 or world_area<1e-20:return False
        radius=float(self.controls.paint_radius)*np.sqrt(world_area/texel_area)
        center=np.asarray(point,dtype=float);center_vector=Vector(center)
        seeds=self.surface_polygons.get(polygon,())
        if not seeds:return False
        seed=min(seeds,key=lambda i:(closest_point_on_tri(center_vector,
            *(Vector(p) for p in self.world[list(self.surface_triangles[i].vertices)]))-center_vector).length_squared)
        pending=[seed];seen=set();tiles={};uv_layer=self.obj.data.uv_layers.active.data
        while pending:
            index=pending.pop()
            if index in seen:continue
            seen.add(index);t=self.surface_triangles[index]
            world=self.world[list(t.vertices)]
            closest=closest_point_on_tri(center_vector,*(Vector(p) for p in world))
            if np.linalg.norm(np.asarray(closest)-center)>radius:continue
            for neighbor in self.surface_neighbors[index]-seen:
                shared=list(set(t.vertices)&set(self.surface_triangles[neighbor].vertices))
                if len(shared)!=2:continue
                a,b=self.world[shared];edge=b-a;length=float(edge@edge)
                nearest=a+edge*np.clip(float((center-a)@edge)/max(length,1e-30),0,1)
                if np.linalg.norm(nearest-center)<=radius:pending.append(neighbor)
            patch=raster(np.asarray([uv_layer[i].uv[:] for i in t.loops]),self.width,self.height)
            if patch is None:continue
            y,x,bary=patch
            amount=np.maximum(1-np.linalg.norm(bary@world-center,axis=1)/radius,0)**2*float(self.controls.paint_opacity)
            keep=amount>0;y,x,amount=y[keep],x[keep],amount[keep]
            for key in set(zip(x//TILE,y//TILE)):
                tx,ty=map(int,key)
                tile=tiles.setdefault((tx,ty),np.zeros((min(TILE,self.height-ty*TILE),min(TILE,self.width-tx*TILE)),dtype=np.float32))
                selected=(x//TILE==tx)&(y//TILE==ty)
                np.maximum.at(tile,(y[selected]-ty*TILE,x[selected]-tx*TILE),amount[selected])
        for (tx,ty),amount in tiles.items():
            x0,y0=tx*TILE,ty*TILE;y1,x1=y0+amount.shape[0],x0+amount.shape[1]
            self.remember(x0,x1,y0,y1)
            patch=self.mask[y0:y1,x0:x1]
            patch[:]=patch*(1-amount) if erase else patch+(1-patch)*amount
            self.compose((x0,x1,y0,y1))
        return bool(tiles)

    def end(self):
        if not self.stroke:return
        item={key:(before,self.mask[key[1]*TILE:(key[1]+1)*TILE,key[0]*TILE:(key[0]+1)*TILE].copy())
              for key,before in self.stroke.items()}
        self.history.append(item);self.redo.clear();self.stroke=None
        while len(self.history)>1 and sum(a.nbytes+b.nbytes for stroke in self.history for a,b in stroke.values())>64*1024**2:
            self.history.pop(0)

    def undo(self,redo=False):
        self.end();source=self.redo if redo else self.history
        if not source:return
        item=source.pop();(self.history if redo else self.redo).append(item)
        for (tx,ty),pair in item.items():
            data=pair[1 if redo else 0];h,w=data.shape
            self.mask[ty*TILE:ty*TILE+h,tx*TILE:tx*TILE+w]=data
            self.compose((tx*TILE,tx*TILE+w,ty*TILE,ty*TILE+h))

    def bind_preview(self,image):
        for material in self.copies:
            if not material.use_nodes:continue
            for node in material.node_tree.nodes:
                if node.type=='TEX_IMAGE' and node.image and node.image in (self.image,self.preview_image):
                    node.image=image

    def publish(self,*,preview=False):
        if preview and max(self.width,self.height)>4096:
            if self.dirty:
                # Python's Image API uploads an entire plane. During a drag,
                # use a display image; authoritative pixels/masks remain 8K.
                # Restore full detail as soon as the stroke is released.
                h,w=self.preview_pixels.shape[:2]
                for tx,ty in self.preview_dirty:
                    # Expand by one display pixel for the bilinear footprint.
                    x0=max(0,int(tx*TILE*w/self.width)-1);x1=min(w,int(np.ceil((tx+1)*TILE*w/self.width))+1)
                    y0=max(0,int(ty*TILE*h/self.height)-1);y1=min(h,int(np.ceil((ty+1)*TILE*h/self.height))+1)
                    self.preview_pixels[y0:y1,x0:x1]=resample_values(self.pixels,w,h,(x0,x1,y0,y1))
                self.preview_dirty.clear()
                self.preview_image.pixels.foreach_set(self.preview_pixels.ravel());self.preview_image.update()
                self.bind_preview(self.preview_image);self.dirty=False
            return
        if self.full_dirty:
            self.image.pixels.foreach_set(self.pixels.ravel());self.image.update()
            self.dirty=False;self.full_dirty=False
        if self.preview_image:self.bind_preview(self.image)

    def uv_at(self,point,polygon):
        for vertices,loops in self.triangles.get(polygon,()):
            triangle=self.world[list(vertices)]
            ab=np.column_stack((triangle[1]-triangle[0],triangle[2]-triangle[0]))
            w=np.linalg.lstsq(ab,np.asarray(point)-triangle[0],rcond=None)[0]
            bary=np.array([1-w.sum(),*w])
            if bary.min()>=-.01:
                uv=np.asarray([self.obj.data.uv_layers.active.data[i].uv[:] for i in loops])
                return bary@uv,triangle,uv
        return None

    def finish(self,accept,*,mask_only=False):
        if self.closed:return
        mask_only=bool(mask_only or self.selection_only)
        self.end()
        if accept:
            if Path(bpy.path.abspath(self.s.project_dir)).resolve()!=self.root or self.s.character_name!=self.name:
                raise ValueError('Return to the original character/project before Apply Paint')
            if mask_only:
                if not np.any(self.mask):raise ValueError('Paint a brush region before keeping it for shading')
                self.retain_mask()
            else:
                self.correct();self.publish()
                self.image.pack()
                atomic_save(self.image,self.path)
                self.image.filepath_raw=str(self.path);self.image.file_format='PNG'
                self.image[TARGET]=str(self.path);self.image[REVISION]=uuid.uuid4().hex
                _REVISIONS[str(self.path)]=self.image[REVISION]
                self.s.status='Repaired '+self.role.lower()+' '+self.channel.lower()+' saved. Geometry and other maps retained.'
            if not mask_only and np.any(self.mask):
                try:
                    self.retain_mask()
                    self.s.status+=' Brush region retained for color or shading repair.'
                except Exception as exc:
                    # The color publication has already succeeded. Do not leave
                    # a modal session open or offer Cancel for a committed map.
                    self.s.status+=' Brush region could not be retained: '+str(exc)
        if not accept or mask_only:
            self.obj.data.materials.clear()
            for material in self.materials:self.obj.data.materials.append(material)
            self.obj.active_material_index=self.active_material
            for material in self.copies:
                if not material.users:bpy.data.materials.remove(material)
            if not self.image.users:bpy.data.images.remove(self.image)
            self.s.status=('Brush region saved for color or shading repair; original albedo retained.' if accept and mask_only
                           else 'Smart Paint canceled; original albedo retained.')
        if self.preview_image and not self.preview_image.users:bpy.data.images.remove(self.preview_image)
        self.preview_image=None
        self.preview_pixels=None;self.preview_dirty.clear()
        self.f.advanced_ui=self.advanced;self.f.last_action_error=False
        self.closed=True
        self.history.clear();self.redo.clear();self.touched.clear();self.stroke=None
        self.snapshot=self.pixels=self.mask=self.reference=self.raw_reference=None

    def retain_mask(self):
        """Retain this brush region with the accepted part, without new scratch."""
        from . import map_provenance
        image=bpy.data.images.new(self.name+'_'+self.role+'_PaintRegion',
            width=self.width,height=self.height,alpha=False)
        path=self.root/'Masks'/(self.name+'_'+self.role+'_paint_'+uuid.uuid4().hex[:8]+'.png')
        try:
            image.colorspace_settings.name='Non-Color'
            # These session buffers are about to be released; reuse one rather
            # than allocating another complete 4K RGBA plane.
            self.snapshot[:,:,:3]=self.mask[:,:,None]
            self.snapshot[:,:,3]=1
            image.pixels.foreach_set(self.snapshot.ravel());image.update()
            atomic_save(image,path)
            image.filepath_raw=str(path);image.file_format='PNG';image.pack()
            image['cm_native_uv_layout']=skin_projection.uv_layout_signature(self.obj)
            image['cm_paint_geometry']=map_provenance.mesh_digest(bpy.context,self.obj)
            image['cm_paint_role']=self.role
            self.obj['cm_last_paint_mask']=image
        except BaseException:
            if not image.users:bpy.data.images.remove(image)
            raise


def reset_history_cache():
    _REVISIONS.clear()


def synchronize_undo(scene):
    """Global undo changes packed image bindings; keep managed map files in sync."""
    s,f=scene.cm_settings,scene.cm_features
    root=Path(bpy.path.abspath(s.project_dir)).resolve()/'Maps'
    for body,obj in ((s.part=='BODY',s.result),(True,f.body_result)):
        if not obj:continue
        for kind in ('ALBEDO','ROUGHNESS'):
            try:image=map_image(obj,kind)
            except ValueError:continue
            target=image.get(TARGET);revision=image.get(REVISION)
            if not target or not revision:continue
            path=Path(target).resolve()
            expected=root/(s.character_name+('_Body' if body and s.part=='FULL_BODY' else '')+'_'+kind+'.png')
            if path!=expected:continue
            if str(path) not in _REVISIONS:
                # Loading/enabling the add-on is not a paint edit. An unrelated
                # geometry Undo must not overwrite externally edited files.
                _REVISIONS[str(path)]=revision
                continue
            if _REVISIONS[str(path)]!=revision:
                atomic_save(image,path)
                _REVISIONS[str(path)]=revision

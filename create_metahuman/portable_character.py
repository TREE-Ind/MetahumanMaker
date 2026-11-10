"""Convert a completed native web character to a self-contained, stock glTF asset.

No renderer hooks, viewer, environment, network, Blender or Unreal process is
required. Native materials and bound strands become portable approximations.
"""
import argparse
import copy
import hashlib
import io
import json
import re
import struct
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


class Character:
    def __init__(self, source, *, compact=False):
        self.compact=bool(compact)
        self.source=Path(source).resolve()
        raw=(self.source/'character.glb').read_bytes()
        magic,version,total=struct.unpack_from('<III',raw)
        if magic!=0x46546c67 or version!=2 or total!=len(raw):raise ValueError('Invalid source GLB')
        size,kind=struct.unpack_from('<II',raw,12)
        self.doc=json.loads(raw[20:20+size]);offset=20+size
        length,kind=struct.unpack_from('<II',raw,offset)
        if kind!=0x004e4942:raise ValueError('Missing embedded geometry buffer')
        self.original=raw[offset+8:offset+8+length]
        self.original_views=copy.deepcopy(self.doc['bufferViews'])
        self.original_accessors=copy.deepcopy(self.doc['accessors'])
        self.original_skins=copy.deepcopy(self.doc['skins'])
        self.binary=bytearray();self.cache={};self.image_cache={};self.skin_reports=[]
        self.doc['bufferViews']=[];self.doc['accessors']=[]
        self.doc['images']=[];self.doc['textures']=[]
        self.doc['samplers']=[{'magFilter':9729,'minFilter':9987,'wrapS':10497,'wrapT':10497}]
        self.report={'generator':'MetaHumanMaker.PortableCharacter','source':str(self.source),
                     'compact':self.compact,'textures':[],'grooms':[],'limitations':[
            'Standard PBR approximates Unreal skin, eyes and hair; native SSS, micro-skin and driven material responses are not embedded.',
            'Native strands are approximated with reduced, skinned ribbons; no native groom simulation or RBF.',
            'Four strongest joint influences per vertex are retained and renormalized for stock game loaders.',
            'Facial sample morphs are retained; RigLogic and expression interpolation runtime are not required or included.',
            'No locomotion clips are present. Retarget game animations to the included MetaHuman skeleton.',
            'Not validated inside a live Spawn world; compatibility is checked with a stock GLTFLoader.']}

    def read(self, index):
        a=self.original_accessors[index];v=self.original_views[a['bufferView']] if 'bufferView' in a else None
        dtype={5121:'u1',5123:'<u2',5125:'<u4',5126:'<f4'}[a['componentType']]
        columns={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
        size=np.dtype(dtype).itemsize
        if v is None:values=np.zeros((a['count'],columns),dtype=dtype)
        else:
            start=v.get('byteOffset',0)+a.get('byteOffset',0)
            values=np.ndarray((a['count'],columns),dtype=dtype,buffer=self.original,offset=start,
                              strides=(v.get('byteStride',columns*size),size)).copy()
        if a.get('sparse'):
            s=a['sparse'];indices=s['indices'];iv=self.original_views[indices['bufferView']]
            ids=np.frombuffer(self.original,dtype={5121:'u1',5123:'<u2',5125:'<u4'}[indices['componentType']],count=s['count'],offset=iv.get('byteOffset',0)+indices.get('byteOffset',0))
            sv=self.original_views[s['values']['bufferView']]
            values[ids]=np.frombuffer(self.original,dtype=dtype,count=s['count']*columns,offset=sv.get('byteOffset',0)+s['values'].get('byteOffset',0)).reshape(-1,columns)
        if a.get('normalized'):values=values.astype(np.float32)/np.iinfo(dtype).max
        return values

    def view(self, raw, target=None):
        self.binary.extend(b'\0'*(-len(self.binary)%4));start=len(self.binary);self.binary.extend(raw)
        record={'buffer':0,'byteOffset':start,'byteLength':len(raw)}
        if target:record['target']=target
        self.doc['bufferViews'].append(record);return len(self.doc['bufferViews'])-1

    def attribute(self, values, kind=None, target=34962, bounds=False):
        values=np.ascontiguousarray(values)
        if values.ndim==1:values=values[:,None]
        if values.dtype.kind=='f':values=values.astype('<f4');component=5126
        elif values.dtype.itemsize<=2:values=values.astype('<u2');component=5123
        else:values=values.astype('<u4');component=5125
        record={'componentType':component,'count':len(values),
                'type':kind or {1:'SCALAR',2:'VEC2',3:'VEC3',4:'VEC4',16:'MAT4'}[values.shape[1]]}
        nonzero=np.flatnonzero(np.any(values!=0,axis=1))
        if len(nonzero) and component==5126 and values.shape[1]==3 and len(nonzero)<len(values)*.7:
            index_type='<u2' if len(values)<=65535 else '<u4'
            record['sparse']={'count':len(nonzero),'indices':{'bufferView':self.view(nonzero.astype(index_type).tobytes()),'componentType':5123 if index_type=='<u2' else 5125},
                              'values':{'bufferView':self.view(values[nonzero].tobytes())}}
        elif len(nonzero):record['bufferView']=self.view(values.tobytes(),target)
        if bounds:record.update(min=values.min(0).tolist(),max=values.max(0).tolist())
        self.doc['accessors'].append(record);return len(self.doc['accessors'])-1

    def copy_accessor(self, index):
        if index not in self.cache:
            a=self.original_accessors[index]
            target=self.original_views[a['bufferView']].get('target') if 'bufferView' in a else 34962
            self.cache[index]=self.attribute(self.read(index),a['type'],target,a['type']=='VEC3' and 'min' in a)
        return self.cache[index]

    def skin(self, primitive):
        attr=primitive['attributes'];indices=[];weights=[]
        for i in range(8):
            if 'JOINTS_'+str(i) not in attr:break
            indices.append(self.read(attr['JOINTS_'+str(i)]));weights.append(self.read(attr['WEIGHTS_'+str(i)]))
        if not indices:return None
        joints=np.concatenate(indices,1).astype(np.uint16);values=np.concatenate(weights,1)
        values/=np.maximum(values.sum(1,keepdims=True),1e-12)
        order=np.argsort(-values,axis=1,kind='stable')[:,:4]
        selected=np.take_along_axis(values,order,1);lost=1-selected.sum(1)
        selected/=np.maximum(selected.sum(1,keepdims=True),1e-12)
        self.skin_reports.append({'vertices':len(values),'discarded_weight_mean':float(lost.mean()),'discarded_weight_p95':float(np.quantile(lost,.95)),'discarded_weight_max':float(lost.max())})
        return np.take_along_axis(joints,order,1),selected

    def image(self, image, name, jpeg=False):
        stream=io.BytesIO()
        if jpeg:image.convert('RGB').save(stream,format='JPEG',quality=97,subsampling=0)
        else:image.save(stream,format='PNG',compress_level=4)
        raw=stream.getvalue();sha=hashlib.sha256(raw).hexdigest()
        if sha in self.image_cache:return self.image_cache[sha]
        i=len(self.doc['images']);self.doc['images'].append({'name':name,'bufferView':self.view(raw),'mimeType':'image/jpeg' if jpeg else 'image/png'})
        t=len(self.doc['textures']);self.doc['textures'].append({'sampler':0,'source':i});self.image_cache[sha]=t
        self.report['textures'].append({'name':name,'size':list(image.size),'bytes':len(raw),'encoding':'JPEG97' if jpeg else 'PNG'})
        return t

    def load_image(self, record):
        p=(self.source/record['file']).resolve()
        if not p.is_relative_to(self.source):raise ValueError('Texture leaves the source character')
        return Image.open(p).convert('RGB')

    def material(self, name, recipe):
        kind=recipe['kind'];maps=recipe['maps'];scalars=recipe['scalars']
        result={'name':name,'doubleSided':True,'pbrMetallicRoughness':{'metallicFactor':0,'roughnessFactor':.5}}
        pbr=result['pbrMetallicRoughness']
        if kind in ('SKIN','TEETH','PBR'):
            if maps.get('basecolor'):
                image=self.load_image(maps['basecolor'])
                # Keep the full face resolution and lossless color. Body maps
                # retain the actual 4K authoring detail, not an 8K bake upscale.
                cap=2048 if self.compact and name.startswith('CMNative_Body') else 1024 if self.compact and kind=='TEETH' else 4096 if name.startswith('CMNative_Body') else 8192
                if max(image.size)>cap:image.thumbnail((cap,cap),Image.Resampling.LANCZOS)
                pbr['baseColorTexture']={'index':self.image(image,name+' BaseColor')}
            if maps.get('normal'):
                image=self.load_image(maps['normal'])
                cap=(4096 if kind=='SKIN' and not name.startswith('CMNative_Body') else 1024 if kind=='TEETH' else 2048) if self.compact else 4096 if name.startswith('CMNative_Body') else 8192
                if max(image.size)>cap:image.thumbnail((cap,cap),Image.Resampling.LANCZOS)
                a=np.asarray(image).copy();a[:,:,1]=255-a[:,:,1]
                result['normalTexture']={'index':self.image(Image.fromarray(a),name+' Normal')}
            if maps.get('roughness') or maps.get('packed'):
                rough=self.load_image(maps.get('packed') or maps['roughness'])
                cap=(1024 if kind=='TEETH' else 2048) if self.compact else 4096 if name.startswith('CMNative_Body') else 8192
                if max(rough.size)>cap:rough.thumbnail((cap,cap),Image.Resampling.LANCZOS)
                size=rough.size;r=np.asarray(rough.getchannel('G') if maps.get('packed') else rough.getchannel('R'))
                cavity=self.load_image(maps['cavity']).resize(size,Image.Resampling.BILINEAR).getchannel('R') if maps.get('cavity') else Image.new('L',size,255)
                ao=np.asarray(cavity,dtype=np.float32);ao=np.rint(255-(255-ao)*(.35 if kind=='SKIN' else 1)).astype(np.uint8)
                metal=np.asarray(self.load_image(maps['metallic']).resize(size,Image.Resampling.BILINEAR).getchannel('R')) if maps.get('metallic') else np.zeros(r.shape,np.uint8)
                orm=self.image(Image.fromarray(np.stack((ao,r,metal),2)),name+' OcclusionRoughnessMetallic')
                pbr.update(metallicRoughnessTexture={'index':orm},metallicFactor=1,roughnessFactor=1)
                result['occlusionTexture']={'index':orm}
        elif kind=='EYE':
            sclera=self.load_image(maps['sclera']).resize((1024,1024),Image.Resampling.LANCZOS)
            iris=self.load_image(maps['iris']);radius=max(.01,scalars.get('Cornea Size',.165));a=np.asarray(sclera).copy()
            extent=max(1,round(2*radius*1024));patch=np.asarray(iris.resize((extent,extent),Image.Resampling.LANCZOS));yy,xx=np.mgrid[:extent,:extent];distance=np.sqrt(((xx+.5)/extent-.5)**2+((yy+.5)/extent-.5)**2)
            alpha=np.clip((.5-distance)*extent/3,0,1)[:,:,None];left=(1024-extent)//2;block=a[left:left+extent,left:left+extent];block[:]=np.rint(block*(1-alpha)+patch*alpha).astype(np.uint8)
            pbr.update(baseColorTexture={'index':self.image(Image.fromarray(a),name+' BakedEye')},roughnessFactor=scalars.get('Cornea Roughness',.075))
        elif kind=='LASH':
            mask=self.load_image(maps['opacity']).getchannel('R');rgba=Image.new('RGBA',mask.size,(23,18,15,255));rgba.putalpha(mask)
            pbr.update(baseColorTexture={'index':self.image(rgba,name+' EyelashCoverage')},roughnessFactor=.6)
            result.update(alphaMode='MASK',alphaCutoff=.35)
        elif kind in ('EYE_SHELL','TEARLINE'):
            pbr.update(baseColorFactor=[.015,.01,.008,.12] if kind=='EYE_SHELL' else [1,1,1,.22],roughnessFactor=1 if kind=='EYE_SHELL' else .03)
            result['alphaMode']='BLEND'
        return result

    def convert(self):
        bank=json.loads((self.source/'native-expression-bank.json').read_text())
        if self.compact:
            full={sample['target'] for group in bank['groups'] for sample in group['samples'] if sample['strength']==1}
            for mesh in self.doc['meshes']:
                names=mesh.get('extras',{}).get('targetNames',[])
                if not names:continue
                keep=[i for i,name in enumerate(names) if name in full]
                for primitive in mesh['primitives']:
                    if 'targets' in primitive:primitive['targets']=[primitive['targets'][i] for i in keep]
                mesh['extras']['targetNames']=[names[i] for i in keep]
                if 'weights' in mesh:mesh['weights']=[mesh['weights'][i] for i in keep]
            self.report['limitations'].append('Compact profile retains full-strength samples for all expression groups; intermediate-strength samples are omitted.')
        recipes=json.loads((self.source/'native-materials.json').read_text())['materials']
        old_materials=self.doc['materials'];self.doc['materials']=[self.material(m['name'],recipes[m['name']]) for m in old_materials]
        self.face=copy.deepcopy(next(p for m in self.doc['meshes'] for p in m['primitives'] if old_materials[p['material']]['name']=='CMNative_Face_0'))
        self.face_skin=self.skin(self.face)
        for mesh in self.doc['meshes']:
            kept=[]
            for p in mesh['primitives']:
                if recipes[old_materials[p['material']]['name']]['kind']=='HIDDEN':continue
                original=copy.deepcopy(p);skin=self.skin(original)
                # Authoring vertex colors encode fit/region masks. The native
                # web material ignores them; stock glTF multiplies them into
                # albedo, so they must not become game character pigmentation.
                p['attributes']={n:self.copy_accessor(i) for n,i in original['attributes'].items() if not n.startswith(('JOINTS_','WEIGHTS_','COLOR_'))}
                if skin:
                    p['attributes'].update(JOINTS_0=self.attribute(skin[0]),WEIGHTS_0=self.attribute(skin[1]))
                if 'indices' in original:p['indices']=self.copy_accessor(original['indices'])
                if 'targets' in original:p['targets']=[{n:self.copy_accessor(i) for n,i in t.items()} for t in original['targets']]
                kept.append(p)
            mesh['primitives']=kept
        for skin in self.doc['skins']:skin['inverseBindMatrices']=self.copy_accessor(skin['inverseBindMatrices'])
        rename={s['target']:g['name']+'_'+str(round(s['strength']*100)).zfill(3) for g in bank['groups'] for s in g['samples']}
        for mesh in self.doc['meshes']:
            if mesh.get('extras',{}).get('targetNames'):mesh['extras']['targetNames']=[rename.get(n,n) for n in mesh['extras']['targetNames']]
        self.grooms(rename)
        self.link_body_and_face()
        self.report['skinning']=self.skin_reports
        self.report['expression_groups']=[g['name'] for g in bank['groups']]
        self.doc['asset']['generator']='MetaHuman Maker portable game character'

    def link_body_and_face(self):
        """Replace UE CopyPose's shared joints with actual shared glTF nodes.

        Compensated inverse binds retain the accepted neutral face even where
        the two imported skeletons have different lower-spine reference poses.
        """
        nodes=self.doc['nodes'];parents={c:i for i,n in enumerate(nodes) for c in n.get('children',[])};cache={}
        def world(i):
            if i in cache:return cache[i]
            n=nodes[i];matrix=np.eye(4)
            if 'matrix' in n:matrix=np.asarray(n['matrix']).reshape(4,4).T
            else:
                matrix[:3,:3]=Rotation.from_quat(n.get('rotation',[0,0,0,1])).as_matrix()@np.diag(n.get('scale',[1,1,1]))
                matrix[:3,3]=n.get('translation',[0,0,0])
            cache[i]=world(parents[i])@matrix if i in parents else matrix
            return cache[i]
        name=lambda i:re.sub(r'\.\d+$','',nodes[i].get('name',''))
        body=self.doc['skins'][0];face=self.doc['skins'][1]
        lookup={name(i):i for i in body['joints']}
        mapping={i:lookup[name(i)] for i in face['joints'] if name(i) in lookup and lookup[name(i)]!=i}
        for i in range(len(nodes)):world(i)
        inverse=self.read(self.original_skins[1]['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1).copy()
        for index,i in enumerate(face['joints']):
            if i in mapping:inverse[index]=np.linalg.inv(world(mapping[i]))@world(i)@inverse[index]
        for i,n in enumerate(nodes):
            parent=parents.get(i)
            if i in mapping:
                if parent is not None:nodes[parent]['children'].remove(i)
            elif parent in mapping:
                nodes[parent]['children'].remove(i);new_parent=mapping[parent];nodes[new_parent].setdefault('children',[]).append(i)
                matrix=np.linalg.inv(world(new_parent))@world(i)
                for prop in ('translation','rotation','scale'):n.pop(prop,None)
                n['matrix']=matrix.T.ravel().tolist()
        face['joints']=[mapping.get(i,i) for i in face['joints']]
        face['skeleton']=body.get('skeleton',parents.get(body['joints'][0],body['joints'][0]))
        face['inverseBindMatrices']=self.attribute(inverse.transpose(0,2,1).reshape(-1,16),'MAT4',target=None)
        self.report['shared_body_face_joints']=len(mapping)

    def grooms(self, rename):
        descriptor=json.loads((self.source/'grooms.json').read_text())
        face=self.read(self.face['attributes']['POSITION']);tree=cKDTree(face)
        face_node=next(i for i,n in enumerate(self.doc['nodes']) if n.get('mesh')==1)
        parent=next(n for n in self.doc['nodes'] if face_node in n.get('children',[]))
        limits={'Eyebrows':1500,'Fuzz':1200}
        for record in descriptor['grooms']:
            raw=(self.source/record['file']).read_bytes()
            def read(name, columns):
                item=record['layout'][name];return np.frombuffer(raw,dtype='<u4' if name=='triangles' else '<f4',offset=item['offset'],count=item['count']).reshape(-1,columns)
            anchors=read('anchors',3);tri=read('triangles',3);local=read('points',3);widths=read('widths',1)[:,0]
            maximum=limits.get(record['component'],4000 if 'cornrows' in record['name'].lower() else 8000)
            curves=record['curves'];chosen=np.linspace(0,len(curves)-1,min(maximum,len(curves)),dtype=int)
            count=6 if record['component']!='Hair' else 10
            selected=[curves[i] for i in chosen];ids=np.array([c['triangle'] for c in selected]);bary=np.array([c['bary'] for c in selected]);corners=anchors[tri[ids]]
            _,links=tree.query(corners.reshape(-1,3));links=links.reshape(-1,3)
            error=np.linalg.norm(face[links]-corners,axis=2).max()
            if error>1e-5:raise ValueError('Portable groom anchor does not match face geometry')
            def frame(corners):
                u=corners[:,1]-corners[:,0];u/=np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-12)
                n=np.cross(u,corners[:,2]-corners[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
                return np.stack((u,np.cross(n,u),n),axis=-1)
            frames=frame(corners);root=(corners*bary[:,:,None]).sum(1)
            points=[];radii=[]
            for c in selected:
                length=c.get('render_count',c['count']);v=local[c['offset']:c['offset']+length];w=widths[c['offset']:c['offset']+length]
                domain=np.r_[0,np.cumsum(np.linalg.norm(np.diff(v,axis=0),axis=1))];domain/=max(domain[-1],1e-12);t=np.linspace(0,1,count)
                density=len(curves)/len(selected)
                # Preserve projected coverage when replacing dense strands
                # with fewer ribbons. Fine fuzz uses a softer compensation.
                coverage=np.sqrt(density) if record['component']=='Fuzz' else density
                points.append(np.column_stack([np.interp(t,domain,v[:,k]) for k in range(3)]));radii.append(np.interp(t,domain,w)*.5*coverage)
            points=np.asarray(points);radii=np.asarray(radii)
            def ribbon(corners):
                frames=frame(corners);root=(corners*bary[:,:,None]).sum(1);p=np.einsum('nij,nkj->nki',frames,points)+root[:,None,:]
                tangent=np.gradient(p,axis=1);side=np.cross(tangent,frames[:,:,2][:,None,:]);side/=np.maximum(np.linalg.norm(side,axis=2,keepdims=True),1e-12)
                vertices=np.stack((p-side*radii[:,:,None],p+side*radii[:,:,None]),axis=2).reshape(-1,3)
                normals=np.repeat(np.repeat(frames[:,:,2],count,axis=0),2,axis=0)
                return vertices,normals
            vertices,normals=ribbon(corners)
            joints=[];weights=[]
            for row,bc in zip(links,bary):
                aggregate={}
                for vertex,amount in zip(row,bc):
                    for j,w in zip(self.face_skin[0][vertex],self.face_skin[1][vertex]):aggregate[int(j)]=aggregate.get(int(j),0)+float(w*amount)
                best=sorted(aggregate.items(),key=lambda p:-p[1])[:4]
                best+= [(0,0)]*(4-len(best));j,w=zip(*best);joints.append(j);weights.append(np.array(w)/max(sum(w),1e-12))
            joint=np.repeat(joints,count*2,axis=0).astype(np.uint16);weight=np.repeat(weights,count*2,axis=0).astype(np.float32)
            indices=[]
            for c in range(len(selected)):
                for i in range(count-1):
                    a=(c*count+i)*2;indices.extend((a,a+1,a+2,a+1,a+3,a+2))
            material=len(self.doc['materials']);self.doc['materials'].append({'name':record['name'],'doubleSided':True,'pbrMetallicRoughness':{'baseColorFactor':record['color']+[1],'metallicFactor':0,'roughnessFactor':max(.45,record['roughness'])}})
            primitive={'attributes':{'POSITION':self.attribute(vertices,bounds=True),'NORMAL':self.attribute(normals),'JOINTS_0':self.attribute(joint),'WEIGHTS_0':self.attribute(weight)},'indices':self.attribute(np.asarray(indices,np.uint32),target=34963),'material':material}
            mesh={'name':record['name'],'primitives':[primitive]}
            if record['component']=='Eyebrows':
                primitive['targets']=[]
                for target in self.face['targets']:
                    delta=self.read(target['POSITION']);moved,_=ribbon(corners+delta[links]);primitive['targets'].append({'POSITION':self.attribute(moved-vertices,bounds=True)})
                source_mesh=self.doc['meshes'][1];mesh['weights']=[0]*len(primitive['targets']);mesh['extras']=copy.deepcopy(source_mesh['extras'])
            mesh_index=len(self.doc['meshes']);self.doc['meshes'].append(mesh)
            node=len(self.doc['nodes']);self.doc['nodes'].append({'name':record['name'],'mesh':mesh_index,'skin':self.doc['nodes'][face_node]['skin']});parent['children'].append(node)
            self.report['grooms'].append({'name':record['name'],'source_strands':len(curves),'portable_ribbons':len(selected),'vertices':len(vertices),'facial_morphs':len(primitive.get('targets',[]))})

    def write(self, output, *, max_bytes=None):
        output=Path(output).resolve()
        if output.exists():raise ValueError('Choose a fresh output GLB')
        self.doc['buffers']=[{'byteLength':len(self.binary)}]
        encoded=json.dumps(self.doc,separators=(',',':')).encode();encoded+=b' '*(-len(encoded)%4)
        self.binary.extend(b'\0'*(-len(self.binary)%4));total=12+8+len(encoded)+8+len(self.binary)
        if max_bytes is not None and total>max_bytes:raise ValueError(f'Character GLB exceeds the upload budget: {total} > {max_bytes} bytes')
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('wb') as stream:
            stream.write(struct.pack('<III',0x46546c67,2,total));stream.write(struct.pack('<II',len(encoded),0x4e4f534a));stream.write(encoded)
            stream.write(struct.pack('<II',len(self.binary),0x004e4942));stream.write(self.binary)
        self.report.update(output=str(output),bytes=total,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),external_resources=0)
        return self.report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--output',required=True);parser.add_argument('--report',required=True);parser.add_argument('--compact',action='store_true');parser.add_argument('--max-bytes',type=int);args=parser.parse_args()
    character=Character(args.source,compact=args.compact);character.convert();report=character.write(args.output,max_bytes=args.max_bytes)
    Path(args.report).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({'output':args.output,'bytes':report['bytes'],'external_resources':0}))

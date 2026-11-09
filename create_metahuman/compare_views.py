"""Owned, resumable source/result comparison without repeated viewport splits."""
import json
import math
import time
import bpy
from bpy.app.handlers import persistent
from mathutils import Vector

MARKER='cm_compare_views'
_SESSIONS={}


def selected(context):
    # Timer/MCP contexts can have no area and expose no selected_objects member.
    return [o for o in context.view_layer.objects if o.select_get()]


def bounds(area):
    return [area.x,area.y,area.width,area.height]


def view_state(area):
    space=area.spaces.active;v=space.region_3d
    return (tuple(v.view_rotation),tuple(view_center(area)),v.view_distance,v.view_perspective,space.lens)


def canvas(area):
    """Visible window rectangle, excluding overlapping sidebars/toolbars."""
    region=next(r for r in area.regions if r.type=='WINDOW')
    left,right=0,region.width
    space=area.spaces.active
    for other in area.regions:
        shown=(other.type=='UI' and space.show_region_ui or
               other.type=='TOOLS' and space.show_region_toolbar)
        if not shown or other.width<=1:continue
        start,end=other.x-region.x,other.x+other.width-region.x
        if start<=0<end<region.width:left=max(left,end)
        elif 0<start<region.width<=end:right=min(right,start)
    right=max(right,left+1)
    return region,(left,0,right,region.height)


def center_offset(area):
    """View-plane pixel offset without float-matrix inverse round trips."""
    v=area.spaces.active.region_3d
    if v.view_perspective=='CAMERA':return (0.,0.,0.)
    region,(left,bottom,right,top)=canvas(area)
    if region.width<2 or region.height<2:return (0.,0.,0.)
    v.update()
    depth=v.view_distance if v.view_perspective=='PERSP' else 1.
    dx=((left+right)/region.width-1)*depth/v.window_matrix[0][0]
    dy=((bottom+top)/region.height-1)*depth/v.window_matrix[1][1]
    rotation=v.view_rotation.to_matrix()
    return tuple(rotation[i][0]*dx+rotation[i][1]*dy for i in range(3))


def view_center(area):
    location=area.spaces.active.region_3d.view_location
    offset=center_offset(area)
    return tuple(float(location[i])+offset[i] for i in range(3))


def set_center(area,point):
    v=area.spaces.active.region_3d
    offset=center_offset(area)
    v.view_location=tuple(float(point[i])-offset[i] for i in range(3))
    v.update()


def layout_state(area):
    region,rectangle=canvas(area)
    return (region.width,region.height,*rectangle)


def copy_view(source,target):
    a,b=source.spaces.active,target.spaces.active
    x,y=a.region_3d,b.region_3d
    center=view_center(source)
    y.view_rotation=x.view_rotation
    y.view_distance=x.view_distance;y.view_perspective=x.view_perspective
    b.lens=a.lens;b.clip_start=a.clip_start;b.clip_end=a.clip_end
    set_center(target,center)
    target.tag_redraw()


def same_view(a,b):
    if a[3]!=b[3]:return False
    # Quaternion assignment can normalize by a few ulps. Ignore that rounding
    # rather than mistaking the synchronized pane for fresh user navigation.
    av=(*a[0],*a[1],a[2],a[4]);bv=(*b[0],*b[1],b[2],b[4])
    return all(math.isclose(x,y,rel_tol=1e-6,abs_tol=1e-7) for x,y in zip(av,bv))


def adjacent(left,right):
    return (left.type==right.type=='VIEW_3D' and abs(left.y-right.y)<=2 and
            abs(left.height-right.height)<=2 and abs(left.x+left.width-right.x)<=3)


def remember(session):
    screen=session['screen']
    width=max(a.x+a.width for a in screen.areas);height=max(a.y+a.height for a in screen.areas)
    data={key:session[key] for key in ('original','hidden')}
    data['centers']=[[(a.x+a.width/2)/width,(a.y+a.height/2)/height] for a in session['areas']]
    screen[MARKER]=json.dumps(data)


def recover(window):
    screen=window.screen;key=screen.as_pointer()
    current=_SESSIONS.get(key)
    if current:
        try:
            # Area sizes settle on the next window event after area_split.
            # Retain ownership immediately so two OPEN calls cannot split twice.
            if all(a in list(screen.areas) and a.type=='VIEW_3D' for a in current['areas']):
                current['areas'].sort(key=lambda a:a.x)
                return current
        except ReferenceError:pass
        _SESSIONS.pop(key,None)
    raw=screen.get(MARKER)
    if not raw:return None
    try:
        data=json.loads(raw)
        width=max(a.x+a.width for a in screen.areas);height=max(a.y+a.height for a in screen.areas)
        candidates=[a for a in screen.areas if a.type=='VIEW_3D']
        pair=[min(candidates,key=lambda a:((a.x+a.width/2)/width-x)**2+((a.y+a.height/2)/height-y)**2)
              for x,y in data['centers']]
        if pair[0]==pair[1] or not adjacent(*pair):raise ValueError('Comparison panes changed')
        current={'screen':screen,'window':window,'areas':pair,'original':data['original'],
                 'hidden':data['hidden'],'previous':None,'driver':0,'members':None}
        _SESSIONS[key]=current
        return current
    except (ValueError,KeyError,TypeError):
        del screen[MARKER]
        return None


def groups(scene):
    s,f=scene.cm_settings,scene.cm_features
    from . import landmark_tools
    from .features import generated_head_update
    template=landmark_tools.active(scene) or f.ui_stage=='WRAP'
    target=[s.template if template else s.result or s.template]
    if s.part=='FULL_BODY' and not (template and generated_head_update(s)):
        target.append(f.body_template if template or not s.result else f.body_result)
    if s.part!='BODY' and s.result and not template:target.extend((f.eye_left,f.eye_right,f.teeth))
    if not template and f.ui_stage=='DETAIL' and not f.expression_active:
        if s.part=='FULL_BODY':
            if f.texture_part=='BODY' and f.body_detail:target=[f.body_detail if obj==f.body_result else obj for obj in target]
            elif f.texture_part=='HEAD' and s.detail:target=[s.detail if obj==s.result else obj for obj in target]
        elif s.detail:target=[s.detail if obj==s.result else obj for obj in target]
    target=[o for o in target if o and o.name in scene.objects]
    source=[s.source] if s.source and s.source.name in scene.objects else []
    return target,source


def members(context,session):
    """Keep helper activation from leaking a source into the result local view."""
    pair=session['areas'];scene=context.scene
    main=groups(scene)
    wanted=[set(objects) for objects in main]
    active=context.view_layer.objects.active
    f=scene.cm_features
    if active and active!=scene.cm_settings.source and (context.mode!='OBJECT' or f.full_sculpt or f.body_editing):
        wanted[0].add(active)
    for landmark in scene.cm_settings.landmarks:
        if landmark.template:wanted[0].add(landmark.template)
        if landmark.source:wanted[1].add(landmark.source)
    for side,area in enumerate(pair):
        space=area.spaces.active
        for obj in main[side]:
            if obj.hide_get():
                session['hidden'].setdefault(obj.name,True)
                obj.hide_set(False)
        if not space.local_view:
            # Enter local view using a temporary selection, then restore selection.
            previous_selection=selected(context);active=context.view_layer.objects.active
            try:
                for obj in previous_selection:obj.select_set(False)
                for obj in main[side]:
                    if obj.hide_get():session['hidden'].setdefault(obj.name,True);obj.hide_set(False)
                    obj.select_set(True)
                if not main[side]:continue
                context.view_layer.objects.active=main[side][0]
                with context.temp_override(area=area,region=next(r for r in area.regions if r.type=='WINDOW')):
                    bpy.ops.view3d.localview(frame_selected=False)
            finally:
                for obj in selected(context):obj.select_set(False)
                for obj in previous_selection:
                    if obj.name in context.view_layer.objects:obj.select_set(True)
                context.view_layer.objects.active=active
        if space.local_view:
            for obj in context.view_layer.objects:
                keep=obj in wanted[side]
                # local_view_get reads an evaluated cache in Blender 4.3 and can
                # report stale flags on original datablocks. Setter checks Base
                # flags itself and only tags actual changes.
                obj.local_view_set(space,keep)


def fit_objects(context, areas, objects, axis='-Y'):
    """Frame explicit working objects inside each unobscured viewport canvas."""
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    if any(canvas(area)[0].width < 2 or canvas(area)[0].height < 2 for area in areas):
        return False
    context.view_layer.update()
    graph = context.evaluated_depsgraph_get()
    corners = [evaluated.matrix_world @ Vector(corner) for obj in dict.fromkeys(objects)
               for evaluated in [obj.evaluated_get(graph)] for corner in evaluated.bound_box]
    if not corners:
        return False
    lower = Vector(tuple(min(p[i] for p in corners) for i in range(3)))
    upper = Vector(tuple(max(p[i] for p in corners) for i in range(3)))
    center = (lower + upper) * .5
    distance = max(max(upper - lower), 1e-4)
    rotation = Vector({'X': (1, 0, 0), '-X': (-1, 0, 0),
                       'Y': (0, 1, 0), '-Y': (0, -1, 0)}.get(axis, (0, -1, 0))).to_track_quat('Z', 'Y')
    factor = 1e-6
    for area in areas:
        view = area.spaces.active.region_3d
        view.view_rotation, view.view_perspective, view.view_distance = rotation, 'ORTHO', distance
        set_center(area, center)
        region, (left, bottom, right, top) = canvas(area)
        pixels = [location_3d_to_region_2d(region, view, point) for point in corners]
        factor = max(factor, max(abs(p.x - (left + right) * .5) for p in pixels) * 2 / ((right - left) * .82),
                     max(abs(p.y - (bottom + top) * .5) for p in pixels) * 2 / ((top - bottom) * .82))
    for area in areas:
        area.spaces.active.region_3d.view_distance = distance * factor
        set_center(area, center)
        area.tag_redraw()
    return True


def frame(context,session):
    """Fit both subjects into both unobscured canvases without selecting them."""
    from bpy_extras.view3d_utils import location_3d_to_region_2d
    if any(canvas(area)[0].width<2 or canvas(area)[0].height<2 for area in session['areas']):
        session['frame_pending']=True
        return
    session.pop('frame_pending',None)
    target,source=groups(context.scene)
    graph=context.evaluated_depsgraph_get()
    corners=[evaluated.matrix_world@Vector(corner) for obj in dict.fromkeys([*target,*source])
             for evaluated in [obj.evaluated_get(graph)] for corner in evaluated.bound_box]
    if not corners:raise ValueError('Choose a Source and a Template or Result before framing')
    lower=Vector(tuple(min(p[i] for p in corners) for i in range(3)))
    upper=Vector(tuple(max(p[i] for p in corners) for i in range(3)))
    center=(lower+upper)*.5
    distance=max(max(upper-lower),1e-4)
    axis=context.scene.cm_features.front_axis
    direction={'X':(1,0,0),'-X':(-1,0,0),'Y':(0,1,0),'-Y':(0,-1,0)}.get(axis,(0,-1,0))
    rotation=Vector(direction).to_track_quat('Z','Y')
    factor=1e-6
    lens=session['areas'][0].spaces.active.lens
    for area in session['areas']:
        space=area.spaces.active;v=space.region_3d
        space.lens=lens;v.view_rotation=rotation;v.view_perspective='ORTHO';v.view_distance=distance
        set_center(area,center)
        region,(left,bottom,right,top)=canvas(area)
        pixels=[location_3d_to_region_2d(region,v,p) for p in corners]
        # An 18% margin leaves room for orbit handles and silhouette inspection.
        factor=max(factor,max(abs(p.x-(left+right)*.5) for p in pixels)*2/((right-left)*.82),
                   max(abs(p.y-(bottom+top)*.5) for p in pixels)*2/((top-bottom)*.82))
    for area in session['areas']:
        area.spaces.active.region_3d.view_distance=distance*factor
        set_center(area,center)
        area.tag_redraw()
    session['previous']=[view_state(a) for a in session['areas']]
    session['layouts']=[layout_state(a) for a in session['areas']]
    session['frame_until']=time.monotonic()+.5


def check_idle(context):
    from . import paint_tools,sculpt_session
    f=context.scene.cm_features
    if context.mode!='OBJECT' or f.body_editing or f.full_sculpt or f.restore_editing or paint_tools.running() or sculpt_session.pending():
        raise ValueError('Finish the active edit before changing comparison views')


def open_views(context):
    check_idle(context)
    current=recover(context.window)
    if current:
        members(context,current)
        return current
    if not context.area or context.area.type!='VIEW_3D':raise ValueError('Open comparison from a 3D viewport')
    target,source=groups(context.scene)
    if not target or not source:raise ValueError('Choose a Source and a Template or Result before comparing')
    if set(target)&set(source):raise ValueError('Source and target must be separate objects to compare')
    hidden={obj.name:obj.hide_get() for obj in [*target,*source]}
    left=context.area;space=left.spaces.active
    original={'sidebar':space.show_region_ui,'toolbar':space.show_region_toolbar,
              'floor':space.overlay.show_floor,'relationships':space.overlay.show_relationship_lines}
    previous=set(context.screen.areas)
    result=bpy.ops.screen.area_split(direction='VERTICAL',factor=.5)
    added=[a for a in context.screen.areas if a not in previous and a.type=='VIEW_3D']
    if result!={'FINISHED'} or len(added)!=1:raise ValueError('Could not split the comparison viewport')
    pair=sorted([left,added[0]],key=lambda a:a.x)
    session={'screen':context.screen,'window':context.window,'areas':pair,'original':original,
             'hidden':hidden,'previous':None,'driver':0,'members':None}
    _SESSIONS[context.screen.as_pointer()]=session
    # Save ownership immediately so an interrupted setup can still be closed.
    remember(session)
    for area in pair:
        space=area.spaces.active
        if space.local_view:
            with context.temp_override(area=area,region=next(r for r in area.regions if r.type=='WINDOW')):
                bpy.ops.view3d.localview(frame_selected=False)
        space.show_region_ui=False;space.show_region_toolbar=False
        space.overlay.show_floor=False;space.overlay.show_relationship_lines=False
    members(context,session);frame(context,session);remember(session)
    label='template' if context.scene.cm_features.ui_stage=='WRAP' or not context.scene.cm_settings.result else 'result'
    context.scene.cm_settings.status='Comparison open: '+label+' on the left, source on the right. Navigate either view.'
    return session


def close_views(context):
    check_idle(context)
    session=recover(context.window)
    if not session:return
    left,right=session['areas']
    if not adjacent(left,right):raise ValueError('Comparison layout changed. Join its panes manually before opening another comparison.')
    # Defer destruction until the originating button/region context has returned.
    session['closing']=True


def close_now(context,session):
    left,right=session['areas'];original=session['original']
    if not adjacent(left,right):raise ValueError('Comparison layout changed before Close; join its panes manually')
    # Source area survives area_join; target area is removed. No region override
    # may point into the removed editor (including a click in the right header).
    source=(left.x+left.width//2,left.y+left.height//2)
    target=(right.x+right.width//2,right.y+right.height//2)
    with context.temp_override(window=session['window'],area=None,region=None):
        result=bpy.ops.screen.area_join(source_xy=source,target_xy=target)
    if result!={'FINISHED'}:raise ValueError('Blender could not close this comparison pane; join it manually')
    space=left.spaces.active
    space.show_region_ui=original['sidebar'];space.show_region_toolbar=original['toolbar']
    space.overlay.show_floor=original['floor'];space.overlay.show_relationship_lines=original['relationships']
    scene=session['window'].scene;layer=session['window'].view_layer
    hidden_state=dict(session['hidden'])
    def restore_visibility():
        # Joining refreshes local-view base flags on the next window event.
        # Restore user visibility after that refresh, in the exact view layer.
        try:
            for name,hidden in hidden_state.items():
                obj=scene.objects.get(name)
                if obj:obj.hide_set(hidden,view_layer=layer)
        except ReferenceError:pass
        return None
    bpy.app.timers.register(restore_visibility,first_interval=.05)
    screen=session['screen'];_SESSIONS.pop(screen.as_pointer(),None)
    if MARKER in screen:del screen[MARKER]
    context.scene.cm_settings.status='Comparison closed. Result view retained.'


def draw(layout,context,source_only=False):
    session=recover(context.window) if context.window else None
    if not session or context.area not in session['areas']:return False
    source=context.area==session['areas'][1]
    if source_only and not source:return False
    from . import landmark_tools
    target='Template' if landmark_tools.active(context.scene) or context.scene.cm_features.ui_stage=='WRAP' or not context.scene.cm_settings.result else 'Result'
    box=layout.box();box.label(text='Source Reference' if source else 'Source / '+target+' Comparison',icon='VIEW3D')
    row=box.row(align=True)
    row.prop(context.scene.cm_features,'sync_views',text='Link Navigation',toggle=True)
    row.operator('cm.dual_views',text='Close',icon='X').action='CLOSE'
    box.operator('cm.dual_views',text='Frame Both',icon='VIEWZOOM').action='FRAME'
    if source:box.label(text='Edit your character in the left view.')
    return source if source_only else True


def draw_header(self,context):
    session=recover(context.window) if context.window else None
    if not session or context.area not in session['areas']:return
    row=self.layout.row(align=True)
    from . import landmark_tools
    target='Template' if landmark_tools.active(context.scene) or context.scene.cm_features.ui_stage=='WRAP' or not context.scene.cm_settings.result else 'Result'
    row.label(text=target if context.area==session['areas'][0] else 'Source')
    row.prop(context.scene.cm_features,'sync_views',text='',icon='LINKED',toggle=True)
    row.operator('cm.dual_views',text='',icon='VIEWZOOM').action='FRAME'
    row.operator('cm.dual_views',text='',icon='X').action='CLOSE'


def tick():
    try:
        for window in bpy.context.window_manager.windows:
            session=recover(window)
            if not session:continue
            with bpy.context.temp_override(window=window):
                if not hasattr(window.scene,'cm_features'):continue
                if session.pop('closing',False):
                    try:close_now(bpy.context,session)
                    except (ValueError,RuntimeError) as exc:window.scene.cm_settings.status=str(exc)
                    continue
                members(bpy.context,session)
                if session.get('frame_pending'):
                    frame(bpy.context,session)
                    if session.get('frame_pending'):continue
                pair=session['areas']
                previous=session['previous']
                layouts=[layout_state(a) for a in pair]
                old_layouts=session.get('layouts')
                if previous and old_layouts and layouts!=old_layouts:
                    deadline=session.get('frame_until',0)
                    if time.monotonic()<deadline:
                        # Split/sidebar sizes settle after the initiating draw.
                        frame(bpy.context,session)
                        session['frame_until']=deadline
                        previous=session['previous']
                    else:
                        for index,area in enumerate(pair):
                            if layouts[index]!=old_layouts[index]:
                                set_center(area,previous[index][1])
                    session['layouts']=[layout_state(a) for a in pair]
                else:
                    session['layouts']=layouts
                states=[view_state(a) for a in pair]
                changed=[i for i in (0,1) if not previous or not same_view(states[i],previous[i])]
                if len(changed)==1:session['driver']=changed[0]
                if window.scene.cm_features.sync_views and changed:
                    driver=session['driver'];copy_view(pair[driver],pair[1-driver])
                    states=[view_state(a) for a in pair]
                session['previous']=states
                # Pausing does not stop the timer. Resume snaps the last driven view.
                linked=window.scene.cm_features.sync_views
                if linked and session.get('linked') is False:
                    driver=session['driver'];copy_view(pair[driver],pair[1-driver])
                    session['previous']=[view_state(a) for a in pair]
                session['linked']=linked
    except (ReferenceError,RuntimeError,AttributeError) as exc:
        # A closed window or file replacement must not leave a dead timer.
        print('MetaHuman Maker comparison:',str(exc))
    return .1


@persistent
def detach(*_):
    _SESSIONS.clear()


@persistent
def save_layout(*_):
    for session in list(_SESSIONS.values()):
        try:remember(session)
        except ReferenceError:pass


@persistent
def loaded(*_):
    if not bpy.app.timers.is_registered(tick):bpy.app.timers.register(tick,first_interval=.2,persistent=True)


def register():
    bpy.types.VIEW3D_HT_header.prepend(draw_header)
    for name,callback in (('load_pre',detach),('load_post',loaded),('save_pre',save_layout)):
        handlers=getattr(bpy.app.handlers,name)
        if callback not in handlers:handlers.append(callback)
    loaded()


def unregister():
    bpy.types.VIEW3D_HT_header.remove(draw_header)
    save_layout()
    if bpy.app.timers.is_registered(tick):bpy.app.timers.unregister(tick)
    for name,callback in (('load_pre',detach),('load_post',loaded),('save_pre',save_layout)):
        handlers=getattr(bpy.app.handlers,name)
        if callback in handlers:handlers.remove(callback)
    detach()

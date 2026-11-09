"""Create millimetre-scale interactive Blender point-cloud error comparisons.

Run with Blender --background --python export_laser_comparison_blender.py.
The measured vertices stay in scanner coordinates; Geometry Nodes supply only
display glyphs. No registration, filtering, smoothing or surface fitting occurs.
"""
from pathlib import Path
import argparse
import json
import math
import sys
import warnings

import bpy
import numpy as np
from mathutils import Quaternion, Vector

warnings.filterwarnings('ignore',category=DeprecationWarning)
ROOT=Path(__file__).resolve().parent
DEST=ROOT/'LaserScans/comparison_results/blender'
DATA=DEST/'data'
UPPER=np.array([.1,.25,.5,1.,2.,5.])
HEX=['440154','443983','31688E','21918C','35B779','90D743','FDE725']
BAND_LABELS=['< 0,10 mm','0,10–0,25','0,25–0,50','0,50–1,00','1,00–2,00','2,00–5,00','≥ 5,00 mm']


def linear(hexcode):
    c=np.array([int(hexcode[i:i+2],16)/255. for i in (0,2,4)])
    return np.where(c<=.04045,c/12.92,((c+.055)/1.055)**2.4)


COLORS=np.array([linear(c) for c in HEX])


def mat(name,rgb):
    result=bpy.data.materials.new(name)
    result.diffuse_color=(*rgb,1.)
    return result


def setup_scene(scene):
    bpy.context.window.scene=scene
    scene.unit_settings.system='METRIC'
    scene.unit_settings.scale_length=.001
    scene.unit_settings.length_unit='MILLIMETERS'
    scene.render.engine='BLENDER_WORKBENCH'
    scene.display.shading.light='FLAT'
    scene.display.shading.color_type='MATERIAL'
    scene.display.shading.show_shadows=False
    scene.display.shading.show_cavity=False
    scene.display.shading.show_specular_highlight=False
    scene.display.shading.background_type='WORLD'
    scene.world=bpy.data.worlds.new(scene.name+'_World')
    scene.world.color=(.055,.065,.075)
    scene.render.resolution_x=1600
    scene.render.resolution_y=1100
    scene.render.resolution_percentage=100
    scene.view_settings.view_transform='Standard'
    scene.view_settings.look='None'
    scene.view_settings.exposure=0
    scene.view_settings.gamma=1
    scene['coordinate_unit']='mm'
    scene['color_bin_upper_edges_mm']=UPPER.tolist()
    scene['color_palette_srgb_hex']=','.join(HEX)
    scene['measurement_interpretation']='Unsigned nearest-laser-point distance after rigid alignment; not a calibrated +/- uncertainty interval.'
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='OUTLINER':
                area.spaces.active.show_restrict_column_viewport=True
            if area.type=='VIEW_3D':
                s=area.spaces.active
                s.clip_end=10000
                s.shading.type='SOLID';s.shading.light='FLAT';s.shading.color_type='MATERIAL'
                s.shading.show_shadows=False;s.shading.show_cavity=False
                s.shading.background_type='WORLD'
                s.overlay.show_floor=False;s.overlay.show_axis_x=False;s.overlay.show_axis_y=False
                s.region_3d.view_rotation=Quaternion((1,0,0,0))
                s.region_3d.view_perspective='ORTHO'


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.save_version=0
    scene=bpy.context.scene;scene.name='01 Poikkeamat'
    setup_scene(scene)
    bands=[mat('Poikkeama '+label,tuple(c)) for label,c in zip(BAND_LABELS,COLORS)]
    mats={'bands':bands,'laser':mat('Laserreferenssi',tuple(linear('4BA3D3'))),
          'photo':mat('Fotogrammetria',tuple(linear('F58A3A'))),
          'text':mat('Tekstit',tuple(linear('EDF2F7'))),
          'muted':mat('Selitteet',tuple(linear('B8C7D5')))}
    return scene,mats


def collection(scene,name):
    c=bpy.data.collections.new(name);scene.collection.children.link(c)
    return c


def point_mesh(name,points,error,bands=False):
    mesh=bpy.data.meshes.new(name)
    mesh.vertices.add(len(points))
    mesh.vertices.foreach_set('co',np.asarray(points,dtype=np.float32).ravel())
    attribute=mesh.attributes.new('error_mm','FLOAT','POINT')
    attribute.data.foreach_set('value',np.asarray(error,dtype=np.float32))
    if bands:
        ids=np.searchsorted(UPPER,error,side='right')
        attribute=mesh.attributes.new('error_band','FLOAT','POINT')
        attribute.data.foreach_set('value',ids.astype(np.float32))
        attribute=mesh.color_attributes.new('error_color','FLOAT_COLOR','POINT')
        rgba=np.column_stack([COLORS[ids],np.ones(len(points))]).astype(np.float32)
        attribute.data.foreach_set('color',rgba.ravel())
        mesh.color_attributes.active_color=attribute
    mesh.update()
    return mesh


def display_nodes(obj,materials,radius,bands=False):
    group=bpy.data.node_groups.new(obj.name+' – pisteiden näyttö','GeometryNodeTree')
    group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
    group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
    nodes,links=group.nodes,group.links
    source=nodes.new('NodeGroupInput');source.location=(-600,0)
    output=nodes.new('NodeGroupOutput');output.location=(500,0)
    join=nodes.new('GeometryNodeJoinGeometry');join.location=(300,0)
    if bands:
        named=nodes.new('GeometryNodeInputNamedAttribute');named.data_type='FLOAT'
        named.inputs['Name'].default_value='error_band';named.location=(-600,250)
    for index,material in enumerate(materials):
        sphere=nodes.new('GeometryNodeMeshIcoSphere')
        sphere.inputs['Radius'].default_value=radius
        sphere.inputs['Subdivisions'].default_value=1
        assign=nodes.new('GeometryNodeSetMaterial');assign.inputs['Material'].default_value=material
        links.new(sphere.outputs['Mesh'],assign.inputs['Geometry'])
        instance=nodes.new('GeometryNodeInstanceOnPoints')
        links.new(source.outputs['Geometry'],instance.inputs['Points'])
        links.new(assign.outputs['Geometry'],instance.inputs['Instance'])
        if bands:
            compare=nodes.new('ShaderNodeMath');compare.operation='COMPARE'
            compare.inputs[1].default_value=index;compare.inputs[2].default_value=.01
            links.new(named.outputs['Attribute'],compare.inputs[0])
            links.new(compare.outputs[0],instance.inputs['Selection'])
        links.new(instance.outputs['Instances'],join.inputs['Geometry'])
    links.new(join.outputs['Geometry'],output.inputs['Geometry'])
    modifier=obj.modifiers.new('Pisteiden näyttö – mitatut koordinaatit säilyvät','NODES')
    modifier.node_group=group
    obj['point_display_radius_mm']=radius
    obj['glyph_size_is_not_uncertainty']=True


def object_for(mesh,model,role,col,materials,radius,bands=False,offset=(0,0,0)):
    labels={'photo_error':'Fotogrammetria – poikkeamavärit','laser_reference':'Laserreferenssi – sininen',
            'photo_overlay':'Fotogrammetria – oranssi'}
    obj=bpy.data.objects.new(model['model']+' | '+labels[role],mesh);col.objects.link(obj)
    obj['role']=role;obj['model_name']=model['model'];obj['comparison']=model['comparison']
    obj['scan_frame']=model['scan_frame'];obj['point_count']=len(mesh.vertices)
    obj['display_offset_mm']=list(offset);obj.location=offset
    obj['error_metric']='photo-to-laser nearest 3D mm' if role!='laser_reference' else 'laser-to-photo nearest 3D mm; includes coverage'
    display_nodes(obj,materials,radius,bands)
    return obj


def text(col,body,location,size,material,name=None):
    data=bpy.data.curves.new(name or body[:32],'FONT');data.body=body
    data.size=size;data.space_line=1.2
    obj=bpy.data.objects.new(name or body[:32],data);col.objects.link(obj)
    obj.location=location;data.materials.append(material)
    obj['role']='annotation'
    return obj


def rect(col,x,y,w,h,z,material):
    mesh=bpy.data.meshes.new('Asteikon väriruutu')
    mesh.from_pydata([(x,y,z),(x+w,y,z),(x+w,y+h,z),(x,y+h,z)],[],[(0,1,2,3)])
    obj=bpy.data.objects.new('Asteikon väriruutu',mesh);col.objects.link(obj)
    mesh.materials.append(material);obj['role']='legend_swatch'
    return obj


def number(x):return f'{x:.3f}'.replace('.',',')


def legend(col,x,y,z,mats,horizontal=False):
    for index,(label,material) in enumerate(zip(BAND_LABELS,mats['bands'])):
        bx=x+index*95 if horizontal else x
        by=y if horizontal else y-index*9
        rect(col,bx,by,7,5,z,material)
        text(col,label,(bx+10,by+.2,z),3.2, mats['text'])


def add_objects(scene,model,npz,mats,meshes=None,offset=(0,0,0),overlay=False):
    if meshes is None:
        meshes=(point_mesh(model['model']+'_photo',npz['points_mm'],npz['error_mm'],True),
                point_mesh(model['model']+'_laser',npz['reference_points_mm'],npz['reference_error_mm']))
    c1=collection(scene,model['model']+' | 1 Poikkeamapilvi')
    c2=collection(scene,model['model']+' | 2 Laserreferenssi')
    c3=collection(scene,model['model']+' | 3 Oranssi fotogrammetria')
    a=object_for(meshes[0],model,'photo_error',c1,mats['bands'],.085,True,offset)
    b=object_for(meshes[1],model,'laser_reference',c2,[mats['laser']],.035,False,offset)
    c=object_for(meshes[0],model,'photo_overlay',c3,[mats['photo']],.075,False,offset)
    bpy.context.window.scene=scene
    for obj,hide in [(a,overlay),(b,not overlay),(c,not overlay)]:
        obj.hide_set(hide);obj.hide_render=hide
    return meshes,(a,b,c)


def set_view(scene,bounds):
    bpy.context.window.scene=scene
    xmin,xmax,ymin,ymax=bounds
    center=Vector(((xmin+xmax)/2,(ymin+ymax)/2,0))
    width,height=xmax-xmin,ymax-ymin
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':
                area.spaces.active.region_3d.view_location=center
                area.spaces.active.region_3d.view_distance=max(width,height)*1.05
    camera_data=bpy.data.cameras.new('Esikatselukamera')
    camera=bpy.data.objects.new('Esikatselukamera',camera_data);scene.collection.objects.link(camera)
    camera.location=center+Vector((0,0,1000));camera.rotation_euler=(0,0,0)
    camera_data.type='ORTHO';camera_data.clip_end=3000
    aspect=scene.render.resolution_x/scene.render.resolution_y
    camera_data.ortho_scale=max(width,height*aspect)*1.06
    scene.camera=camera;camera.hide_set(True)
    scene['preview_bounds_mm']=list(bounds)


def instructions(models):
    block=bpy.data.texts.new('LUE ENSIN – virhevertailu')
    if any(m.get('source_coordinate_conversion') for m in models):
        block.write('SINI: koordinaatiston kätisyys on korjattu epäsymmetristen syvennysten perusteella ennen jäykkää kohdistusta. Mittakaavaa ei soviteta. Alkuperäisen akselisuuntaeron syntypaikka ei ole tiedossa.\n\n')
    block.write('LASERREFERENSSIIN VERRATUT FOTOGRAMMETRIAPISTEET\n\n'
      'Valitse yläpalkin Scene-valikosta 01 Poikkeamat tai 02 Päällekkäin.\n'
      '01 näyttää pistekohtaiset poikkeamat yhteisellä millimetriasteikolla.\n'
      '02 näyttää laserpilven sinisenä ja fotogrammetrian oranssina.\n'
      'Oikean Outliner-listan silmäkuvakkeilla voit piilottaa ja näyttää pilvet.\n'
      'Numpad 7: ylhäältä; Numpad 1/3: sivulta; keskihiiren painike: kierto.\n'
      'Mesh-datan jokainen vertex on oikea mitattu piste. Geometry Nodes lisää vain näyttömerkit.\n'
      'error_mm = pisteen etäisyys lähimpään toisen menetelmän pisteeseen millimetreissä.\n'
      'Fotogrammetrian error_mm on ensisijainen poikkeama; laserpilven error_mm kuvaa myös puuttuvaa peittoa.\n'
      'error_band ja error_color vastaavat kuvassa näkyvää väriasteikkoa.\n'
      'Asteikon rajat: 0,10; 0,25; 0,50; 1; 2; 5 mm. Alaraja kuuluu luokkaan.\n'
      'Yli 5 mm:n pisteet ovat mukana ja näyttävät keltaisilta; tarkka arvo säilyy error_mm-kentässä.\n'
      'Pisteiden näyttökoko EI tarkoita mittausepävarmuutta.\n'
      'Tulokset ovat jäykän kohdistuksen jälkeisiä poikkeamia, eivät kalibroitu ±-virhemarginaali.\n'
      'Harmu: vain skannaus 0003, jossa ovat kuvissa näkyvät pyöreät ulokkeet.\n'
      'Yksittäistiedostossa mesh-koordinaatit ovat alkuperäisessä skannerin mm-koordinaatistossa.\n'
      'Yhteisnäkymässä kappaleparit on siirretty rinnakkain; kummankin pilven siirto on sama.\n\n'
      'Kappaleet: '+', '.join(m['model'] for m in models))


def save(scene,path,render=True):
    bpy.context.window.scene=scene
    scene.unit_settings.length_unit='MILLIMETERS'
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active=None
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    if render:
        scene.render.filepath=str(path.with_suffix('.png'))
        bpy.ops.render.render(write_still=True)


def individual(model,render=True):
    scene,mats=reset()
    with np.load(ROOT/model['files']['npz']) as arrays:
        data={k:arrays[k] for k in arrays.files}
    meshes,_=add_objects(scene,model,data,mats)
    points=np.vstack([data['points_mm'],data['reference_points_mm']])
    lo,hi=points.min(axis=0),points.max(axis=0)
    top=max(float(hi[1])+13,65.)
    bottom=min(float(lo[1])-20,-65.)
    left=float(lo[0])-100;z=float(hi[2])+10
    ann=collection(scene,'00 Ohjeet ja asteikko')
    d=model['forward_photo_to_scan']
    object_name=model['object']
    kind='3 asennon yhdistelmä' if model['model'].endswith('_threepiece') else 'Yhden asennon kuvaus'
    text(ann,object_name+' | '+kind,(left,top,z),5.,mats['text'])
    text(ann,'POIKKEAMA LASERREFERENSSISTÄ',(left,top-11,z),3.3,mats['muted'])
    legend(ann,left,top-24,z,mats)
    text(ann,'Etäisyys lähimpään laserpisteeseen\nAlaraja kuuluu väriluokkaan.',(left,top-91,z),2.6,mats['muted'])
    stat=f"Mediaani  {number(d['median_mm'])} mm\nRMSE       {number(d['rmse_mm'])} mm\n95 % ≤      {number(d['p95_mm'])} mm\nSuurin      {number(d['max_mm'])} mm"
    text(ann,stat,(left,top-108,z),3.2,mats['text'],'Mitatut poikkeamat')
    text(ann,f"{model['photogrammetry_points']:,} fotogrammetriapistettä\nLaser: skannaus {model['scan_frame']:04d}",
         (left,top-136,z),2.8,mats['muted'])
    bottom=min(bottom,top-165)
    text(ann,'Kohdistuksen jälkeinen poikkeama; ei kalibroitu ±-epävarmuus.',(left,bottom,z),2.8,mats['muted'])
    text(ann,'Vaihda Scene: 02 Päällekkäin  |  Pilvien näkyvyys: silmäkuvake',(left,bottom-7,z),2.8,mats['muted'])
    for key in ['model','comparison','scan_frame']:
        scene['model_name' if key=='model' else key]=model[key]
    for key in ['median_mm','rmse_mm','p95_mm','max_mm']:
        scene[key]=d[key]
    scene['accuracy_comparison_eligible']=True
    set_view(scene,(left-5,max(float(hi[0])+12,left+200),bottom-14,top+10))
    overlay=bpy.data.scenes.new('02 Päällekkäin');setup_scene(overlay)
    add_objects(overlay,model,data,mats,meshes,overlay=True)
    for key in scene.keys():overlay[key]=scene[key]
    oa=collection(overlay,'00 Ohjeet ja tunnusluvut')
    text(oa,object_name+' | laser ja fotogrammetria',(left,top,z),5,mats['text'])
    rect(oa,left,top-22,7,5,z,mats['laser']);text(oa,'Laserreferenssi',(left+10,top-21,z),3.5,mats['text'])
    rect(oa,left,top-33,7,5,z,mats['photo']);text(oa,'Fotogrammetria',(left+10,top-32,z),3.5,mats['text'])
    text(oa,'Sama koordinaatisto ja mittakaava.\nKierrä näkymää tai piilota toinen pilvi\nsilmäkuvakkeesta.',(left,top-51,z),3,mats['muted'])
    text(oa,stat,(left,top-84,z),3.4,mats['text'])
    text(oa,'Poikkeamien värikartta: Scene 01 Poikkeamat',(left,bottom,z),3,mats['muted'])
    set_view(overlay,(left-5,max(float(hi[0])+12,left+200),bottom-14,top+10))
    instructions([model])
    path=DEST/(model['model']+'_virhevertailu.blend')
    save(scene,path,render)
    return {'file':path.name,'models':[model['model']],'comparisons':[model['comparison']],
            'scenes':['01 Poikkeamat','02 Päällekkäin'],'overview':False}


def overview(models,render=True):
    scene,mats=reset()
    scene.name='01 Poikkeamat'
    scene.render.resolution_x=2400;scene.render.resolution_y=1100
    overlay=bpy.data.scenes.new('02 Päällekkäin');setup_scene(overlay)
    overlay.render.resolution_x=2400;overlay.render.resolution_y=1100
    annotations=[collection(scene,'00 Ohjeet ja asteikko'),collection(overlay,'00 Ohjeet ja asteikko')]
    for index,model in enumerate(models):
        with np.load(ROOT/model['files']['npz']) as arrays:data={k:arrays[k] for k in arrays.files}
        center=(data['reference_points_mm'].min(axis=0)+data['reference_points_mm'].max(axis=0))/2
        offset=np.array([index*140+65.,0.,0.])-center
        meshes,_=add_objects(scene,model,data,mats,offset=offset)
        add_objects(overlay,model,data,mats,meshes,offset=offset,overlay=True)
        d=model['forward_photo_to_scan']
        for ann in annotations:
            text(ann,model['object'],(index*140+15,85,45),7,mats['text'])
            text(ann,f"RMSE  {number(d['rmse_mm'])} mm\n95 % ≤ {number(d['p95_mm'])} mm",(index*140+15,-88,45),4,mats['text'])
    for i,(s,ann) in enumerate(zip((scene,overlay),annotations)):
        text(ann,'FOTOGRAMMETRIA / LASERVERTAILU',(15,121,45),8,mats['text'])
        text(ann,'Viisi kappaletta · kolmen asennon yhdistelmät · millimetrit',(15,106,45),4,mats['muted'])
        if i==0:legend(ann,15,-121,45,mats,horizontal=True)
        else:
            rect(ann,15,-121,7,5,45,mats['laser']);text(ann,'Laserreferenssi',(25,-120,45),4,mats['text'])
            rect(ann,175,-121,7,5,45,mats['photo']);text(ann,'Fotogrammetria',(185,-120,45),4,mats['text'])
        text(ann,'Väri: etäisyys lähimpään laserpisteeseen. Kohdistuksen jälkeinen poikkeama, ei kalibroitu ±-epävarmuus.',(15,-139,45),3.5,mats['muted'])
        text(ann,'Harmu: oikea puoli, skannaus 0003. Yksittäistiedostoissa on tarkempi näkymä ja pistekohtaiset arvot.',(15,-150,45),3.5,mats['muted'])
        s['overview']=True;s['model_names']=[m['model'] for m in models]
        set_view(s,(0,700,-162,138))
    instructions(models)
    path=DEST/'Kaikki_kappaleet_virhevertailu.blend'
    save(scene,path,render)
    return {'file':path.name,'models':[m['model'] for m in models],
            'comparisons':[m['comparison'] for m in models],
            'scenes':['01 Poikkeamat','02 Päällekkäin'],'overview':True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--models',nargs='*');parser.add_argument('--no-render',action='store_true')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    models=manifest['models']
    assert len(models)==11 and all(m['accuracy_comparison_eligible'] for m in models)
    assert all(m['scan_frame']==3 for m in models if m['object']=='Harmu')
    DEST.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for model in models:
        if args.models and model['model'] not in args.models:continue
        outputs.append(individual(model,not args.no_render))
        print('COMPARISON_EXPORTED',model['model'],flush=True)
    if not args.models:
        outputs.append(overview([m for m in models if m['model'].endswith('_threepiece')],not args.no_render))
    result={'coordinate_units':'mm','mesh_vertices_preserve_scanner_coordinates':True,
            'display_glyphs_are_not_uncertainty':True,'color_band_upper_edges_mm':UPPER.tolist(),
            'color_palette_srgb_hex':HEX,'files':outputs,'source_data_manifest':'data/manifest.json'}
    (DEST/('manifest.json' if not args.models else 'partial_manifest.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('LASER_COMPARISON_BLENDER_EXPORT_COMPLETE',len(outputs),flush=True)


if __name__=='__main__':main()

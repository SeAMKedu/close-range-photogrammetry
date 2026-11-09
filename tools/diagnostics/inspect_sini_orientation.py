"""Diagnostic only: test Sini axis parity without changing primary results."""

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.paths import PROJECT_ROOT

from pathlib import Path
import itertools
import json
import zipfile
from io import BytesIO
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline.compare_laser_scans import cloud, icp_rigid, apply_transform, stats, score_alignment

ROOT=PROJECT_ROOT
OUT=ROOT/'LaserScans/_inspection/Sini_orientation'
OUT.mkdir(parents=True,exist_ok=True)
ref=np.load(ROOT/'LaserScans/reference_import/frame_0001_Sini.npz')['points']
def original_photo(model):
    relative=f'LaserScans/comparison_results/{model}__scan_0001/distances.npz'
    backup=OUT/'superseded_before_correction.zip'
    if backup.exists():
        with zipfile.ZipFile(backup) as archive:
            return np.load(BytesIO(archive.read(relative)))['photo_aligned']
    return np.load(ROOT/relative)['photo_aligned']

photo=original_photo('Sini_threepiece')
source=np.asarray(cloud(photo).voxel_down_sample(.5).points)
target=np.asarray(cloud(ref).voxel_down_sample(.5).points)
centre=(np.percentile(target,1,axis=0)+np.percentile(target,99,axis=0))/2
centre[2]=np.median(target[:,2])
results=[]
variants={}
for signs in itertools.product([1,-1],repeat=3):
    transform=np.eye(4)
    transform[:3,:3]=np.diag(signs)
    transform[:3,3]=centre-transform[:3,:3]@centre
    trial=icp_rigid(source,target,transform,gates=(5.,2.,1.),iterations=60)
    aligned=apply_transform(photo,trial)
    error=cKDTree(ref).query(aligned,workers=4)[0]
    result={'signs':list(signs),'determinant':float(np.linalg.det(trial[:3,:3])),
            'score':score_alignment(source,target,trial),'forward':stats(error),
            'diagnostic_transform_from_current':trial.tolist()}
    results.append(result)
    variants[signs]=aligned
    print(signs, result['score']['selection_score'],result['forward']['p95_mm'],flush=True)
results.sort(key=lambda r:r['score']['selection_score'])
(OUT/'diagnostic.json').write_text(json.dumps({'diagnostic_only':True,'results':results},indent=2))
best=variants[tuple(results[0]['signs'])]
fig,axes=plt.subplots(1,3,figsize=(17,7),layout='constrained')
lo,hi=np.percentile(ref[:,2],[1,99])
for ax,points,title in zip(axes,[ref,photo,best],['Raw laser scan 0001','Current aligned photogrammetry','Best diagnostic: signs '+str(results[0]['signs'])]):
    # A common top view of the exact same XY coordinate system.
    p=points[::2]
    sc=ax.scatter(p[:,1],p[:,0],c=p[:,2],s=.6,cmap='viridis',vmin=lo,vmax=hi,rasterized=True)
    ax.set_aspect('equal'); ax.set_title(title); ax.set_xlabel('Scanner Y (mm)'); ax.set_ylabel('Scanner X (mm)')
    ax.set_xlim(ref[:,1].min()-3,ref[:,1].max()+3)
    ax.set_ylim(ref[:,0].min()-3,ref[:,0].max()+3)
fig.colorbar(sc,ax=axes,label='Scanner Z (mm)',shrink=.6)
fig.savefig(OUT/'height_orientation.png',dpi=180)
print('DONE',OUT,flush=True)

# Check the independent single model under the same horizontal-axis hypothesis.
single=original_photo('Sini_single')
sf=np.asarray(cloud(single).voxel_down_sample(.5).points)
initial=np.eye(4); initial[:3,:3]=np.diag([1.,-1.,1.])
initial[:3,3]=centre-initial[:3,:3]@centre
trial=icp_rigid(sf,target,initial,gates=(5.,2.,1.),iterations=60)
after=apply_transform(single,trial)
single_result={'diagnostic_only':True,'hypothesis':'Horizontal Y reversal followed by rigid refinement',
               'before':stats(cKDTree(ref).query(single,workers=4)[0]),
               'after':stats(cKDTree(ref).query(after,workers=4)[0]),
               'transform_from_current':trial.tolist()}
(OUT/'single_diagnostic.json').write_text(json.dumps(single_result,indent=2))
print('SINGLE_P95',single_result['before']['p95_mm'],single_result['after']['p95_mm'],flush=True)

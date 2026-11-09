"""Compare archived and corrected Sini heights using identical axes/scales."""

# Allow both direct script execution and python -m from the project root.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.paths import PROJECT_ROOT

from pathlib import Path
from io import BytesIO
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=PROJECT_ROOT
DEST=ROOT/'LaserScans/_inspection/Sini_orientation'
ref=np.load(ROOT/'LaserScans/reference_import/frame_0001_Sini.npz')['points']
lo,hi=np.percentile(ref[:,2],[1,99])
fig,axes=plt.subplots(2,3,figsize=(17,9),layout='constrained')
with zipfile.ZipFile(DEST/'superseded_before_correction.zip') as archive:
    for row,model in enumerate(['Sini_single','Sini_threepiece']):
        path=f'LaserScans/comparison_results/{model}__scan_0001/distances.npz'
        before=np.load(BytesIO(archive.read(path)))['photo_aligned']
        after=np.load(ROOT/path)['photo_aligned']
        for ax,points,title in zip(axes[row],[ref,before,after],['Raw laser scan','Superseded comparison','Corrected comparison']):
            p=points[::2]
            sc=ax.scatter(p[:,1],p[:,0],c=p[:,2],s=.5,cmap='viridis',vmin=lo,vmax=hi,rasterized=True)
            ax.set_aspect('equal'); ax.set_title(model+' | '+title,fontsize=11)
            ax.set_xlabel('Scanner Y (mm)'); ax.set_ylabel('Scanner X (mm)')
            ax.set_xlim(ref[:,1].min()-3,ref[:,1].max()+3)
            ax.set_ylim(ref[:,0].min()-3,ref[:,0].max()+3)
fig.colorbar(sc,ax=axes.ravel().tolist(),label='Scanner Z (mm)',shrink=.65)
fig.savefig(DEST/'corrected_height_correspondence.png',dpi=180)

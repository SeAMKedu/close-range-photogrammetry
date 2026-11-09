"""Show native measured Harmu height grids for face identification only."""
from pathlib import Path
import sys
import json
import os
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'LaserScans/_inspection/mpl_cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from laser_reference_io import load_scan

OUT = Path(__file__).resolve().parent
fig, axes = plt.subplots(1, 2, figsize=(10, 5.6), sharey=True)
fig.subplots_adjust(left=.08, right=.89, top=.82, bottom=.20, wspace=.24)
evidence = []
for ax, frame in zip(axes, (3, 4)):
    path = next((ROOT/'LaserScans').glob(f'*Frame-{frame:04d}*.pcd'))
    scan = load_scan(path)
    grid, valid = scan['grid_xyz'], scan['valid']
    rr, cc = np.nonzero(valid)
    dx = scan['metadata']['pcd_grid_spacing_fit_mm']['x']
    dy = scan['metadata']['pcd_grid_spacing_fit_mm']['y']
    x0 = float(np.median(grid[valid,0] - cc*dx))
    y0 = float(np.median(grid[valid,1] - rr*dy))
    height, width = valid.shape
    im=ax.imshow(np.ma.masked_invalid(grid[:,:,2]), origin='lower',
        extent=(x0-dx/2,x0+(width-.5)*dx,y0-dy/2,y0+(height-.5)*dy),
        interpolation='nearest', cmap='viridis', vmin=1, vmax=12)
    shift = 0 if frame == 3 else 2.1
    centers=[(13,12),(24,12),(19,23)]
    for x,y in centers:
        ax.add_patch(Circle((x+shift,y),1.55,fill=False,color='#D55E00',
                            linewidth=1.8,linestyle='-' if frame==3 else '--'))
    ax.set_xlim(9+shift,31+shift)
    ax.set_ylim(9,25)
    ax.set_aspect('equal')
    ax.set_xlabel('Scanner X (mm)')
    ax.set_title(f"Scan {frame:04d}: {'rounded bosses present' if frame==3 else 'rounded bosses absent'}",fontsize=12,pad=10)
    ax.spines[['top','right']].set_visible(False)
    evidence.append({'scan':frame,'source':str(path.relative_to(ROOT)),
                     'source_sha256':scan['metadata']['source_sha256'],
                     'plot_extent_scanner_mm':list(ax.get_xlim())+list(ax.get_ylim()),
                     'marker_centres_scanner_mm':[(x+shift,y) for x,y in centers]})
axes[0].set_ylabel('Scanner Y (mm)')
cax=fig.add_axes([.915,.28,.018,.43]);fig.colorbar(im,cax=cax,label='Measured Z (mm)')
fig.suptitle('Harmu face identification from rounded features',fontsize=15,y=.96)
fig.text(.08,.135,'Solid circles mark rounded bosses in scan 0003; dashed circles mark corresponding areas in scan 0004.',fontsize=10)
fig.text(.08,.09,'Native height cells are shown without interpolation or filling. The two scans have different X offsets.',fontsize=10)
fig.text(.08,.045,'Feature evidence matches the photographed face to scan 0003; this choice does not use reconstruction error scores.',fontsize=10)
for ext in ('png','pdf'):
    fig.savefig(OUT/f'harmu_face_features.{ext}',dpi=250,facecolor='white')
(OUT/'face_evidence.json').write_text(json.dumps({'conclusion':'0003 matches photographed rounded-boss face; 0004 is opposite face','basis':'user confirms opposite faces and identifies missing rounded features on reverse; native scans visually inspected','records':evidence},indent=2),encoding='utf-8')
print(OUT/'harmu_face_features.png')

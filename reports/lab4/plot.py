from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

root = Path(__file__).resolve().parents[2]
data = root / 'build/lab4-tests'
out = root / 'reports/lab4'
figures = out / 'figures'
figures.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
for ax, name in zip(axes, ['CCD', 'FABR']):
    a = np.genfromtxt(data / f'trajectory_{name}.csv', delimiter=',', names=True)
    ax.scatter(a['tx'], a['tz'], s=16, c='#cad5df', label='Target')
    ax.scatter(a['x'], a['z'], s=3, c='#bb3131', label='End effector')
    ax.set(xlabel='x', ylabel='z', title=f'{name}: VCI, 490 target points', aspect='equal')
    ax.grid(alpha=.2)
    ax.legend(loc='lower left', fontsize=8)
fig.savefig(figures / 'ik-vci.png', dpi=180)
plt.close(fig)

springs = []
for i in range(11):
    for j in range(11):
        k = i * 11 + j
        if i > 0: springs.append((k, (i-1)*11+j))
        if i > 1: springs.append((k, (i-2)*11+j))
        if j > 0: springs.append((k, i*11+j-1))
        if j > 1: springs.append((k, i*11+j-2))
        if i > 0 and j > 0: springs.append((k, (i-1)*11+j-1))
        if i > 0 and j < 10: springs.append((k, (i-1)*11+j+1))
snapshots = []
for frame in [59, 179, 599]:
    a = np.genfromtxt(data / f'cloth_t{frame+1}.csv', delimiter=',', names=True)
    snapshots.append(np.column_stack([a['x'], a['z'], a['y']]))
all_points = np.concatenate(snapshots)
limits = [(all_points[:,i].min()-.15, all_points[:,i].max()+.15) for i in range(3)]
fig = plt.figure(figsize=(15, 4.5))
for panel, (frame, p) in enumerate(zip([59, 179, 599], snapshots), 1):
    ax = fig.add_subplot(1, 3, panel, projection='3d')
    ax.add_collection3d(Line3DCollection(p[np.array(springs)], colors='#3268a8', linewidths=.45, alpha=.7))
    ax.scatter(*p[[0,10]].T, color='#c4362e', s=24, label='Fixed')
    ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2], xlabel='x', ylabel='z', zlabel='y', title=f't = {(frame+1)/60:g} s')
    ax.view_init(elev=22, azim=-55)
    ax.set_box_aspect([b-a for a,b in limits])
    ax.set_zticks([0, .5, 1, 1.5])
    ax.tick_params(labelsize=8, pad=0)
fig.subplots_adjust(left=.01, right=.96, bottom=.1, top=.88, wspace=.18)
fig.savefig(figures / 'cloth.png', dpi=180, bbox_inches='tight')
plt.close(fig)

print('Created report figures.')

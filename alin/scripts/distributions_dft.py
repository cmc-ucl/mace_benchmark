import argparse
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score as r2, root_mean_squared_error as rmse

from  ase.io import read,write
from pathlib import Path
from ase.formula import Formula

def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

# Parse arguments:
parser = argparse.ArgumentParser(
    description="distributions"
)
parser.add_argument(
    "--xyz",
    help="input xyz file, output from some calculations",
)
parser.add_argument(
    "--model",
    default="mace_mp",
    help="model",
)
parser.add_argument(
    "--e0s",
    help="input xyz file, output from some calculations to recover e0s used",
)
parser.add_argument(
    "--title",
    help="title for the graph",
)
parser.add_argument("--save", help="File to save the plot", default=None)
args = parser.parse_args()
scale = 1.0e5
nbin = 51

sample_file = args.xyz
title = args.title
model = args.model
sample = read(sample_file,index=":")
e0s_file = args.e0s
p = Path(sample_file)

force_tag_dft = f"dft_forces"
e_tag_dft = f"dft_energy"
s_tag_dft = f"dft_stress"
desc=[]
desc_ps=[]
desc_pa=[]
desc_p={}

dft_forces = []
dft_e = []
dft_s = []
dft_e_pa = []

e0s={}
if e0s_file:
  for f in read(e0s_file,index=":"):
    if len(f) == 1:
       e0s[f[0].symbol] = f.info[e_tag_dft]

print(e0s)

_ = [ Path(f"{p.stem}-{x}.extxyz").unlink(missing_ok=True) for x in ['bad',"pruned"] ]

for i,f in enumerate(sample):
    n = len(f)
    if n == 1:
       write(f"{p.stem}-pruned.extxyz",f,append=True,write_info=True)
       continue
    symbols = f.symbols
    specs = set(symbols)
    ps = [ f'{model}_{x}_descriptor' for x in specs]
    ds = [ f.info[d]*scale for d in ps if d in f.info ]
    if len(ds)>0:
        desc_ps.append(ds)
        for d,m in zip(ps,specs):
            if m in desc_p:
              desc_p[m].append(f.info[d]*scale )
            else:
              desc_p[m] = [ f.info[d]*scale ]

    if f'{model}_descriptor' in f.info:
        desc.append(f.info[f'{model}_descriptor']*scale)
    if f'{model}_descriptors' in f.arrays:
        desc_pa.append(f.arrays[f'{model}_descriptors'].flatten()*scale)

    e = Formula(str(symbols)).count()
    e0_dft = 0.0
    for k in e:
        e0_dft += e[k]*e0s[k]
    me = (f.info[e_tag_dft]-e0_dft)/n

    dft_e_pa.append(me)
    dft_e.append(f.info[e_tag_dft])
    dft_forces.append(f.arrays[force_tag_dft].flatten())
    dft_s.append(f.info[s_tag_dft])

dft_forces = np.concatenate(dft_forces, axis=0)

dft_s = np.concatenate(dft_s, axis=0)

if desc_pa:
  desc_pa = np.concatenate(desc_pa, axis=0)
if desc_ps:
  desc_ps = np.concatenate(desc_ps, axis=0)

nr = 2
fs = 8

fig, axs = plt.subplots(nrows=nr, ncols=4, figsize=(16, fs))

if title:
  plt.suptitle(f"{title}")
else:
  plt.suptitle(f"DFT data/descriptors distribution - {p.stem}")

axs[0,0].set_title("Energy/atom distribution")
axs[0,0].hist(dft_e_pa,label="DFT",bins=nbin)
axs[0,0].set_xlabel("energy/atom [eV/Å]")
axs[0,0].set_ylabel("count")
axs[0,0].legend()

axs[0,1].set_title("Potential energy distribution")
axs[0,1].hist(dft_e, label="DFT",bins=nbin)
axs[0,1].set_xlabel("energy [eV]")
axs[0,1].set_ylabel("count")
axs[0,1].legend()

axs[0,2].set_title("Force components distribution")
axs[0,2].hist(dft_forces,label="DFT",bins=nbin)
axs[0,2].set_xlabel("force [eV/Å]")
axs[0,2].set_ylabel("count")
axs[0,2].legend()

axs[0,3].set_title("Stress (Voigt) components distribution")
axs[0,3].hist(dft_s,label="DFT",bins=nbin)
axs[0,3].set_xlabel("stress [eV/Å³]")
axs[0,3].set_ylabel("count")
axs[0,3].legend()

if desc:
    axs[1,0].set_title("Descriptors per system")
    axs[1,0].hist(desc,bins=nbin)
if len(desc_ps)>0:
    axs[1,1].set_title("Descriptors per species")
    for p in desc_p:
      axs[1,1].hist(desc_p[p],bins=nbin,label=p)
    axs[1,1].legend()
if len(desc_pa)>0:
    axs[1,2].set_title("Descriptors per atom")
    axs[1,2].hist(desc_pa,bins=nbin)

plt.tight_layout()
if args.save is None:
    plt.show()
else:
    plt.savefig(args.save)

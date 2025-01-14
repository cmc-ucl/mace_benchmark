# -*- coding: utf-8 -*-
# Author; alin m elena, alin@elena.re
# Contribs;
# Date: 14-09-2024
# ©alin m elena, GPL v3 https://www.gnu.org/licenses/gpl-3.0.en.html
from ase.io import read,write
from fpsample import fps_npdu_kdtree_sampling as sample
import numpy as np
from pathlib import Path
import argparse

def extract(data,inds):
    return [ data[i] for i in inds ]

def sampling(data, dims, n_samples):
   n = len(data)
   ldata = np.array(data)
   ldata.reshape(n,dims)
   return set(sample(ldata, n_samples=n_samples, start_idx=0))

def write_samples(frames,inds,f_s):
    for i in inds:
      write(f_s, frames[i], write_info=True, append=True)

parser = argparse.ArgumentParser(
                    prog='split data using descriptors',
                    description='labels data coming from dft and mace descriptors')

parser.add_argument("trajectory", type=str)
parser.add_argument("--scale",type=float,default=1.0e5)
parser.add_argument("--n_samples",type=int,default=10)
parser.add_argument("--config_types",type=str, nargs='+')
parser.add_argument("--pre", type=str)

args = parser.parse_args()

config_types = args.config_types
traj = Path(args.trajectory)
scale = args.scale
n_samples = args.n_samples
pre=args.pre
if pre != "":
    pre +="-"

a = read(traj,index=":")

train = Path(f"{pre}train.xyz")
valid = Path(f"{pre}valid.xyz")
test = Path(f"{pre}test.xyz")
_ = [ p.unlink(missing_ok=True) for p in [train,valid,test] ]

systems = set(sorted([x.info['system_name'] for x in a]))
types = set(sorted([x.info['config_type'] for x in a if 'config_type' in x.info ]))
print(f"{types=}")
print(f"{config_types=}")
print(f"create files: {train=}, {valid=} and {test=}")
stats = {}
for i,f in enumerate(a):
   if 'config_type' in f.info:
     key = (f.info['config_type'],f.info['system_name'])
   else:
     key = ('all',f.info['system_name'])
   if key in stats:
      stats[key] += [i]
   else:
      stats[key] = [i]
k = 0
for key in stats:
   run = key[0]
   if run in config_types or run == 'all':
       n = len(stats[key])
       if n > n_samples:
         Ds = 2
         ns = int(0.8*n_samples)
         specs = set(sorted(a[stats[key][0]].get_chemical_symbols()))
         De = len(specs)
         desc_per_spec = [ [ a[x].info[f"mace_mp_{s}_descriptor"]*scale for s in specs ] for x in stats[key] ]
         ind_spec = sampling(desc_per_spec,De,ns)
         if len(ind_spec) != ns:
           ns = len(ind_spec)
           nvt = ns//4
           k += n_samples - ns - nvt
         else:
           nvt = n_samples - ns
         print(key,n,ns,nvt)
         train_ind = extract(stats[key],list(ind_spec))
         if key[0] == 'geometry_optimisation':
             if stats[key][-1] not in train_ind:
                 train_ind.append(stats[key][-1])
                 nvt -= 1
         left = set(stats[key])-set(train_ind)

         desc_per_spec_vt = [ [ a[x].info[f"mace_mp_{s}_descriptor"]*scale for s in specs ] for x in left ]
         vt_spec = sampling(desc_per_spec_vt,De,nvt)
         vt_ind = extract(list(left),list(vt_spec))
         test_ind = vt_ind[0::2]
         valid_ind = vt_ind[1::2]
         write_samples(a,train_ind,train)
         write_samples(a,test_ind,test)
         write_samples(a,valid_ind,valid)
       else:
         train_ind = stats[key]
         write_samples(a,train_ind,train)
   else:
     print(run)
     train_ind = stats[key]
     write_samples(a,train_ind,train)
print(f"too similar structures {k=}")

"""Direct100epoch protocol. No20epoch selection; original v7 jobs remain reusable."""
import argparse,json
from pathlib import Path
import torch
import mixed_addon as addon
base=addon.base

def fixed_seeds(c):
    seeds=c['seeds']
    if len(seeds)!=1:raise ValueError('Direct protocol requires one declared seed; use separate studies for repeats')
    return {arch:int(seeds[0]) for arch in c['architectures']}

def phase_jobs(c,phase):
    if phase=='select':return []
    if phase!='paper':return []
    jobs=[]
    for arch,seed in fixed_seeds(c).items():
        methods=['real','Polyp-DDPM','mixed']
        methods += [m for m in ['LDM','SinGAN-Seg'] if c['baseline_pairs'].get(m)]
        if c['baseline_pairs'].get('TextureBackground'):methods+=['TextureBackground','mixed_repaired']
        jobs.extend({'phase':'paper','arch':arch,'seed':seed,'method':m,'epochs':c['paper_epochs']} for m in methods)
    return jobs

base.select_seeds=fixed_seeds
base.phase_jobs=phase_jobs
prepare=addon.prepare
write=base.write
tables=base.tables
select_seeds=fixed_seeds
key=base.key

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--phase',choices=['paper','quality'],required=True)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1)
    a=p.parse_args();c=json.loads(Path(a.config).read_text())
    if base.digest(base.__file__)!=c['runner_hash']:raise ValueError('Base runner changed')
    if c['paper_epochs']!=100 and not c['parent'].get('cpu_fixture'):raise ValueError('Paper specifies100epochs')
    device=torch.device('cpu' if c['parent'].get('cpu_fixture') else f'cuda:{a.rank}')
    if device.type=='cuda':torch.cuda.set_device(device)
    torch.set_num_threads(2)
    from reference_io import install
    install(c)
    try:
        if a.phase=='quality':addon.quality(c,device)
        else:base.train_phase(c,'paper',a.rank,a.world,device)
    except Exception as exc:
        base.state(c,f'worker_{a.phase}_{a.rank}','ERROR',error=repr(exc));raise

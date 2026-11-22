"""Add repaired-bank arms without changing the original v7 training identities."""
import argparse,copy,json,sys
from pathlib import Path
import torch
import pandas as pd
import runner as base

_samples=base.samples
_jobs=base.phase_jobs

def samples(c,method,ids):
    if method=='mixed_repaired':
        return _samples(c,'real',ids)+_samples(c,'TextureBackground',ids)
    return _samples(c,method,ids)

def phase_jobs(c,phase):
    jobs=_jobs(c,phase)
    if phase=='paper' and c.get('baseline_pairs',{}).get('TextureBackground'):
        for arch,seed in base.select_seeds(c).items():
            jobs.extend({'phase':'paper','arch':arch,'seed':seed,'method':method,'epochs':c['paper_epochs']}
                        for method in ['TextureBackground','mixed_repaired'])
    return jobs

base.samples=samples
base.phase_jobs=phase_jobs

def prepare(c):
    c=base.prepare(c)
    rows=base.read_pairs({'pairs_csv':c.get('repaired_pairs_csv')},c['parent']['train_ids'])
    if rows:
        # Repaired images originate from precisely the current training masks.
        if {r['id'] for r in rows}!=set(c['parent']['train_ids']):
            raise ValueError('Repaired sample_id must match the current train900 IDs')
        native={r['id']:r for r in c['parent']['records']}
        for row in rows:
            if base.digest(row['mask'])!=native[row['id']]['mask_sha256']:
                raise ValueError('Repaired mask differs from original condition mask: '+row['id'])
        tests={r['image_sha256'] for r in c['parent']['records'] if r['id'] in c['parent']['test_ids']}
        if any(base.digest(r['image']) in tests for r in rows):raise ValueError('Test image in repaired bank')
    c['baseline_pairs']['TextureBackground']=rows
    if rows:
        bank=[{'id':r['id'],'image':base.digest(r['image']),'mask':base.digest(r['mask'])} for r in rows]
        c['method_hash']['TextureBackground']=base.ident({'bank':bank,'adapter':base.digest(__file__)})
        c['method_hash']['mixed_repaired']=base.ident([c['method_hash']['real'],c['method_hash']['TextureBackground']])
    return c

# Expose unchanged base helpers to notebook cells.
write=base.write
tables=base.tables
select_seeds=base.select_seeds
key=base.key

def quality(c,device):
    base.quality(c,device)
    rows=c.get('baseline_pairs',{}).get('TextureBackground',[])
    if not rows:return
    import extension
    qc=copy.deepcopy(c);qc['out']=c['metric_cache_out'];qc['metric_n']=len(rows)
    qc['code_sha256']=c['metrics_frozen_hash']
    native={r['id']:r for r in c['parent']['records']}
    value=extension.metrics_pair(qc,'repaired_vs_matched_real',
        [r['image'] for r in rows],[native[r['id']]['image'] for r in rows],device)
    path=Path(c['out'])/'image_ablation.csv'
    old=pd.read_csv(path)
    row={'source':'texture_background','reference':'matched real train IDs','available':len(rows),
         **{k:v for k,v in value.items() if k!='spec'}}
    pd.concat([old,pd.DataFrame([row])],ignore_index=True).to_csv(path,index=False)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--phase',choices=['select','paper','quality'],required=True)
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--world',type=int,default=1)
    a=parser.parse_args();c=json.loads(Path(a.config).read_text())
    if base.digest(base.__file__)!=c['runner_hash']:raise ValueError('Base runner changed')
    device=torch.device('cpu' if c['parent'].get('cpu_fixture') else f'cuda:{a.rank}')
    if device.type=='cuda':torch.cuda.set_device(device)
    torch.set_num_threads(2)
    try:
        if a.phase=='quality':quality(c,device)
        else:base.train_phase(c,a.phase,a.rank,a.world,device)
    except Exception as exc:
        base.state(c,f'worker_{a.phase}_{a.rank}','ERROR',error=repr(exc));raise

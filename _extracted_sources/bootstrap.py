"""Resolve read-only master data and independent evaluation checkpoints."""
import copy,json,shutil
from pathlib import Path
from runner import digest,write
import extension

def resolve_master(master_root, input_root):
    if master_root:
        root=Path(master_root)
        if not (root/'run_config.json').is_file():raise FileNotFoundError(root/'run_config.json')
        return root
    candidates=[]
    for path in Path(input_root).rglob('run_config.json'):
        if (path.parent/'author/images').is_dir():candidates.append(path.parent)
    if len(candidates)!=1:
        raise ValueError('Set MASTER_ROOT to one attached master output. Candidates: '+str([str(p) for p in candidates]))
    return candidates[0]

def load_parent(root, input_root, images=None, masks=None):
    root=Path(root);parent=json.loads((root/'run_config.json').read_text());parent['out']=str(root)
    ids={r['id'] for r in parent['records']}
    if not images:
        matches=[]
        for path in Path(input_root).rglob('images'):
            if (path.parent/'masks').is_dir() and 'kvasir' in str(path).lower():
                found=extension.images(path)
                if set(found)==ids:matches.append(path)
        if len(matches)!=1:raise ValueError('Set REAL_IMAGES and REAL_MASKS. Candidate real datasets: '+str(matches))
        images=str(matches[0]);masks=str(matches[0].parent/'masks')
    im=extension.images(images);ma=extension.images(masks)
    if set(im)!=ids or set(ma)!=ids:raise ValueError('Dataset differs from original master IDs')
    for row in parent['records']:
        row['image']=im[row['id']];row['mask']=ma[row['id']]
        for kind in ['image','mask']:
            expected=row.get(kind+'_sha256')
            if expected and digest(row[kind])!=expected:raise ValueError('Changed original data: '+row['id']+' '+kind)
    for role,history in parent.get('histories',{}).items():
        if not history:continue
        original=Path(history['root'])
        if original.is_dir():continue
        matches=[p for p in Path(input_root).rglob(original.name) if p.is_dir()]
        if len(matches)==1:history['root']=str(matches[0])
        else:parent['histories'][role]=None # Optional ablation unavailable; core remains runnable.
    # Do not construct the generator or optimizer in the notebook process.
    checkpoint=Path(parent['author_checkpoint'])
    if not checkpoint.is_file():
        matches=[p for p in Path(input_root).rglob(checkpoint.name) if p.is_file()]
        matches=[p for p in matches if digest(p)==parent['author_sha256']]
        parent['author_checkpoint']=str(matches[0]) if len(matches)==1 else ''
    return parent

def recover_output(out, resume, input_root):
    out=Path(out)
    if (out/'config.json').exists():return 'local'
    if not resume:
        candidates=[p.parent for p in Path(input_root).rglob('config.json') if p.parent.name=='polyp_ddpm_evaluation_v8']
        if len(candidates)>1:raise ValueError('Multiple evaluation outputs. Set RESUME_EVALUATION: '+str(candidates))
        resume=str(candidates[0]) if candidates else None
    if resume:
        src=Path(resume)
        previous=json.loads((src/'config.json').read_text())
        if 'preprocessing_mode' not in previous:raise ValueError('Cannot resume v7 training into v8; clear RESUME_EVALUATION')
        if not (src/'config.json').is_file():raise FileNotFoundError(src/'config.json')
        shutil.copytree(src,out,dirs_exist_ok=True,ignore=shutil.ignore_patterns('runtime','__pycache__'))
    out.mkdir(parents=True,exist_ok=True)
    return resume or 'new'

def recover_metric_cache(root, out):
    dest=Path(out)/'metrics'
    for src in [Path(root)/'evaluation_v5',Path(root)/'paper_tables_v6/metrics']:
        for name in ['quality','quality_extra']:
            if (src/name).is_dir() and not (dest/name).exists():
                shutil.copytree(src/name,dest/name)

def discover_external(specs,input_root,master_root):
    result=copy.deepcopy(specs)
    old=Path(master_root)/'evaluation_v5/study_config.json'
    saved=json.loads(old.read_text()).get('external',{}) if old.exists() else {}
    for name,spec in result.items():
        if spec.get('images'):continue
        previous=saved.get(name,{})
        if previous.get('images') and Path(previous['images']).is_dir():result[name]=previous;continue
        token='hyper' if name=='HyperKvasir' else 'etis'
        candidates=[p for p in Path(input_root).rglob('images') if token in str(p).lower() and (p.parent/'masks').is_dir()]
        if len(candidates)==1:
            spec.update(images=str(candidates[0]),masks=str(candidates[0].parent/'masks'))
    return result

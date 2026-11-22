"""New evaluation study. Original master artifacts are immutable inputs."""
import os,sys,json,time,hashlib,copy,argparse,gc
from pathlib import Path
import numpy as np,pandas as pd,torch
from PIL import Image

def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));os.replace(t,p)
def images(folder):
 if not folder or not Path(folder).is_dir():return {}
 result={}
 for p in sorted(Path(folder).rglob('*')):
  if p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp','.tif','.tiff']:
   if p.stem in result:raise ValueError('Duplicate image ID: '+p.stem)
   result[p.stem]=str(p)
 return result

def external(spec):
 im=images(spec.get('images'));ma=images(spec.get('masks'))
 if not im or not ma:return []
 prefix=spec.get('mask_prefix','')
 if prefix:ma={k[len(prefix):] if k.startswith(prefix) else k:v for k,v in ma.items()}
 if set(im)!=set(ma):raise ValueError('External image/mask IDs differ. Configure paired leaf directories.')
 expected=spec.get('expected_n')
 if expected and len(im)!=expected:raise ValueError(f'Expected {expected} annotated external pairs, found {len(im)}')
 for sid in im:
  with Image.open(im[sid]) as a,Image.open(ma[sid]) as b:
   if a.size!=b.size:raise ValueError('External image/mask native dimensions differ: '+sid)
 return [{'id':i,'image':im[i],'mask':ma[i]} for i in sorted(im)]

def file_identity(paths):return [{'name':Path(p).name,'sha256':sha(p)} for p in paths]

def resolve(c):
 """Record exact input identities once per notebook launch, not each GPU batch."""
 p=c['parent'];root=Path(p['out']);records={r['id']:r for r in p['records']}
 banks={'deployment':{i:str(root/'author/images'/(i+'.png')) for i in p['train_ids'] if (root/'author/images'/(i+'.png')).exists()}}
 h=p['histories'].get('color')
 if h:
  color=Path(h['root']);banks.update(own_raw=images(color/'color/raw_bank'),color=images(color/'synthetic/images'))
 for role,key,folder in [('v2','background','pilot/tuned_120'),('v21','boundary','pilot/trained_guard')]:
  h=p['histories'].get(role)
  if h:
   banks[key]={f.stem.split('__r')[0]:str(f) for f in sorted((Path(h['root'])/folder).glob('*__r0.png'))}
 for name,bank in banks.items():
  if not set(bank)<=set(p['train_ids']):raise ValueError('Generation bank includes non-training IDs: '+name)
 c['banks']=banks;c['bank_hashes']={k:identity(file_identity(list(v.values()))) for k,v in banks.items()}
 c['external_records']={name:external(spec) for name,spec in c['external'].items()}
 for name,rs in c['external_records'].items():
  c.setdefault('external_hashes',{})[name]=identity([{'id':r['id'],'image':sha(r['image']),'mask':sha(r['mask'])} for r in rs])
 # Exact duplicates are flagged, never silently described as independent external cases.
 training_hashes={r['image_sha256'] for r in p['records'] if r['id'] in p['train_ids']}
 c['external_overlap']={name:sum(sha(r['image']) in training_hashes for r in rs) for name,rs in c['external_records'].items()}
 c['code_sha256']=sha(__file__)
 write(Path(c['out'])/'input_inventory.json',{'banks':{k:len(v) for k,v in banks.items()},'external':{k:len(v) for k,v in c['external_records'].items()},'exact_train_overlap':c['external_overlap']})
 return c

def jobs(c):
 result=[]
 for arch in c['architectures']:
  for seed in c['seeds']:
   for name,ratio,factor in [('real',0,1),('synthetic',1,1),('mixed25',.25,1),('mixed50',.5,1),('real_long',0,2),('mixed50_long',.5,2)]:
    result.append({'architecture':arch,'seed':seed,'arm':name,'synthetic_ratio':ratio,'steps':c['base_steps']*factor})
 return result

def job_key(j):return f"{j['architecture']}_s{j['seed']}_{j['arm']}"
def job_identity(c,j):return identity({'job':j,'bank':c['bank_hashes']['deployment'],'split':c['parent']['split'],'batch':c['parent']['seg_batch'],'lr':c['parent']['seg_lr'],'encoder':c['parent']['encoder_weights'],'code':c['code_sha256']})
def model(c,arch,device,pretrained=False):
 if c['parent'].get('cpu_fixture'):
  return torch.nn.Sequential(torch.nn.Conv2d(3,8,3,padding=1),torch.nn.ReLU(),torch.nn.Conv2d(8,1,1)).to(device)
 import segmentation_models_pytorch as smp
 cls={'unetpp':smp.UnetPlusPlus,'fpn':smp.FPN,'deeplabv3plus':smp.DeepLabV3Plus}[arch]
 return cls(encoder_name='resnet34',encoder_weights=c['parent']['encoder_weights'] if pretrained else None,in_channels=3,classes=1).to(device)

def legacy_checkpoint(c,j):
 if j['architecture']=='unetpp' and j['seed']==c['parent']['seed'] and j['steps']==c['parent']['seg_steps'] and j['arm'] in ['real','synthetic','mixed50']:
  arm={'mixed50':'mixed'}.get(j['arm'],j['arm']);path=Path(c['parent']['out'])/'segmentation'/(arm+'.pt')
  if path.exists():return path
 return None

def initialization(c,arch,seed,device):
 from common import load_pt,save_pt,fixed_seed
 path=Path(c['out'])/'initial'/f'{arch}_{seed}.pt'
 if path.exists():return load_pt(path)['model']
 old=Path(c['parent']['out'])/'segmentation/initial.pt'
 if arch=='unetpp' and seed==c['parent']['seed'] and old.exists():state=load_pt(old)['model']
 else:
  with fixed_seed(seed,device):
   net=model(c,arch,device,True);state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};del net
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');torch.save({'model':state,'architecture':arch,'seed':seed,'encoder_weights':c['parent']['encoder_weights']},tmp);os.replace(tmp,path)
 return state

def train(c,rank,world,device):
 from common import load_pt,save_pt,seed_all,stream_ids,tensor_pair,normalize_segmenter,dice_loss,rng_state,restore_rng,resource_stop
 p=c['parent'];records={r['id']:r for r in p['records']};batch=p['seg_batch']
 if batch%4:raise ValueError('Study batch must be divisible by4 for exact25% and50% mixtures')
 if len(c['banks']['deployment'])!=len(p['train_ids']):raise ValueError('Complete deployment bank required before seed study')
 for j in jobs(c)[rank::world]:
  if resource_stop(c,device):break
  key=job_key(j);protocol=job_identity(c,j);path=Path(c['out'])/'checkpoints'/(key+'.pt')
  old=legacy_checkpoint(c,j)
  if old:
   cp=load_pt(old)
   if cp['protocol']!=p['protocol'] or cp.get('bank_sha256')!=sha(Path(p['out'])/'author/manifest.json'):raise ValueError('Wrong legacy checkpoint or bank')
   if cp['step']==j['steps']:
    del cp;continue
   del cp
  step=0;history=[]
  if path.exists():
   cp=load_pt(path)
   if cp['protocol']!=protocol:raise ValueError('Study recipe changed. Use a new STUDY_DIR.')
   step=cp['step'];del cp
   if step==j['steps']:continue
  seed_all(j['seed']);net=model(c,j['architecture'],device)
  net.load_state_dict(initialization(c,j['architecture'],j['seed'],device));opt=torch.optim.Adam(net.parameters(),lr=p['seg_lr'])
  if path.exists():
   cp=load_pt(path);net.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);step=cp['step'];history=cp['history'];restore_rng(cp['rng'],device);del cp
  loaded=step
  while step<j['steps'] and not resource_stop(c,device):
   positions=list(range(step*batch,(step+1)*batch));ids=stream_ids(p['train_ids'],positions,j['seed']+1000)
   # Preserve old parity recipe for50%; fixed25% slots are an explicit new treatment.
   synthetic=[j['synthetic_ratio']==1 or (j['synthetic_ratio']==.5 and i%2==1) or (j['synthetic_ratio']==.25 and i%4==3) for i in range(batch)]
   pairs=[tensor_pair(records[sid],p['size'],j['seed']+100000+pos,c['banks']['deployment'][sid] if syn else None) for sid,pos,syn in zip(ids,positions,synthetic)]
   x=torch.stack([a for a,b in pairs]).to(device);y=torch.stack([b for a,b in pairs]).to(device)
   net.train();opt.zero_grad(set_to_none=True);loss=dice_loss(net(normalize_segmenter(x)),y)
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite study loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step();step+=1
   history.append({'step':step,'loss':float(loss.detach())});del x,y,loss
   if step%50==0:print(f'{key}: {step}/{j["steps"]}, loss={history[-1]["loss"]:.5f}',flush=True)
   if step%p['save_every']==0 or step==j['steps']:
    save_pt(path,{'protocol':protocol,'job':j,'model':net.state_dict(),'optimizer':opt.state_dict(),'step':step,'history':history,'rng':rng_state(device)})
  if step>loaded or not path.exists():save_pt(path,{'protocol':protocol,'job':j,'model':net.state_dict(),'optimizer':opt.state_dict(),'step':step,'history':history,'rng':rng_state(device)})
  del net,opt;gc.collect()
  if device.type=='cuda':torch.cuda.empty_cache()

def summarize(d):
 tp,fp,fn,tn=d[['tp','fp','fn','tn']].sum()
 return {'dataset_iou':float(tp/max(tp+fp+fn,1)),'dataset_dice':float(2*tp/max(2*tp+fp+fn,1)),
 'accuracy':float((tp+tn)/max(tp+tn+fp+fn,1)),'precision':float(tp/max(tp+fp,1)),
 'mean_image_iou':float(d.iou.mean()),'mean_image_dice':float(d.dice.mean()),'boundary_f1':float(d.boundary_f1.mean()),'n':len(d)}

@torch.no_grad()
def evaluate(c,rank,world,device):
 from common import load_pt,tensor_pair,normalize_segmenter,binary_metrics,resource_stop
 p=c['parent'];r={x['id']:x for x in p['records']};datasets={'Kvasir_Test100':[r[i] for i in p['test_ids']],**{n:rs for n,rs in c['external_records'].items() if rs}}
 for j in jobs(c)[rank::world]:
  key=job_key(j);path=Path(c['out'])/'checkpoints'/(key+'.pt');old=legacy_checkpoint(c,j)
  if not path.exists() and old:path=old
  if not path.exists():continue
  cp=load_pt(path)
  if cp['step']!=j['steps']:del cp;continue
  expected=p['protocol'] if path==old else job_identity(c,j)
  if cp.get('protocol')!=expected:raise ValueError('Wrong evaluation checkpoint')
  if path==old and cp.get('bank_sha256')!=sha(Path(p['out'])/'author/manifest.json'):raise ValueError('Legacy bank changed')
  net=model(c,j['architecture'],device);net.load_state_dict(cp['model']);net.eval();del cp
  for name,rows in datasets.items():
   if resource_stop(c,device):break
   fingerprint=identity({'checkpoint':sha(path),'data':file_identity([x[k] for x in rows for k in ['image','mask']]),'threshold':.5,'size':p['size'],'code':c['code_sha256']})
   out=Path(c['out'])/'evaluation'/key/name;meta=out/'summary.json'
   if meta.exists() and json.loads(meta.read_text()).get('identity')==fingerprint:continue
   out.mkdir(parents=True,exist_ok=True);result=[]
   for i in range(0,len(rows),p['eval_batch']):
    if resource_stop(c,device):break
    group=rows[i:i+p['eval_batch']];pairs=[tensor_pair(x,p['size']) for x in group]
    x=torch.stack([a for a,b in pairs]).to(device);probs=net(normalize_segmenter(x)).sigmoid().cpu().numpy()[:,0]
    for row,prob,(_,mask) in zip(group,probs,pairs):
     result.append({'id':row['id'],**binary_metrics(prob,mask[0].numpy())})
     if len(result)<=c['panel_cases']:np.save(out/(row['id']+'.npy'),prob)
   if len(result)==len(rows):
    d=pd.DataFrame(result);d.to_csv(out/'per_image.csv',index=False);write(meta,{'identity':fingerprint,'job':j,'dataset':name,'metrics':summarize(d),'exact_train_overlap':c.get('external_overlap',{}).get(name,0)})
   # No partial score is presented as a completed dataset.
  del net;gc.collect()

class MetricImages(torch.utils.data.Dataset):
 def __init__(self,paths):self.paths=list(paths)
 def __len__(self):return len(self.paths)
 def __getitem__(self,index):
  with Image.open(self.paths[index]) as im:a=np.array(im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR),copy=True)
  return torch.from_numpy(a.transpose(2,0,1))

def metrics_pair(c,name,fake,real,device):
 n=min(c['metric_n'],len(fake),len(real))
 if n<50:return {'status':'insufficient_sample_pilot','n':n}
 import importlib.metadata
 from common import resource_stop,fixed_seed
 from torchmetrics.image.fid import FrechetInceptionDistance
 from torchmetrics.image.kid import KernelInceptionDistance
 from torchmetrics.image.inception import InceptionScore
 n=min(c['metric_n'],len(fake),len(real));fake=list(fake)[:n];real=list(real)[:n]
 if n<50:return {'status':'insufficient_sample_pilot','n':n}
 spec={'fake':file_identity(fake),'real':file_identity(real),'resize':256,'feature':2048,'seed':c['metric_seed'],'kid_subsets':50,'kid_subset':min(100,n),'IS_splits':10,'torchmetrics':importlib.metadata.version('torchmetrics'),'torch_fidelity':importlib.metadata.version('torch-fidelity'),'code':c['code_sha256']}
 fingerprint=identity(spec);path=Path(c['out'])/'quality'/(name+'_'+fingerprint[:16]+'.json')
 if path.exists():return json.loads(path.read_text())
 result={'status':'complete','n_real':n,'n_fake':n,'identity':fingerprint,'IS_scope':'Generated samples only; independent of reference distribution','paper_1000_count_matched':n==1000,'spec':spec}
 for metric_name in ['FID','KID','IS']:
  if resource_stop(c,device):return {'status':'pending_session_guard','n':n}
  metric=(FrechetInceptionDistance(feature=2048,normalize=False).set_dtype(torch.float64) if metric_name=='FID' else KernelInceptionDistance(feature=2048,subsets=50,subset_size=min(100,n),normalize=False) if metric_name=='KID' else InceptionScore(splits=10,normalize=False)).to(device)
  with fixed_seed(c['metric_seed'],device):
   for kind,files in [('fake',fake),('real',real)]:
    if metric_name=='IS' and kind=='real':continue
    for batch in torch.utils.data.DataLoader(MetricImages(files),batch_size=16,num_workers=0):
     if resource_stop(c,device):del metric;return {'status':'pending_session_guard','n':n}
     if metric_name=='IS':metric.update(batch.to(device))
     else:metric.update(batch.to(device),real=kind=='real')
   value=metric.compute()
  if metric_name=='FID':result['FID']=float(value)
  else:result[metric_name+'_mean']=float(value[0]);result[metric_name+'_std']=float(value[1])
  del metric;gc.collect()
  if device.type=='cuda':torch.cuda.empty_cache()
 write(path,result);return result

@torch.no_grad()
def extra_quality_images(c,device):
 from common import load_pt,factory,png,full_reverse,seed_for,save_png,resource_stop
 p=c['parent'];root=Path(c['out'])/'quality_extra';root.mkdir(exist_ok=True,parents=True)
 if sha(p['author_checkpoint'])!=p['author_sha256']:raise ValueError('Deployment generator changed')
 diff=factory(p,device);cp=load_pt(p['author_checkpoint']);diff.load_state_dict(cp['ema']);del cp;diff.eval().requires_grad_(False)
 records={r['id']:r for r in p['records']}
 for sid in p['train_ids'][:100]:
  if resource_stop(c,device):break
  out=root/(sid+'.png');meta=out.with_suffix('.json');seed=seed_for('quality_extra:'+sid,0)
  if out.exists() and meta.exists():
   d=json.loads(meta.read_text())
   if d['generator']==p['author_sha256'] and d['seed']==seed and d['sha256']==sha(out):continue
   raise ValueError('Quality extra image changed')
  cond=png(records[sid]['mask'],p['size'],True).repeat(3,1,1)[None].to(device)*2-1
  image,_=full_reverse(diff,cond,seed);save_png(image[0],out);write(meta,{'id':sid,'seed':seed,'generator':p['author_sha256'],'sha256':sha(out),'purpose':'Image metrics only; not segmentation bank'})
 del diff

def quality(c,device):
 from common import resource_stop
 p=c['parent'];rec={r['id']:r for r in p['records']}
 if c['generate_100_extra']:extra_quality_images(c,device)
 fake=list(c['banks']['deployment'].values())
 extra=Path(c['out'])/'quality_extra'
 for out in sorted(extra.glob('*.png')):
  meta=out.with_suffix('.json')
  if meta.exists():
   d=json.loads(meta.read_text())
   if d['generator']!=p['author_sha256'] or d['sha256']!=sha(out):raise ValueError('Invalid quality extra')
   fake.append(str(out))
 if len(c['banks']['deployment'])!=len(p['train_ids']):raise ValueError('Incomplete deployment bank')
 order=np.random.default_rng(c['metric_seed']).permutation(len(p['records']))
 refs={'Kvasir_SEG':[p['records'][int(i)]['image'] for i in order]}
 if c['external_records'].get('HyperKvasir'):refs['HyperKvasir']=[r['image'] for r in c['external_records']['HyperKvasir']]
 rows=[]
 for name,real in refs.items():
  if resource_stop(c,device):break
  value=metrics_pair(c,'deployment_vs_'+name,fake,real,device);rows.append({'source':'deployment','reference':name,**{k:v for k,v in value.items() if k!='spec'}})
 # Paired source ablation on the SAME IDs. Refiner pilots are not full900 banks.
 if c['image_ablation']:
  for name in ['own_raw','color','background','boundary']:
   bank=c['banks'].get(name,{})
   ids=sorted(set(bank)&set(c['banks']['deployment']))
   if not ids:continue
   value=metrics_pair(c,'ablation_'+name,[bank[i] for i in ids],[rec[i]['image'] for i in ids],device)
   rows.append({'source':name,'reference':'matched real train IDs','available':len(ids),**{k:v for k,v in value.items() if k!='spec'}})
 Path(c['out']).mkdir(exist_ok=True,parents=True);pd.DataFrame(rows).to_csv(Path(c['out'])/'image_metrics.csv',index=False)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--phase',choices=['train','evaluate','quality'],required=True);a=parser.parse_args()
 c=json.loads(Path(a.config).read_text());sys.path.insert(0,c['runtime'])
 if sha(__file__)!=c['code_sha256']:raise ValueError('Extension source changed')
 rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
 device=torch.device('cpu' if c['parent'].get('cpu_fixture') else f'cuda:{local}')
 if device.type=='cuda':torch.cuda.set_device(device)
 torch.set_num_threads(2)
 {'train':lambda:train(c,rank,world,device),'evaluate':lambda:evaluate(c,rank,world,device),'quality':lambda:quality(c,device)}[a.phase]()

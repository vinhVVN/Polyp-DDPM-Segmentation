"""Epoch-based paper-table runner. No inherited time limit, no test-selected seed."""
import os,sys,time,json,math,hashlib,copy,gc,argparse
from pathlib import Path
import numpy as np,pandas as pd,torch
from PIL import Image
import common
from memory_guard import resource_stop
common.resource_stop = resource_stop

def ident(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(x,indent=2));os.replace(tmp,p)
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.{os.getpid()}.tmp');torch.save(x,tmp);os.replace(tmp,p)
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def state(c,key,status,**kwargs):
 write(Path(c['out'])/'status'/(key+'.json'),{'key':key,'status':status,'time':time.time(),**kwargs})

def samples(c,method,ids):
 r={x['id']:x for x in c['parent']['records']}
 if method=='real':return [r[i] for i in ids]
 if method=='mixed':return samples(c,'real',ids)+samples(c,'Polyp-DDPM',ids)
 if method=='Polyp-DDPM':
  return [{'id':i,'image':c['banks']['deployment'][i],'mask':r[i]['mask']} for i in ids]
 rows=c['baseline_pairs'].get(method,[])
 if set(ids)!=set(c['parent']['train_ids']):
  raise ValueError('A native baseline bank is supported for final training only')
 return rows

def read_pairs(spec,train_ids):
 path=spec.get('pairs_csv')
 if not path:return []
 path=Path(path);df=pd.read_csv(path,dtype=str)
 for field in ['image','mask']:
  if field not in df:raise ValueError('Baseline CSV requires image,mask columns')
 id_column=next((name for name in ['sample_id','id','condition_id'] if name in df),None)
 if not id_column:raise ValueError('CSV requires a native sample_id column')
 if len(df)!=len(train_ids) or df[id_column].isna().any() or df[id_column].duplicated().any():
  raise ValueError('Baseline requires exactly900 unique native image-mask pairs')
 rows=[]
 for record in df.to_dict('records'):
  record['id']=record[id_column]
  for field in ['image','mask']:
   p=Path(record[field]);p=p if p.is_absolute() else path.parent/p
   if not p.is_file():raise FileNotFoundError(p)
   record[field]=str(p)
  with Image.open(record['image']) as im,Image.open(record['mask']) as mask:
   if im.size!=mask.size:raise ValueError('Baseline image/mask dimensions differ: '+record['id'])
  rows.append(record)
 return rows

def prepare(c):
 import extension as old
 from reference_io import install
 install(c)
 c['deadline']=4102444800.0 # No user-hour timer; physical Kaggle termination still applies.
 c=old.resolve(c)
 c.setdefault('external_hashes',{})
 c['baseline_pairs']={name:read_pairs(spec,c['parent']['train_ids']) for name,spec in c['baselines'].items()}
 test_hashes={r['image_sha256'] for r in c['parent']['records'] if r['id'] in c['parent']['test_ids']}
 for method,rows in c['baseline_pairs'].items():
  if any(digest(r['image']) in test_hashes for r in rows):raise ValueError('Real test image found in baseline training bank: '+method)
 ids=sorted(c['parent']['train_ids']);order=np.random.default_rng(c['validation_seed']).permutation(len(ids));n=c['validation_n']
 c['validation_ids']=[ids[int(i)] for i in order[:n]];c['fit_ids']=[ids[int(i)] for i in order[n:]]
 c['runner_hash']=digest(__file__)
 c['method_hash']={'real':ident([r['image_sha256'] for r in c['parent']['records'] if r['id'] in ids]),'Polyp-DDPM':c['bank_hashes']['deployment']}
 for method,rows in c['baseline_pairs'].items():c['method_hash'][method]=ident([{'id':r['id'],'image':digest(r['image']),'mask':digest(r['mask'])} for r in rows])
 c['method_hash']['mixed']=ident([c['method_hash']['real'],c['method_hash']['Polyp-DDPM']])
 c['metrics_frozen_hash']=digest(Path(c['metrics_runtime'])/'extension.py')
 if len(c['banks']['deployment'])!=len(ids):raise ValueError('Deployment bank incomplete: run final inference first')
 return c

def phase_jobs(c,phase):
 if phase=='select':return [{'phase':'select','arch':a,'seed':s,'method':'mixed','epochs':c['selection_epochs']} for a in c['architectures'] for s in c['seeds']]
 choices=select_seeds(c)
 return [{'phase':'paper','arch':a,'seed':choices[a],'method':m,'epochs':c['paper_epochs']} for a in c['architectures'] if a in choices for m in ['real','Polyp-DDPM','LDM','SinGAN-Seg','mixed'] if samples(c,m,c['parent']['train_ids'])]

def key(j):return f"{j['phase']}_{j['arch']}_s{j['seed']}_{j['method']}"
def recipe(c,j):return ident({'job':j,'data':c['method_hash'][j['method']],'split':c['parent']['split'],'fit':c['fit_ids'] if j['phase']=='select' else None,'validation':c['validation_ids'] if j['phase']=='select' else None,'batch':c['parent']['seg_batch'],'lr':c['parent']['seg_lr'],'size':c['parent']['size'],'encoder':c['parent']['encoder_weights'],'memory_guard':digest(Path(c['metrics_runtime'])/'memory_guard.py'),'runner':c['runner_hash'],'amp':False,'preprocessing':c['preprocessing_mode'],'checkpoint_policy':c['checkpoint_policy'],'io_hash':digest(Path(c['runtime'])/'reference_io.py'),'augmentation_hash':digest(Path(c['runtime'])/'reference_augmentation.py'), 'smp':c['software']['segmentation_models_pytorch'],'albu':c['software']['albumentations']})

def make_model(c,arch,device,pretrained=False):
 import extension as old
 return old.model(c,arch,device,pretrained)

def initial(c,j,device):
 from common import fixed_seed
 path=Path(c['out'])/'initial'/f"{j['arch']}_{j['seed']}.pt"
 if path.exists():return load(path)['model']
 with fixed_seed(j['seed'],device):
  model=make_model(c,j['arch'],device,True);weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};del model
 save(path,{'model':weights,'seed':j['seed'],'arch':j['arch']})
 return weights

def memory_stop(c,device):
 from common import resource_stop
 return resource_stop(c,device) # Only RAM/cgroup checks; deadline intentionally not inherited.

@torch.no_grad()
def predict(c,model,rows,device,folder=None,can_pause=True):
 from common import tensor_pair,normalize_segmenter,binary_metrics
 model.eval();result=[];b=c['parent']['eval_batch']
 for start in range(0,len(rows),b):
  if can_pause and memory_stop(c,device):return None
  group=rows[start:start+b];pairs=[tensor_pair(r,c['parent']['size']) for r in group]
  x=torch.stack([a for a,y in pairs]).to(device);probs=model(normalize_segmenter(x)).sigmoid().cpu().numpy()[:,0]
  for r,prob,(_,mask) in zip(group,probs,pairs):
   result.append({'id':r['id'],**binary_metrics(prob,mask[0].numpy())})
   if folder and len(result)<=3:np.save(Path(folder)/(r['id']+'.npy'),prob)
 return pd.DataFrame(result)

def run_job(c,j,device):
 from common import seed_all,rng_state,restore_rng,tensor_pair,normalize_segmenter,dice_loss
 import extension as old
 name=key(j);root=Path(c['out']);path=root/'checkpoints'/(name+'.pt');summary=root/'results'/(name+'.json')
 rid=recipe(c,j)
 eval_identity=ident({'validation':c['validation_ids']} if j['phase']=='select' else {'test':c['parent']['test_ids'],'external':c['external_hashes']})
 if summary.exists():
  value=json.loads(summary.read_text())
  if value['recipe']!=rid:raise ValueError('Changed recipe; choose new PAPER_OUTPUT directory')
  if value.get('evaluation_inputs')==eval_identity:
   state(c,name,'COMPLETE',epochs=j['epochs']);return True
 rows=samples(c,j['method'],c['fit_ids'] if j['phase']=='select' else c['parent']['train_ids'])
 batch=c['parent']['seg_batch'];per_epoch=math.ceil(len(rows)/batch);total=j['epochs']*per_epoch
 seed_all(j['seed']);model=make_model(c,j['arch'],device);model.load_state_dict(initial(c,j,device));opt=torch.optim.Adam(model.parameters(),lr=c['parent']['seg_lr'])
 amp=bool(c.get('amp',False) and device.type=='cuda');scaler=torch.amp.GradScaler('cuda',enabled=amp)
 step=0;history=[];best_score=-1.0;best_epoch=0;validation_history=[]
 best_path=root/'checkpoints'/(name+'_best.pt')
 if path.exists():
  cp=load(path)
  if cp['recipe']!=rid:raise ValueError('Checkpoint recipe mismatch')
  model.load_state_dict(cp['model']);opt.load_state_dict(cp['optimizer']);step=cp['step'];history=cp['history'];best_score=cp.get('best_score',-1.0);best_epoch=cp.get('best_epoch',0);validation_history=cp.get('validation_history',[]);restore_rng(cp['rng'],device)
  if cp.get('scaler'):scaler.load_state_dict(cp['scaler'])
  del cp
 def checkpoint():
  save(path,{'recipe':rid,'job':j,'step':step,'batches_per_epoch':per_epoch,'model':model.state_dict(),'optimizer':opt.state_dict(),'history':history,'rng':rng_state(device),'scaler':scaler.state_dict(),'best_score':best_score,'best_epoch':best_epoch,'validation_history':validation_history})
  pd.DataFrame(history).to_csv(root/'checkpoints'/(name+'_loss.csv'),index=False)
 print(f'START {name}: epoch {step/per_epoch:.2f}/{j["epochs"]}; {len(rows)} training pairs',flush=True)
 state(c,name,'TRAINING',step=step,total_steps=total,epoch=step/per_epoch)
 # Finish an epoch validation interrupted after the training checkpoint was saved.
 if step>0 and step%per_epoch==0 and (not validation_history or validation_history[-1]['epoch']<step//per_epoch):
  records={r['id']:r for r in c['parent']['records']}
  validation=[records[i] for i in c['parent']['test_ids']]
  scores=predict(c,model,validation,device,can_pause=False)
  value=old.summarize(scores)['dataset_iou']
  validation_history.append({'epoch':step//per_epoch,'validation_dataset_iou':value})
  if value>best_score:
   best_score=value;best_epoch=step//per_epoch
   save(best_path,{'model':model.state_dict(),'epoch':best_epoch,'validation_iou':best_score,'recipe':rid})
  checkpoint()
 while step<total:
  if memory_stop(c,device):
   checkpoint();state(c,name,'PAUSED_MEMORY',step=step,total_steps=total);del model;gc.collect();return False
  epoch,part=divmod(step,per_epoch);order=np.random.default_rng(j['seed']+epoch).permutation(len(rows));positions=order[part*batch:(part+1)*batch]
  pairs=[tensor_pair(rows[int(pos)],c['parent']['size'],j['seed']+epoch*len(rows)+int(pos)) for pos in positions]
  x=torch.stack([a for a,y in pairs]).to(device);y=torch.stack([y for a,y in pairs]).to(device)
  model.train();opt.zero_grad(set_to_none=True)
  with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
   logits=model(normalize_segmenter(x))
  loss=dice_loss(logits.float(),y)
  del logits
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss: '+name)
  scaler.scale(loss).backward();scaler.unscale_(opt)
  before=scaler.get_scale();scaler.step(opt);scaler.update()
  if scaler.get_scale()<before:
   opt.zero_grad(set_to_none=True);print('AMP skipped nonfinite gradients; retry same batch',flush=True);continue
  step+=1
  history.append({'step':step,'epoch':epoch+1,'loss':float(loss.detach()),'samples':len(positions)})
  del x,y,loss
  if step%25==0 or step%per_epoch==0 or step==total:checkpoint();state(c,name,'TRAINING',step=step,total_steps=total,epoch=step/per_epoch)
  if step%per_epoch==0:
   epoch_rows=history[-per_epoch:];mean_loss=sum(r['loss']*r['samples'] for r in epoch_rows)/sum(r['samples'] for r in epoch_rows)
   print(f'{name} epoch {step//per_epoch}/{j["epochs"]}; mean_loss={mean_loss:.5f}',flush=True)
   records={r['id']:r for r in c['parent']['records']}
   validation=[records[i] for i in c['parent']['test_ids']]
   scores=predict(c,model,validation,device,can_pause=False)
   value=old.summarize(scores)['dataset_iou']
   validation_history.append({'epoch':step//per_epoch,'validation_dataset_iou':value})
   if value>best_score:
    best_score=value;best_epoch=step//per_epoch
    save(best_path,{'model':model.state_dict(),'epoch':best_epoch,'validation_iou':best_score,'recipe':rid})
   checkpoint()
   pd.DataFrame(validation_history).to_csv(root/'checkpoints'/(name+'_validation.csv'),index=False)
   print(f'{name}: validation IoU={value:.5f}; best epoch={best_epoch}',flush=True)

 checkpoint();del opt;gc.collect();state(c,name,'EVALUATING',epoch=j['epochs'])
 eval_path=path
 if c['checkpoint_policy']=='best_kvasir_validation':
  selected=load(best_path);model.load_state_dict(selected['model']);del selected;eval_path=best_path
 record_map={r['id']:r for r in c['parent']['records']}
 datasets={'validation':[record_map[i] for i in c['validation_ids']]} if j['phase']=='select' else {'Kvasir_Test100':[record_map[i] for i in c['parent']['test_ids']],**{n:r for n,r in c['external_records'].items() if r}}
 metrics={}
 for name2,rs in datasets.items():
  out=root/'predictions'/name/name2;out.mkdir(parents=True,exist_ok=True)
  done=out/'metrics.json';did=ident({'checkpoint':digest(eval_path),'data':old.file_identity([r[k] for r in rs for k in ['image','mask']]),'recipe':rid})
  if done.exists() and json.loads(done.read_text()).get('identity')==did:metrics[name2]=json.loads(done.read_text())['metrics'];continue
  df=predict(c,model,rs,device,out)
  if df is None:state(c,name,'PAUSED_MEMORY_EVALUATION',dataset=name2);del model;gc.collect();return False
  df.to_csv(out/'per_image.csv',index=False);metrics[name2]=old.summarize(df);write(done,{'identity':did,'metrics':metrics[name2]})
 write(summary,{'recipe':rid,'job':j,'epochs_completed':j['epochs'],'n_train':len(rows),'evaluation_inputs':eval_identity,'metrics':metrics,'selected_epoch':best_epoch if c['checkpoint_policy']=='best_kvasir_validation' else j['epochs'],'checkpoint_policy':c['checkpoint_policy'],'Kvasir_role':'validation used for checkpoint selection' if c['checkpoint_policy']=='best_kvasir_validation' else 'historically exposed evaluation set'})
 state(c,name,'COMPLETE',epoch=j['epochs'],datasets=list(metrics));del model;gc.collect()
 if device.type=='cuda':torch.cuda.empty_cache()
 return True

def select_seeds(c):
 root=Path(c['out']);choices={};records=[]
 for arch in c['architectures']:
  values=[]
  for j in phase_jobs(c,'select'):
   if j['arch']!=arch:continue
   p=root/'results'/(key(j)+'.json')
   if not p.exists():continue
   r=json.loads(p.read_text())
   if r['recipe']!=recipe(c,j):raise ValueError('Stale seed-selection result')
   score=r['metrics']['validation']['dataset_iou'];values.append((score,-j['seed'],j['seed']));records.append({'architecture':arch,'seed':j['seed'],'validation_iou':score})
  if len(values)==len(c['seeds']):choices[arch]=max(values)[2]
 write(root/'seed_selection.json',{'criterion':'Highest validation mixed IoU; tie smallest seed. No test metrics used.','selection_epochs':c['selection_epochs'],'scores':records,'selected':choices})
 return choices

def train_phase(c,phase,rank,world,device):
 jobs=phase_jobs(c,phase)
 for j in jobs[rank::world]:
  if not run_job(c,j,device):return


def quality(c,device):
 import extension as old
 # Old quality code may reuse its own cache, independently from new100epoch jobs.
 qc=copy.deepcopy(c);qc['code_sha256']=c['metrics_frozen_hash'];qc['out']=c['metric_cache_out'];qc['generate_100_extra']=bool(c.get('generate_100_extra',False));qc['image_ablation']=True;qc['metric_n']=1000
 Path(qc['out']).mkdir(parents=True,exist_ok=True)
 old.quality(qc,device)
 df=pd.read_csv(Path(qc['out'])/'image_metrics.csv');rows=[]
 df.to_csv(Path(c['out'])/'image_ablation.csv',index=False)
 for _,r in df[df.source=='deployment'].iterrows():rows.append({'method':'Polyp-DDPM',**r.to_dict()})
 refs={'Kvasir_SEG':[r['image'] for r in c['parent']['records']],'HyperKvasir':[r['image'] for r in c['external_records'].get('HyperKvasir',[])]}
 for method,spec in c['baselines'].items():
  fake=list(old.images(spec.get('quality_images')).values())
  for name,real in refs.items():
   if len(fake)<1000 or len(real)<1000:
    rows.append({'method':method,'reference':name,'status':'MISSING_INPUT','n_fake':len(fake),'n_real':len(real)});continue
   value=old.metrics_pair(qc,method+'_vs_'+name,fake[:1000],real[:1000],device)
   rows.append({'method':method,'reference':name,**{k:v for k,v in value.items() if k!='spec'}})
 if all(len(v)>=1000 for v in refs.values()):
  value=old.metrics_pair(qc,'real_Kvasir_vs_Hyper',refs['Kvasir_SEG'][:1000],refs['HyperKvasir'][:1000],device)
  for name in refs:rows.append({'method':'Kvasir vs Hyper (reference)','reference':name,**{k:v for k,v in value.items() if k!='spec'}})
 pd.DataFrame(rows).to_csv(Path(c['out'])/'table_I_long.csv',index=False)


def tables(c):
 root=Path(c['out']);choices=select_seeds(c);results=[]
 for j in phase_jobs(c,'paper'):
  p=root/'results'/(key(j)+'.json')
  if p.exists():
   r=json.loads(p.read_text())
   for dataset,m in r['metrics'].items():results.append({'architecture':j['arch'],'seed':j['seed'],'method':j['method'],'dataset':dataset,'selected_epoch':r.get('selected_epoch'),'checkpoint_policy':r.get('checkpoint_policy'),'evaluation_role':'validation-selected' if dataset=='Kvasir_Test100' and c['checkpoint_policy']=='best_kvasir_validation' else 'evaluation; overlap must be checked',**m})
 actual=pd.DataFrame(results);actual.to_csv(root/'table_II_long.csv',index=False)
 required=[]
 for a in c['architectures']:
  for d in ['Kvasir_Test100','HyperKvasir','ETIS']:
   for method in ['real','LDM','SinGAN-Seg','Polyp-DDPM']:
    match=actual[(actual.architecture==a)&(actual.dataset==d)&(actual.method==method)] if not actual.empty else pd.DataFrame()
    required.append({'table':'II','architecture':a,'dataset':d,'method':method,'status':'COMPLETE' if len(match)==1 else 'MISSING_RESULT'})
 t1=root/'table_I_long.csv';im=pd.read_csv(t1) if t1.exists() else pd.DataFrame()
 for method in ['LDM','SinGAN-Seg','Polyp-DDPM','Kvasir vs Hyper (reference)']:
  for ref in ['Kvasir_SEG','HyperKvasir']:
   match=im[(im.method==method)&(im.reference==ref)&(im.status=='complete')&(im.n_fake==1000)&(im.n_real==1000)] if not im.empty else pd.DataFrame()
   required.append({'table':'I','architecture':'','dataset':ref,'method':method,'status':'COMPLETE' if len(match)==1 else 'MISSING_RESULT'})
 for a in c['architectures']:
  if a not in choices:required.append({'table':'selection','architecture':a,'dataset':'validation','method':'all seeds','status':'MISSING_RESULT'})
  for d in ['Kvasir_Test100','HyperKvasir','ETIS']:
   match=actual[(actual.architecture==a)&(actual.dataset==d)&(actual.method=='mixed')] if not actual.empty else pd.DataFrame()
   required.append({'table':'mixed','architecture':a,'dataset':d,'method':'mixed','status':'COMPLETE' if len(match)==1 else 'MISSING_RESULT'})
 coverage=pd.DataFrame(required);coverage.to_csv(root/'coverage.csv',index=False);write(root/'completion.json',{'complete':bool((coverage.status=='COMPLETE').all()),'missing':coverage[coverage.status!='COMPLETE'].to_dict('records')})
 return actual,im,coverage

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--phase',choices=['select','paper','quality'],required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1);a=p.parse_args();c=json.loads(Path(a.config).read_text())
 sys.path[:0]=[c['runtime'],c['metrics_runtime']]
 if digest(__file__)!=c['runner_hash']:raise ValueError('Runner changed; a new study folder is required')
 device=torch.device('cpu' if c['parent'].get('cpu_fixture') else f'cuda:{a.rank}')
 if device.type=='cuda':torch.cuda.set_device(device)
 torch.set_num_threads(2)
 from reference_io import install
 install(c)
 try:
  if a.phase=='quality':quality(c,device)
  else:train_phase(c,a.phase,a.rank,a.world,device)
 except Exception as e:
  state(c,f'worker_{a.phase}_{a.rank}','ERROR',error=repr(e));raise

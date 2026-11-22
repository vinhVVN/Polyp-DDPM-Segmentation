"""Input/metric adapters derived from reference[9]; source snapshots embedded."""
import random
import numpy as np
import torch
from PIL import Image
import albumentations as albu
import common

_train=None
_valid=None
MODE='reference_code'
def configure(c):
    global _train,_valid,MODE
    from reference_augmentation import get_training_augmentation,get_validation_augmentation
    _train=get_training_augmentation();_valid=get_validation_augmentation()
    MODE=c.get('preprocessing_mode','reference_code')

def tensor_pair(record,size,seed=None,synthetic=None):
    with Image.open(synthetic or record['image']) as handle:image=np.array(handle.convert('RGB'),copy=True)
    with Image.open(record['mask']) as handle:mask=(np.array(handle.convert('RGB'))[:,:,0]>128).astype(float)
    py=random.getstate();nr=np.random.get_state()
    try:
        if seed is not None:random.seed(seed);np.random.seed(seed%(2**32))
        result=(_train if seed is not None else _valid)(image=image,mask=mask)
    finally:random.setstate(py);np.random.set_state(nr)
    x=torch.from_numpy(result['image'].copy().transpose(2,0,1)).float()/255
    y=torch.from_numpy(result['mask'].copy()).float()[None]
    if MODE=='reference_code':x=common.normalize_segmenter(x[None])[0]
    return x,y

def install(c):
    configure(c)
    common.tensor_pair=tensor_pair
    original=common.binary_metrics
    def metrics(prob,mask):
        # Reference uses >0.5, whereas old notebook used >=0.5.
        return original((prob>0.5).astype(np.float32),mask)
    common.binary_metrics=metrics
    if not c['parent'].get('cpu_fixture'):
        import segmentation_models_pytorch as smp
        common.dice_loss=smp.losses.DiceLoss(smp.losses.BINARY_MODE,from_logits=True)

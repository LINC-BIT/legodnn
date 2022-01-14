from mmseg.apis import init_segmentor
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint

LOAD_MODE = ['lego_jit', 'mmseg_test', 'mmseg_train']

def mmseg_init_model(config, checkpoint=None, mode='lego_jit', device='cuda'):
    assert mode in LOAD_MODE
    if mode=='lego_jit':
        segmentor = init_segmentor(config, checkpoint, device=device)
        segmentor.forward = segmentor.forward_dummy
    elif mode=='mmseg_test':
        segmentor = init_segmentor(config, checkpoint, device=device)
        segmentor = segmentor
    elif mode=='mmseg_train':
        segmentor = build_segmentor(config.model, train_cfg=config.get('train_cfg'), test_cfg=config.get('test_cfg'))
        if checkpoint is not None:
            checkpoint = load_checkpoint(segmentor, checkpoint, map_location='cpu')
            segmentor.CLASSES = checkpoint['meta']['CLASSES']
            segmentor.PALETTE = checkpoint['meta']['PALETTE']
        segmentor.cfg = config
    else:
        raise NotImplementedError
    segmentor.to(device)
    segmentor.eval()
    return segmentor
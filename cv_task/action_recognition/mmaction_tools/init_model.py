from mmaction.apis import init_recognizer
from mmaction.models import build_model
from mmcv.runner import load_checkpoint

LOAD_MODE = ['lego_jit', 'mmaction_test', 'mmaction_train']

def mmaction_init_model(config, checkpoint=None, mode='lego_jit', device='cuda'):
    assert mode in LOAD_MODE
    if mode=='lego_jit':
        recognizer = init_recognizer(config, checkpoint, device=device)
        recognizer.forward = recognizer.forward_dummy
    elif mode=='mmaction_test':
        recognizer = init_recognizer(config, checkpoint, device=device)
        recognizer = recognizer
    elif mode=='mmaction_train':
        recognizer = build_model(config.model, train_cfg=config.get('train_cfg'), test_cfg=config.get('test_cfg'))
        if checkpoint is not None:
            checkpoint = load_checkpoint(recognizer, checkpoint, map_location='cpu')
            # recognizer.CLASSES = checkpoint['meta']['CLASSES']
            # recognizer.PALETTE = checkpoint['meta']['PALETTE']
        recognizer.cfg = config
    else:
        raise NotImplementedError
    recognizer.to(device)
    recognizer.eval()
    return recognizer
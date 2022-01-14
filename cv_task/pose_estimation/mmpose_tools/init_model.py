from mmcv.runner import load_checkpoint
from mmpose.models import build_posenet
from mmpose.apis import init_pose_model

LOAD_MODE = ['lego_jit', 'mmpose_test', 'mmpose_train']

def mmpose_init_model(config, checkpoint=None, mode='lego_jit', device='cuda'):
    assert mode in LOAD_MODE
    if mode=='lego_jit':
        # posenet = build_posenet(config)
        posenet = init_pose_model(config, checkpoint, device)
        posenet.forward = posenet.forward_dummy
    elif mode=='mmpose_test':
        posenet = init_pose_model(config, checkpoint, device)
        posenet = posenet
    elif mode=='mmpose_train':
        posenet = build_posenet(config.model)
        if checkpoint is not None:
            checkpoint = load_checkpoint(posenet, checkpoint, map_location='cpu')
            # posenet.CLASSES = checkpoint['meta']['CLASSES']
            # posenet.PALETTE = checkpoint['meta']['PALETTE']
        posenet.cfg = config
    else:
        raise NotImplementedError
    posenet.to(device)
    posenet.eval()
    return posenet
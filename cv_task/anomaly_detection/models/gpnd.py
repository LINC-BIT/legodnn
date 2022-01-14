
import torch
from cv_task.anomaly_detection.methods.gpnd.net import GPND, NewGPND

def gpnd_caltech256(pretrained=True, device='cuda'):
    image_channel = 3
    latent_size = 128
    if pretrained:
        model = torch.load('/data/gxy/legodnn-auto-on-cv-models/cv_task_model/anomaly_detection/gpnd_caltech256/gpnd_caltech256').to(device)
    else:
        model = NewGPND(zsize=latent_size, nc=image_channel)
    model.eval()
    return model
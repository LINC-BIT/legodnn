import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from cv_task.datasets.anomaly_detection.gpnd_caltech256 import gpnd_caltech256_dataloader
from cv_task.anomaly_detection.models.gpnd import gpnd_caltech256
from cv_task.anomaly_detection.models_tools.test import test_gpnd_model

if __name__=='__main__':
    device = 'cuda'
    train_loader, test_loader = gpnd_caltech256_dataloader()
    model = gpnd_caltech256()
    auc = test_gpnd_model(model, train_loader, test_loader, device=device)
    print(auc)
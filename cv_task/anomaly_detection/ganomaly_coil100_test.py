import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from cv_task.anomaly_detection.models.ganomaly import ganomaly_coil100_netg
from cv_task.datasets.anomaly_detection.ganomaly_coil100 import ganomaly_coil100_dataloader
from cv_task.anomaly_detection.models_tools.test import test_ganomaly_model

if __name__=='__main__':
    model_save_path = '/data/gxy/legodnn-auto-on-cv-models/cv_task_model/anomaly_detection/ganomaly_coil100/ganomaly_coil100'
    device = 'cuda'
    model = ganomaly_coil100_netg(pretrained=True, device=device)
    # print(model)
    # exit(0)
    train_loader, test_loader = ganomaly_coil100_dataloader(train_batch_size=64, test_batch_size=64)
    auc = test_ganomaly_model(model, test_loader)
    print(auc)

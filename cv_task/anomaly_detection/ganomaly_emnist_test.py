import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from cv_task.anomaly_detection.models.ganomaly import ganomaly_emnist_netg
from cv_task.datasets.anomaly_detection.ganomaly_emnist import ganomaly_emnist_dataloader
from cv_task.anomaly_detection.models_tools.test import test_ganomaly_model

if __name__=='__main__':
    # model_save_path = '/data/gxy/legodnn-auto-on-cv-models/cv_task_model/anomaly_detection/ganomaly_emnist/ganomaly_emnist'
    device = 'cuda'
    model = ganomaly_emnist_netg(pretrained=True, device=device)
    # print(model)
    # exit(0)
    train_loader, test_loader = ganomaly_emnist_dataloader(train_batch_size=64, test_batch_size=64)
    # dataloader = dict(train=train_loader, test=test_loader)
    auc = test_ganomaly_model(model, test_loader, 'emnist')
    print(auc)

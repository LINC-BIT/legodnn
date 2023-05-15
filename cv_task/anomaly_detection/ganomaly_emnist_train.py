import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from cv_task.anomaly_detection.models.ganomaly import ganomaly_emnist_whole
from cv_task.datasets.anomaly_detection.ganomaly_emnist import ganomaly_emnist_dataloader

if __name__=='__main__':
    model_save_path = '/data/gxy/legodnn-auto-on-cv-models/cv_task_model/anomaly_detection/ganomaly_emnist/ganomaly_emnist'
    device = 'cuda'
    model = ganomaly_emnist_whole(device=device, model_save_path=model_save_path)
    train_loader, test_loader = ganomaly_emnist_dataloader(train_batch_size=model.opt.batchsize, test_batch_size=model.opt.batchsize)
    dataloader = dict(train=train_loader, test=test_loader)
    model.train_model(dataloader)

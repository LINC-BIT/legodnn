import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from cv_task.anomaly_detection.models.ganomaly import ganomaly_coil100_netg, ganomaly_coil100_whole
from cv_task.datasets.anomaly_detection.ganomaly_coil100 import ganomaly_coil100_dataloader

if __name__=='__main__':
    model_save_path = '/data/gxy/legodnn-auto-on-cv-models/cv_task_model/anomaly_detection/ganomaly_coil100/ganomaly_coil100'
    device = 'cuda'
    batch_size = 64
    model = ganomaly_coil100_whole(device=device, model_save_path=model_save_path, batch_size=batch_size)
    train_loader, test_loader = ganomaly_coil100_dataloader(train_batch_size=batch_size, test_batch_size=batch_size)
  
    dataloader = dict(train=train_loader, test=test_loader)
    model.train_model(dataloader)
    
    
    
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from cv_task.anomaly_detection.methods.gpnd.train_AAE import train
from cv_task.anomaly_detection.methods.gpnd.test_AAE import test
# from cv_task.anomaly_detection.methods.gpnd.net import Generator, Encoder, GPND
from cv_task.anomaly_detection.methods.gpnd.net import NewGenerator, NewEncoder, NewGPND
from cv_task.datasets.anomaly_detection.gpnd_caltech256 import gpnd_caltech256_dataloader
from legodnn.utils.dl.common.env import set_random_seed

set_random_seed(2)

if __name__=='__main__':
    image_channel = 3
    image_size = 32
    latent_size = 128
    epoch_num = 300
    learning_rate = 0.001
    batch_size = 64
    
    model_save_path = '/data/gxy/legodnn-auto-on-cv-models/cv_task_model/anomaly_detection/gpnd_caltech256/gpnd_caltech256'
    device = 'cuda'
    
    train_loader, test_loader = gpnd_caltech256_dataloader()
    G = NewGenerator(latent_size, channels=image_channel).to(device)
    E = NewEncoder(latent_size, channels=image_channel).to(device)
    
    train(G, E, model_save_path, train_loader, image_channel, image_size, latent_size, epoch_num, batch_size, learning_rate)
    # def train(G, E, model_save_path, train_loader, nc, isize, zsize, epoch_num, batch_size=64, lr=0.002):
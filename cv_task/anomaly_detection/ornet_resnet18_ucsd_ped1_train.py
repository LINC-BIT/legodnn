import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torchvision.models as fl_models
from torchvision.models.resnet import resnet18
from cv_task.anomaly_detection.methods.ornet.anomaly_score_learner import AnomalyScoreLearner
from cv_task.anomaly_detection.methods.ornet.util.train import train_us_net, test, public_test_loader, public_all_video_frames_label, train

if __name__ == '__main__':
    epoch_num = 50
    learning_rate = 0.001
    model_save_path = '/data/gxy/legodnn-auto-on-cv-models/cv_task_model/anomaly_detection/ornet_resnet18_ucsd_ped1_2/ornet_resnet18_ucsd_ped1'
    device = 'cuda'
    
    backbone = resnet18(pretrained=True).to(device)
    model = AnomalyScoreLearner(backbone, 1000).to(device)
    
    train(model, '/data/zql/mcw/model/ornet/ucsd-ped1-raw-data.pth', 
        '/data/zql/mcw/model/ornet/ucsd-ped1-AN-tensor.pth', model_save_path, epoch_num, lr=learning_rate, init_epoch_num=3)
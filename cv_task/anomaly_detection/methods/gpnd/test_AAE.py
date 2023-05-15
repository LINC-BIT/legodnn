import torch.utils.data
from torchvision.utils import save_image
import sys
sys.path.insert(0, '/data/zql/legodnn-rtss-baselines/cgnet')
from cgnet_open_api.cg_layers import CGConv2d
from cv_task.anomaly_detection.methods.gpnd.net import *
from torch.autograd import Variable
from cv_task.anomaly_detection.methods.gpnd.utils.jacobian import compute_jacobian_autograd
import numpy as np
import logging
import os
import scipy.optimize
# from cv_task.anomaly_detection.methods.gpnd.dataloading import make_datasets, make_dataloader, create_set_with_outlier_percentage
from cv_task.anomaly_detection.methods.gpnd.evaluation import get_f1, evaluate
from cv_task.anomaly_detection.methods.gpnd.utils.threshold_search import find_maximum
# from cv_task.anomaly_detection.methods.gpnd.utils.save_plot import save_plot
import matplotlib.pyplot as plt
import scipy.stats
from scipy.special import loggamma
import tqdm

# from cv_task.anomaly_detection.methods.ganomaly.lib.evaluate import evaluate_all
import sys
sys.path.insert(0, '/data/zql/zedl')
from zedl.common.log import logger


def r_pdf(x, bins, counts):
    if bins[0] < x < bins[-1]:
        i = np.digitize(x, bins) - 1
        return max(counts[i], 1e-308)
    if x < bins[0]:
        return max(counts[0] * x / bins[0], 1e-308)
    return 1e-308


def extract_statistics(G, E, nc, isize, zsize, batch_size, train_loader):
    zlist = []
    rlist = []

    # train_dataloader = make_dataloader(train_set, batch_size, torch.cuda.current_device())
    for x, label in train_loader:
        x = x.to('cuda')
        
        x = x.view(-1, nc * isize * isize)
        z = E(x.view(-1, nc, isize, isize))
        recon_batch = G(z)
        z = z.squeeze()

        recon_batch = recon_batch.squeeze().cpu().detach().numpy()
        x = x.squeeze().cpu().detach().numpy()

        z = z.cpu().detach().numpy()

        for i in range(x.shape[0]):
            distance = np.linalg.norm(x[i].flatten() - recon_batch[i].flatten())
            rlist.append(distance)

        zlist.append(z)

    zlist = np.concatenate(zlist)

    counts, bin_edges = np.histogram(rlist, bins=30, density=True)

    # if cfg.MAKE_PLOTS:
    #     plt.plot(bin_edges[1:], counts, linewidth=2)
    #     save_plot(r"Distance, $\left \|\| I - \hat{I} \right \|\|$",
    #               'Probability density',
    #               r"PDF of distance for reconstruction error, $p\left(\left \|\| I - \hat{I} \right \|\| \right)$",
    #               cfg.OUTPUT_FOLDER + '/mnist_%s_reconstruction_error.pdf' % ("_".join([str(x) for x in inliner_classes])))

    # for i in range(cfg.MODEL.LATENT_SIZE):
    #     plt.hist(zlist[:, i], bins='auto', histtype='step')

    # if cfg.MAKE_PLOTS:
    #     save_plot(r"$z$",
    #               'Probability density',
    #               r"PDF of embeding $p\left(z \right)$",
    #               cfg.OUTPUT_FOLDER + '/mnist_%s_embedding.pdf' % ("_".join([str(x) for x in inliner_classes])))

    def fmin(func, x0, args, disp):
        x0 = [2.0, 0.0, 1.0]
        return scipy.optimize.fmin(func, x0, args, xtol=1e-12, ftol=1e-12, disp=0)

    gennorm_param = np.zeros([3, zsize])
    for i in range(zsize):
        betta, loc, scale = scipy.stats.gennorm.fit(zlist[:, i], optimizer=fmin)
        gennorm_param[0, i] = betta
        gennorm_param[1, i] = loc
        gennorm_param[2, i] = scale

    return counts, bin_edges, gennorm_param


def test(G, E, train_loader, test_loader, nc, isize, zsize, batch_size):
    # logger = logging.getLogger("logger")
    logger.info('testing...')

    mul = 0.2
    batch_size = batch_size * 8

    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # device = torch.cuda.current_device()
    # print("Running on ", torch.cuda.get_device_name(device))

    # train_set, valid_set, test_set = make_datasets(cfg, folding_id, inliner_classes)

    # print('Validation set size: %d' % len(valid_set))
    # print('Test set size: %d' % len(test_set))

    # train_set.shuffle()

    # G = Generator(cfg.MODEL.LATENT_SIZE, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)
    # E = Encoder(cfg.MODEL.LATENT_SIZE, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)

    # G.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_FOLDER, "models/Gmodel_%d_%d.pkl" %(folding_id, ic))))
    # E.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_FOLDER, "models/Emodel_%d_%d.pkl" %(folding_id, ic))))

    G.eval()
    E.eval()

    sample = torch.randn(64, zsize).cuda()
    sample = G(sample.view(-1, zsize, 1, 1)).cpu()
    # save_image(sample.view(64, nc, isize, isize), 'sample.png')

    logger.info('extracting statistics...')
    counts, bin_edges, gennorm_param = extract_statistics(G, E, nc, isize, zsize, batch_size, train_loader)

    def run_novely_prediction_on_dataset(dataloader):
        # dataset.shuffle()
        # dataset = create_set_with_outlier_percentage(dataset, inliner_classes, percentage, concervative)

        result = []
        gt_novel = []

        # dataloader = make_dataloader(dataset, batch_size, torch.cuda.current_device())

        include_jacobian = True

        N = (isize * isize - zsize) * mul
        logC = loggamma(N / 2.0) - (N / 2.0) * np.log(2.0 * np.pi)

        def logPe_func(x):
            # p_{\|W^{\perp}\|} (\|w^{\perp}\|)
            # \| w^{\perp} \|}^{m-n}
            return logC - (N - 1) * np.log(x) + np.log(r_pdf(x, bin_edges, counts))

        logger.info('testing in dataloader...')
        from tqdm import tqdm
        for x, label in tqdm(dataloader):
            # print(1111)
            x = x.cuda()
            
            x = x.view(-1, nc * isize * isize)
            x = Variable(x.data, requires_grad=True)

            z = E(x.view(-1, nc, isize, isize))
            recon_batch = G(z)
            z = z.squeeze()

            if include_jacobian:
                J = compute_jacobian_autograd(x, z)
                J = J.cpu().numpy()

            z = z.cpu().detach().numpy()

            recon_batch = recon_batch.squeeze().cpu().detach().numpy()
            x = x.squeeze().cpu().detach().numpy()

            for i in range(x.shape[0]):
                if include_jacobian:
                    u, s, vh = np.linalg.svd(J[i, :, :], full_matrices=False)
                    logD = -np.sum(np.log(np.abs(s)))  # | \mathrm{det} S^{-1} |
                    # logD = np.log(np.abs(1.0/(np.prod(s))))
                else:
                    logD = 0

                p = scipy.stats.gennorm.pdf(z[i], gennorm_param[0, :], gennorm_param[1, :], gennorm_param[2, :])
                logPz = np.sum(np.log(p))

                # Sometimes, due to rounding some element in p may be zero resulting in Inf in logPz
                # In this case, just assign some large negative value to make sure that the sample
                # is classified as unknown.
                if not np.isfinite(logPz):
                    logPz = -1000

                distance = np.linalg.norm(x[i].flatten() - recon_batch[i].flatten())

                logPe = logPe_func(distance)

                P = logD + logPz + logPe

                result.append(P)
                gt_novel.append(1 - label[i].item())
                # print(P)

        result = np.asarray(result, dtype=np.float32)
        # normalize
        # print(np.max(result), np.min(result))
        # result = (result - np.min(result)) / (np.max(result) - np.min(result))
        # print(result[0:50])
        ground_truth = np.asarray(gt_novel, dtype=np.float32)
        return result, ground_truth
    
    # auc_res, auprc_res, f1_res = evaluate_all(torch.from_numpy(labels), torch.from_numpy(scores))
    # def compute_threshold(valid_set):
    #     scores, labels = run_novely_prediction_on_dataset(valid_set)
    #     minP = min(scores) - 1
    #     maxP = max(scores) + 1
    #     y_false = np.logical_not(labels)

    #     def _evaluate(e):
    #         y = np.greater(scores, e)
    #         true_positive = np.sum(np.logical_and(y, labels))
    #         false_positive = np.sum(np.logical_and(y, y_false))
    #         false_negative = np.sum(np.logical_and(np.logical_not(y), labels))
    #         return get_f1(true_positive, false_positive, false_negative)

    #     best_th, best_f1 = find_maximum(_evaluate, minP, maxP, 1e-4)
    #     logger.info('best threshold in valid dataset: {}, best f1 {}'.format(best_th, best_f1))
    #     return best_th
    
    # best_th = compute_threshold(valid_set)
    scores, labels = run_novely_prediction_on_dataset(test_loader)
    
    auc_res, auprc_res, f1_res = evaluate(scores, 0, labels)
    logger.info('auc: {:.6f}, auprc: {:.6f}, f1: {:.6f}'.format(auc_res, auprc_res, f1_res))
    
    normal_samples_index = np.nonzero(labels == 1)[0]
    abnormal_samples_index = np.nonzero(labels == 0)[0]
    normal_samples_score = scores[normal_samples_index].tolist()
    abnormal_samples_score = scores[abnormal_samples_index].tolist()
    
    return (auc_res, auprc_res, f1_res), (normal_samples_score, abnormal_samples_score)


def test_cgnet(G, E, train_loader, test_loader, nc, isize, zsize, batch_size):
    # logger = logging.getLogger("logger")
    logger.info('testing...')

    mul = 0.2
    batch_size = batch_size * 8

    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # device = torch.cuda.current_device()
    # print("Running on ", torch.cuda.get_device_name(device))

    # train_set, valid_set, test_set = make_datasets(cfg, folding_id, inliner_classes)

    # print('Validation set size: %d' % len(valid_set))
    # print('Test set size: %d' % len(test_set))

    # train_set.shuffle()

    # G = Generator(cfg.MODEL.LATENT_SIZE, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)
    # E = Encoder(cfg.MODEL.LATENT_SIZE, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)

    # G.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_FOLDER, "models/Gmodel_%d_%d.pkl" %(folding_id, ic))))
    # E.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_FOLDER, "models/Emodel_%d_%d.pkl" %(folding_id, ic))))

    G.eval()
    E.eval()

    sample = torch.randn(64, zsize).cuda()
    sample = G(sample.view(-1, zsize, 1, 1)).cpu()
    # save_image(sample.view(64, nc, isize, isize), 'sample.png')

    logger.info('extracting statistics...')
    counts, bin_edges, gennorm_param = extract_statistics(G, E, nc, isize, zsize, batch_size, train_loader)

    def run_novely_prediction_on_dataset(dataloader):
        # dataset.shuffle()
        # dataset = create_set_with_outlier_percentage(dataset, inliner_classes, percentage, concervative)

        result = []
        gt_novel = []

        # dataloader = make_dataloader(dataset, batch_size, torch.cuda.current_device())

        include_jacobian = True

        N = (isize * isize - zsize) * mul
        logC = loggamma(N / 2.0) - (N / 2.0) * np.log(2.0 * np.pi)

        def logPe_func(x):
            # p_{\|W^{\perp}\|} (\|w^{\perp}\|)
            # \| w^{\perp} \|}^{m-n}
            return logC - (N - 1) * np.log(x) + np.log(r_pdf(x, bin_edges, counts))

        logger.info('testing in dataloader...')
        
        E_full_flops, E_final_flops = 0, 0
        G_full_flops, G_final_flops = 0, 0
        batch_num = 0
        
        for x, label in dataloader:
            print(1111)
            x = x.cuda()
            
            x = x.view(-1, nc * isize * isize)
            x = Variable(x.data, requires_grad=True)

            z = E(x.view(-1, nc, isize, isize))
            recon_batch = G(z)
            z = z.squeeze()
            
            for name, module in E.named_modules():
                if isinstance(module, CGConv2d):
                    E_full_flops += module.num_out
                    E_final_flops += module.num_full
            for name, module in G.named_modules():
                if isinstance(module, CGConv2d):
                    G_full_flops += module.num_out
                    G_final_flops += module.num_full
            batch_num += 1

            if include_jacobian:
                J = compute_jacobian_autograd(x, z)
                J = J.cpu().numpy()

            z = z.cpu().detach().numpy()

            recon_batch = recon_batch.squeeze().cpu().detach().numpy()
            x = x.squeeze().cpu().detach().numpy()

            for i in range(x.shape[0]):
                if include_jacobian:
                    u, s, vh = np.linalg.svd(J[i, :, :], full_matrices=False)
                    logD = -np.sum(np.log(np.abs(s)))  # | \mathrm{det} S^{-1} |
                    # logD = np.log(np.abs(1.0/(np.prod(s))))
                else:
                    logD = 0

                p = scipy.stats.gennorm.pdf(z[i], gennorm_param[0, :], gennorm_param[1, :], gennorm_param[2, :])
                logPz = np.sum(np.log(p))

                # Sometimes, due to rounding some element in p may be zero resulting in Inf in logPz
                # In this case, just assign some large negative value to make sure that the sample
                # is classified as unknown.
                if not np.isfinite(logPz):
                    logPz = -1000

                distance = np.linalg.norm(x[i].flatten() - recon_batch[i].flatten())

                logPe = logPe_func(distance)

                P = logD + logPz + logPe

                result.append(P)
                gt_novel.append(1 - label[i].item())
                # print(P)

        result = np.asarray(result, dtype=np.float32)
        # normalize
        # print(np.max(result), np.min(result))
        # result = (result - np.min(result)) / (np.max(result) - np.min(result))
        # print(result[0:50])
        ground_truth = np.asarray(gt_novel, dtype=np.float32)
        
        cgnet_reduced_flops_ratio = 1 - (E_final_flops + G_final_flops) / (E_full_flops + G_full_flops)
        
        return result, ground_truth, cgnet_reduced_flops_ratio
    
    # auc_res, auprc_res, f1_res = evaluate_all(torch.from_numpy(labels), torch.from_numpy(scores))
    # def compute_threshold(valid_set):
    #     scores, labels = run_novely_prediction_on_dataset(valid_set)
    #     minP = min(scores) - 1
    #     maxP = max(scores) + 1
    #     y_false = np.logical_not(labels)

    #     def _evaluate(e):
    #         y = np.greater(scores, e)
    #         true_positive = np.sum(np.logical_and(y, labels))
    #         false_positive = np.sum(np.logical_and(y, y_false))
    #         false_negative = np.sum(np.logical_and(np.logical_not(y), labels))
    #         return get_f1(true_positive, false_positive, false_negative)

    #     best_th, best_f1 = find_maximum(_evaluate, minP, maxP, 1e-4)
    #     logger.info('best threshold in valid dataset: {}, best f1 {}'.format(best_th, best_f1))
    #     return best_th
    
    # best_th = compute_threshold(valid_set)
    scores, labels, cgnet_reduced_flops_ratio = run_novely_prediction_on_dataset(test_loader)
    
    auc_res, auprc_res, f1_res = evaluate(scores, 0, labels)
    logger.info('auc: {:.6f}, auprc: {:.6f}, f1: {:.6f}, sparsity {:.2f}%'.format(auc_res, auprc_res, f1_res,
                                                                                  cgnet_reduced_flops_ratio * 100))
    
    normal_samples_index = np.nonzero(labels == 1)[0]
    abnormal_samples_index = np.nonzero(labels == 0)[0]
    normal_samples_score = scores[normal_samples_index].tolist()
    abnormal_samples_score = scores[abnormal_samples_index].tolist()
    
    return (auc_res, auprc_res, f1_res), (normal_samples_score, abnormal_samples_score), cgnet_reduced_flops_ratio

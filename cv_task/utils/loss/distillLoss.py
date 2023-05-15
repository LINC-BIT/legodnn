from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class Similarity(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, g_s, g_t):
        return self.similarity_loss(g_s, g_t)

    def similarity_loss(self, f_s, f_t):
        bsz = f_s.shape[0]
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss

class ATLoss(nn.Module):
    def __init__(self):
        super(ATLoss, self).__init__()

    def at(self, x):
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
    
    def forward(self, student_feature_map, teacher_feature_map):
        return (self.at(student_feature_map) - self.at(teacher_feature_map)).pow(2).mean()

def cosine_similarity_loss(output, target, eps=0.0000001):
    
    output_net = output.view(output.size(0), -1)
    target_net = target.view(target.size(0), -1)
    # Normalize each vector by its norm
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0

    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0

    # Calculate the cosine similarity
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1,
                                                    keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1,
                                                      keepdim=True)

    # Calculate the KL-divergence
    loss = torch.sum(target_similarity * torch.log(
        (target_similarity + eps) / (model_similarity + eps)))

    return loss

class KLCosineSimilarity(nn.Module):
    r"""
    KL divergence between two distribution which is represented by cosine similarity loss between
    """

    def __init__(self):
        super(KLCosineSimilarity, self).__init__()

    def forward(self, x, target):
        return cosine_similarity_loss(x, target)
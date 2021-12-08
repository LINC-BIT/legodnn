# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from . import LevelPrunerMasker, SlimPrunerMasker, L1FilterPrunerMasker, \
    L2FilterPrunerMasker, FPGMPrunerMasker, TaylorFOWeightFilterPrunerMasker, \
    ActivationAPoZRankFilterPrunerMasker, ActivationMeanRankFilterPrunerMasker, \
    TRRPrunerMasker, HRankPrunerMasker, PFPMasker

MASKER_DICT = {
    'level': LevelPrunerMasker,
    'slim': SlimPrunerMasker,
    'l1': L1FilterPrunerMasker,
    'l2': L2FilterPrunerMasker,
    'fpgm': FPGMPrunerMasker,
    'taylorfo': TaylorFOWeightFilterPrunerMasker,
    'apoz': ActivationAPoZRankFilterPrunerMasker,
    'mean_activation': ActivationMeanRankFilterPrunerMasker,

    # implemented by queyu, 2020/11/23
    'trr': TRRPrunerMasker,
    # implemented by queyu, 2021/6/10
    'hrank': HRankPrunerMasker,
    'pfp': PFPMasker
}

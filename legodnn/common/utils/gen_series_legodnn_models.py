import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint
import os
from legodnn.common.utils.common.log import logger
from legodnn.common.utils.common.data_record import write_json
from legodnn.common.utils.common.file import ensure_dir, create_dir
from legodnn.common.utils.dl.common.model import save_model, ModelSaveMethod

def gen_series_legodnn_models(deadline, model_size_search_range, target_model_num, optimal_runtime, descendant_models_save_path, device='cuda'):

    min_model_size = model_size_search_range[0]*1024**2
    max_model_size = model_size_search_range[1]*1024**2

    logger.info('min model size: {:.3f}MB, max model size: {:.3f}MB'.format(min_model_size/1024**2, max_model_size/1024**2))
    # target_model_num = 50
    
    models_info = []
    create_dir(descendant_models_save_path)

    models_info_path = os.path.join(descendant_models_save_path, 'models-info.json')
    for i, target_model_size in enumerate(np.linspace(min_model_size, max_model_size, target_model_num)):
        
        logger.info('target model size: {:.3f}MB'.format(target_model_size / 1024**2))
        
        update_info = optimal_runtime.update_model(deadline, target_model_size)
        logger.info('update info: \n{}'.format(pprint.pformat(update_info, indent=2, depth=2)))
        cur_model = optimal_runtime._pure_runtime.get_model()

        cur_model = cur_model.to('cpu')
        model_save_path = os.path.join(descendant_models_save_path, '{}.pt.cpu-jit'.format(i))
        save_model(cur_model, model_save_path, ModelSaveMethod.JIT, optimal_runtime._model_input_size)

        models_info += [{
            'model_file_name': '{}.pt.cpu-jit'.format(i),
            'model_info': update_info
        }]
        
        write_json(models_info_path, models_info, backup=False)

    # visualize
    models_size = [i['model_info']['model_size'] / 1024**2 for i in models_info]
    models_acc = [i['model_info']['esti_test_accuracy'] for i in models_info]
    
    plt.rc('font', family='Times New Roman')
    plt.rcParams['font.size'] = '20'
    plt.plot(models_size, models_acc, linewidth=2, color='black')
    plt.xlabel('model size (MB)')
    plt.ylabel('acc')
    plt.tight_layout()
    plt.savefig(os.path.join(descendant_models_save_path, 'models-info.png'), dpi=300)
    plt.clf()
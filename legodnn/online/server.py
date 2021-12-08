# *_*coding:utf-8 *_*
import random
import threading
import time

from flask import Flask, render_template, request, jsonify
from legodnn import data_path
import os
import sys
from werkzeug.utils import secure_filename
import torch
from PIL import Image
from torchvision import transforms
sys.path.insert(0, '../../')
from legodnn.common.manager import CommonBlockManager, CommonModelManager
from optimal_runtime import OptimalRuntime
from legodnn.common.utils import set_random_seed,memory_stat,logger
from legodnn.common.manager import ResNet110BlockManager,out_info

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = data_path + '/upload/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
cur_memory = memory_stat()

def init_model():
    set_random_seed(0)
    device = 'cpu'
    model_input_size = (1, 3, 32, 32)
    # legodnn config
    block_sparsity = [0.0, 0.125, 0.25, 0.375, 0.5]
    compressed_blocks_dir_path = data_path + '/blocks/resnet110/resnet110-cifar10-m6/compressed'
    trained_blocks_dir_path = data_path + '/blocks/resnet110/resnet110-cifar10-m6/trained'
    default_blocks_id = ResNet110BlockManager.get_default_blocks_id()
    model_manager = CommonModelManager()
    block_manager = ResNet110BlockManager(default_blocks_id,
                                          [block_sparsity for _ in range(len(default_blocks_id))], model_manager)
    optimal_runtime = OptimalRuntime(trained_blocks_dir_path, model_input_size,
                                     block_manager, model_manager, device)
    optimal_runtime.update_model(10, cur_memory * 1024 ** 2)
    model = optimal_runtime._pure_runtime.get_model()
    model.eval()
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        normalize,
    ])
    return model,transform,optimal_runtime

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def runtime_ajdust_model(optimal_runtime):
    global cur_memory
    while True:
        memory = memory_stat()
        logger.debug(f"memory:{memory} cur_memory:{cur_memory}")
        if abs(memory-cur_memory) > 50:
            logger.info(f"update_model------ memory:{memory} cur_memory:{cur_memory}")
            optimal_runtime.update_model(10,memory*1024**2)
            cur_memory = memory
        time.sleep(1)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    set_random_seed(0)
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            # save file
            filename = secure_filename(file.filename)
            save_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_file_path)
            # refer
            img_pil = Image.open(save_file_path)  # PIL.Image.Image对象
            img_pil = transform(img_pil)
            data = torch.stack([img_pil], dim=0)
            output = model(data)
            output = torch.argmax(output, 1).item()
            return jsonify({"class_num": str(output),"success":True,"class_name":out_info.get(output)})
        else:
            return jsonify({"success":False,"info":"illegal file or filename"})
    else:
        return render_template('upload.html')


if __name__ == '__main__':
    model,transform,optimal_runtime = init_model()
    threading.Thread(target=runtime_ajdust_model,args=(optimal_runtime,)).start()
    app.run(host='0.0.0.0', port=8080)


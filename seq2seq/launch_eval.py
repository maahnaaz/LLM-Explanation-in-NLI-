import os
import argparse
import numpy as np
from shutil import copy2

import torch
from torch.autograd import Variable
import torch.nn as nn


import eval_sent_embeddings_labels_in_expl

import streamtologger
import sys
sys.path.append("..")
GLOVE_PATH = '../dataset/GloVe/glove.840B.300d.txt'

parser = argparse.ArgumentParser(description='eval')
# paths
parser.add_argument("--directory", type=str, default='')
parser.add_argument("--state_path", type=str, default='')
parser.add_argument("--eval_batch_size", type=int, default=32)
parser.add_argument("--train_snli_classif",
                    action='store_true', dest='train_snli_classif')
parser.add_argument("--use_prototype_senteval",
                    action='store_true', dest='use_prototype_senteval')
parser.add_argument("--do_image_caption",
                    action='store_true', dest='do_image_caption')
parser.add_argument("--cudnn_nondeterministic",
                    action='store_false', dest='cudnn_deterministic')


eval_params = parser.parse_args()

streamtologger.redirect(target=eval_params.directory + '/log_eval.txt')

state = torch.load(os.path.join(eval_params.directory, eval_params.state_path))
model_config = state['config_model']

model_state_dict = state['model_state']
params = state['params']

params.eval_batch_size = eval_params.eval_batch_size
params.current_run_dir = eval_params.directory
params.train_snli_classif = eval_params.train_snli_classif
params.use_prototype_senteval = eval_params.use_prototype_senteval
params.do_image_caption = eval_params.do_image_caption
params.cudnn_deterministic = eval_params.cudnn_deterministic


"""
SEED
"""
# set gpu device, changed by Mahnaaz
torch.cuda.set_device(params.gpu_id)
device = torch.device('cuda:{}'.format(params.gpu_id) if torch.cuda.is_available() else 'cpu')

np.random.seed(params.seed)
# Changed by Mahnaaz from torch.cuda.manual_seed(params.seed) and torch.manual_seed(params.seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(params.seed)
elif device.type == 'cpu':
    torch.manual_seed(params.seed)
else:
    raise ValueError("Unsupported device type. Please use 'cuda' or 'cpu'.")

# CUDNN deterministic
torch.backends.cudnn.deterministic = params.cudnn_deterministic


copy2('launch_eval.py', eval_params.directory)
copy2('eval_sent_embeddings_labels_in_expl.py', eval_params.directory)

#import sys
#sys.path.insert(0, eval_params.directory)
import models_esnli_init
esnli_net = models_esnli_init.eSNLINet(model_config).to(device)
esnli_net.load_state_dict(model_state_dict)


# criterion
pad_idx = model_config['word_index']["<p>"]
criterion_expl = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)
criterion_expl.size_average = False

eval_sent_embeddings_labels_in_expl.eval_all(esnli_net, criterion_expl, params)

txt_file = 'DONE_eval.txt'
file = os.path.join(params.current_run_dir, txt_file)
f = open(file, 'w')
f.write("DONE")
f.close()

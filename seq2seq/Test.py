import os
import math
import csv
import time
import numpy as np
import datetime

import torch
from torch.autograd import Variable
import torch.nn as nn

from data_label_in_expl import get_dev_test_original_expl, get_batch, build_vocab, get_target_expl_batch, get_dev_test_with_expl, get_dev_or_test_without_expl, NLI_DIC_LABELS, NLI_LABELS_TO_NLI, get_train, get_word_dict
from utils.mutils import get_optimizer, makedirs, pretty_duration, get_sentence_from_indices, get_key_from_val, n_parameters, remove_file, assert_sizes, permute, bleu_prediction
from models_esnli_init import eSNLINet

esnli_path = '../dataset/eSNLI/'
GLOVE_PATH = '../dataset/GloVe/glove.840B.300d.txt'
print_every = 100
eval_batch_size = 64
preproc = 'preproc1_'
min_freq = 15
date = datetime.datetime.fromtimestamp(time.time())
torch.cuda.set_device(torch.cuda.current_device())
device = torch.device('cuda:{}'.format(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu')

# Data
"""
ALL DATA, some will only be needed for eval for we want to build glove vocab once
"""
train = get_train(esnli_path, preproc, min_freq, n_train=-1)

# Dev
snli_dev = get_dev_test_with_expl(
        esnli_path, 'dev', preproc, min_freq)
    
# Test 
snli_test = get_dev_test_with_expl(esnli_path
    , 'test', preproc, min_freq)

expl_sentences = train['expl_1'] + snli_dev['expl_1'] + snli_dev['expl_2'] + \
    snli_dev['expl_3'] + snli_test['expl_1'] + \
    snli_test['expl_2'] + snli_test['expl_3']
word_index = get_word_dict(expl_sentences)
    
snli_sentences = snli_dev['s1'] + snli_dev['s2'] + snli_dev['expl_1'] + snli_dev['expl_2'] + snli_dev['expl_3'] + \
        snli_test['s1'] + snli_test['s2'] + snli_test['expl_1'] + \
        snli_test['expl_2'] + snli_test['expl_3']
word_vec = build_vocab(snli_sentences, GLOVE_PATH)


for split in ['s1', 's2', 'expl_1', 'expl_2', 'expl_3']:
    snli_dev[split] = np.asarray([['<s>'] + [word for word in sent.split()
                                           if word in word_vec] + ['</s>'] for sent in snli_dev[split]], dtype="object")

expl_no_unk_test = get_dev_test_original_expl('../dataset/eSNLI/', 'dev')


# for split in ['s1', 's2', 'expl_1', 'expl_2', 'expl_3']:
#    snli_test[split] = np.asarray([['<s>'] + [word for word in sent.split()
#                                           if word in word_vec] + ['</s>'] for sent in snli_test[split]], dtype="object")
#
#expl_no_unk_test = get_dev_test_original_expl('../dataset/eSNLI/', 'test')
                         
# loss labels
criterion_labels = nn.CrossEntropyLoss()
criterion_labels.size_average = False
# loss expl
criterion_expl = nn.CrossEntropyLoss(ignore_index=word_index["<p>"])
criterion_expl.size_average = False
# cuda by default, changed by Mahnaaz
if device.type == 'cuda':
    criterion_labels.cuda()
    criterion_expl.cuda()

# load best model from state
# model config
config_nli_model = {
    'word_emb_dim': 300,
    'enc_rnn_dim': 2048,
    'dec_rnn_dim': 512,
    'dpout_enc': 0,
    'dpout_dec': 0.5,
    'dpout_fc': 512,
    'fc_dim': 512,
    'bsize': 64,
    'n_classes': 3,
    'pool_type': 'max',
    'nonlinear_fc': False,
    'encoder_type': 'BLSTMEncoder',
    'decoder_type': 'lstm',
    'use_cuda': True,  # Changed from True by Mahnaaz
    'n_vocab': len(word_index),
    'word_vec': word_vec,
    'word_index': word_index,
    'max_T_decoder': 40,
    'use_vocab_proj': False,
    'vocab_proj_dim': 512,
    'use_init': True,
    'n_layers_dec': 1,
    'only_diff_prod': False,
    'use_smaller_inp_dec_dim': False,
    'smaller_inp_dec_dim': 2048,
    'use_diff_prod_sent_embed': True,
    'relu_before_pool': False,

}





# model
esnli_net = eSNLINet(config_nli_model).to(device)

best_model_path = '../models/eSNLI_PredictAndExplain/state_dict_best_devacc__devACC84.370_devppl10.200__epoch_12_model.pt'
state_best_model = torch.load(best_model_path, encoding='latin1')['model_state']
esnli_net.load_state_dict(state_best_model)

# Evaluation
esnli_net.eval()
correct = 0.
correct_labels_expl = 0.
cum_test_ppl = 0
cum_test_n_words = 0

# Added by Mahnaz
s1 = snli_dev['s1'][:]
s2 = snli_dev['s2'][:]
expl_1 = snli_dev['expl_1'][:]
expl_2 = snli_dev['expl_2'][:]
expl_3 = snli_dev['expl_3'][:]
label = snli_dev['label'][:]
label_expl = snli_dev['label_expl'][:]

for i in range(0, len(s1), eval_batch_size):
    # prepare batch
    s1_batch, s1_len = get_batch(
        s1[i:i + eval_batch_size], word_vec)
    s2_batch, s2_len = get_batch(
        s2[i:i + eval_batch_size], word_vec)
    s1_batch, s2_batch = Variable(
        s1_batch.to(device)), Variable(s2_batch.to(device))
    tgt_label_batch = Variable(torch.LongTensor(
        label[i:i + eval_batch_size]).to(device))
    # print example
    if i % print_every == 0:
        print("SNLI DEV example")
        print("Sentence1:  ", ' '.join(s1[i]), " LENGTH: ", s1_len[0])
        print("Sentence2:  ", ' '.join(s2[i]), " LENGTH: ", s2_len[0])
        print("Gold label:  ", get_key_from_val(label[i], NLI_DIC_LABELS))
    
    out_lbl = [0, 1, 2, 3]
    for index in range(1, 4):
        expl = eval("expl_" + str(index))
        input_expl_batch, _ = get_batch(expl[i:i + eval_batch_size], word_vec)
        input_expl_batch = Variable(input_expl_batch[:-1].to(device))
        if i % print_every == 0:
            print("Explanation " + str(index) + " :  ", ' '.join(expl[i]))
            print("Predicted label by decoder " +
                str(index) + " :  ", ' '.join(expl[i][0]))
        tgt_expl_batch, lens_tgt_expl = get_target_expl_batch(
            expl[i:i + eval_batch_size], word_index)
        assert tgt_expl_batch.dim() == 2, "tgt_expl_batch.dim()=" + \
            str(tgt_expl_batch.dim())
        tgt_expl_batch = Variable(tgt_expl_batch.to(device))
        if i % print_every == 0:
            print("Target expl " + str(index) + " :  ", get_sentence_from_indices(
                word_index, tgt_expl_batch[:, 0]), " LENGHT: ", lens_tgt_expl[0])
        
        # model forward, tgt_label is None for both v1 and v2 bcs it's test time for v2
        out_expl, out_lbl[index - 1] = esnli_net(
            (s1_batch, s1_len), (s2_batch, s2_len), input_expl_batch, 'teacher')
        # ppl
        loss_expl = criterion_expl(out_expl.view(out_expl.size(0) * out_expl.size(
            1), -1), tgt_expl_batch.view(tgt_expl_batch.size(0) * tgt_expl_batch.size(1)))
        cum_test_n_words += lens_tgt_expl.sum()
        cum_test_ppl += loss_expl.item()
        answer_idx = torch.max(out_expl, 2)[1]
        if i % print_every == 0:
            print("Decoded explanation " + str(index) + " :  ",
                  get_sentence_from_indices(word_index, answer_idx[:, 0]))
            print("\n")
        # Print toa file
        with open(f"Explanations_{date}.txt", "a") as f:
            for count in range(answer_idx.shape[1]):
                f.write(get_sentence_from_indices(word_index, answer_idx[:, count]) + "\n")
            f.write(f" ----------> {index}" + "\n")
        f.close()
            
    #pred_expls, out_lbl[3] = esnli_net(
    #    (s1_batch, s1_len), (s2_batch, s2_len), input_expl_batch, mode="forloop")
    #if i % print_every == 0:
    #    print("Fully decoded explanation: ",
    #          pred_expls[0].strip().split()[1:-1])
    #    print("Predicted label from decoder: ",
    #         pred_expls[0].strip().split()[0])

    #for b in range(len(pred_expls)):
    #    assert tgt_label_expl_batch[b] in [
    #        'entailment', 'neutral', 'contradiction']
    #    if len(pred_expls[b]) > 0:
    #        words = pred_expls[b].strip().split()
    #        assert words[0] in ['entailment',
    #                            'neutral', 'contradiction'], words[0]
    #        if words[0] == tgt_label_expl_batch[b]:
    #            correct_labels_expl += 1

    #assert(torch.equal(out_lbl[0], out_lbl[1]))
    #assert(torch.equal(out_lbl[1], out_lbl[2]))
    #assert(torch.equal(out_lbl[2], out_lbl[3]))

    # accuracy
    pred = out_lbl[0].data.max(1)[1]
    if i % print_every == 0:
       print("Predicted label from classifier:  ",
              get_key_from_val(pred[0], NLI_DIC_LABELS), "\n\n\n")
    correct += pred.long().eq(tgt_label_batch.data.long()).cpu().sum()
  
eval_acc = round(100 * correct.item() / len(s1), 2)
# eval_acc_label_expl = round(100 * correct_labels_expl / len(s1), 2)
eval_ppl = math.exp(cum_test_ppl / cum_test_n_words)
# bleu_score = 100 * bleu_prediction(expl_csv, expl_no_unk_test)

print('SNLI accuracy: ', eval_acc, 'ppl: ', eval_ppl) 
# 'bleu score: ', bleu_score, 'eval_acc_label_expl: ', eval_acc_label_expl

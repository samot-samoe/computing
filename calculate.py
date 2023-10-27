import re
import subprocess
from math import ceil
import numpy as np
import pandas as pd
# from tqdm.notebook import tqdm
import multiprocessing
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import os
from tqdm import tqdm
from stats_count import *
from grab_weights import grab_attention_weights, text_preprocessing
import warnings
from config import *
from utils import *
warnings.filterwarnings('ignore')
import timeit
import ripser_count
import json
import itertools
from collections import defaultdict
from multiprocessing import Pool,Process, Queue

np.random.seed(42)

max_tokens_amount  = 128 # The number of tokens to which the tokenized text is truncated / padded.
stats_cap          = 500 # Max value that the feature can take. Is NOT applicable to Betty numbers.
    
layers_of_interest = [i for i in range(12)]  # Layers for which attention matrices and features on them are 
                                             # calculated. For calculating features on all layers, leave it be
                                             # [i for i in range(12)].
stats_name = "s_e_v_c_b0b1" # The set of topological features that will be count (see explanation below)

thresholds_array = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75] # The set of thresholds
thrs = len(thresholds_array)                           # ("t" in the paper)

model_path = tokenizer_path = model_name

subset = subset          # .csv file with the texts, for which we count topological features
input_dir = input_dir  # Name of the directory with .csv file
output_dir = output_dir # Name of the directory with calculations results

prefix = output_dir + subset

r_file     = output_dir + 'attentions/' + subset  + "_all_heads_" + str(len(layers_of_interest)) + "_layers_MAX_LEN_" + \
             str(max_tokens_amount) + "_" + model_path.split("/")[-1]
# Name of the file for attention matrices weights

stats_file = output_dir + 'features/' + subset + "_all_heads_" + str(len(layers_of_interest)) + "_layers_" + stats_name \
             + "_lists_array_" + str(thrs) + "_thrs_MAX_LEN_" + str(max_tokens_amount) + \
             "_" + model_path.split("/")[-1] + '.npy'
# Name of the file for topological features array
ripser_file = output_dir + 'features/' + subset + "_all_heads_" + str(len(layers_of_interest)) + "_layers" \
             + "_MAX_LEN_" + str(max_tokens_amount) + \
             "_" + model_path.split("/")[-1] + "_ripser" + '.npy'

data = pd.read_csv(csv_location)

model = BertForSequenceClassification.from_pretrained(model_path, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=True)
device = device
model = model.to(device)
MAX_LEN = max_tokens_amount

def get_token_length(batch_texts):
    inputs = tokenizer.batch_encode_plus(batch_texts,
       return_tensors='pt',
       add_special_tokens=True,
       max_length=MAX_LEN,             # Max length to truncate/pad
       pad_to_max_length=True,         # Pad sentence to max length
       truncation=True
    )
    inputs = inputs['input_ids'].numpy()
    n_tokens = []
    indexes = np.argwhere(inputs == tokenizer.pad_token_id)
    for i in range(inputs.shape[0]):
        ids = indexes[(indexes == i)[:, 0]]
        if not len(ids):
            n_tokens.append(MAX_LEN)
        else:
            n_tokens.append(ids[0, 1])
    return n_tokens

data['tokenizer_length'] = get_token_length(data[col_with_text].values)
ntokens_array = data['tokenizer_length'].values
#-----------------------------------------attention extraction---------------------------

batch_size = batch_size # batch size
number_of_batches = ceil(len(data[col_with_text]) / batch_size)
DUMP_SIZE = dump_size # number of batches to be dumped
batched_sentences = np.array_split(data[col_with_text].values, number_of_batches)
number_of_files = ceil(number_of_batches / DUMP_SIZE)
adj_matricies = []
adj_filenames = []


# #----------------------------Calculating barcodes------------------------------------------
dim = 1
lower_bound = 1e-3

prefix = output_dir + subset

r_file     = output_dir + 'attentions/' + subset  + "_all_heads_" + str(len(layers_of_interest)) + "_layers_MAX_LEN_" + \
             str(max_tokens_amount) + "_" + model_path.split("/")[-1]
# Name of the file for attention matrices weights

barcodes_file = output_dir + 'barcodes/' + subset  + "_all_heads_" + str(len(layers_of_interest)) + "_layers_MAX_LEN_" + \
             str(max_tokens_amount) + "_" + model_path.split("/")[-1]
# Name of the file for barcodes information
def format_barcodes(barcodes):
    """Reformat barcodes to json-compatible format"""
    return [{d: b[d].tolist() for d in b} for b in barcodes]
def get_only_barcodes(adj_matricies, ntokens_array, dim, lower_bound):
    """Get barcodes from adj matricies for each layer, head"""
    barcodes = {}
    layers, heads = range(adj_matricies.shape[1]), range(adj_matricies.shape[2])
    
    for (layer, head) in itertools.product(layers, heads):
        matricies = adj_matricies[:, layer, head, :, :]
        
        barcodes[(layer, head)] = ripser_count.get_barcodes(matricies, ntokens_array, dim, lower_bound, (layer, head))
        
    return barcodes
def save_barcodes(barcodes, filename):
    """Save barcodes to file"""
    formatted_barcodes = defaultdict(dict)
    for layer, head in barcodes:
        formatted_barcodes[layer][head] = format_barcodes(barcodes[(layer, head)])
    json.dump(formatted_barcodes, open(filename, 'w'))\
    
def unite_barcodes(barcodes, barcodes_part):
    """Unite 2 barcodes"""
    for (layer, head) in barcodes_part:
        barcodes[(layer, head)].extend(barcodes_part[(layer, head)])
    return barcodes

def split_matricies_and_lengths(adj_matricies, ntokens, number_of_splits):
    splitted_ids = np.array_split(np.arange(ntokens.shape[0]), number_of_splits)
    splitted = [(adj_matricies[ids], ntokens[ids]) for ids in splitted_ids]
    return splitted

def get_list_of_ids(sentences, tokenizer):
    inputs = tokenizer.batch_encode_plus([text_preprocessing(s) for s in sentences],
                                       add_special_tokens=True,
                                       max_length=MAX_LEN,             # Max length to truncate/pad
                                       pad_to_max_length=True,         # Pad sentence to max length)
                                       truncation=True
                                      )
    return np.array(inputs['input_ids'])

# def subprocess_wrap(queue, function, args):
#     queue.put(function(*args))
# #     print("Putted in Queue")
#     queue.close()
#     exit()

feature_list=['self', 'beginning', 'prev', 'next', 'comma', 'dot']
num_of_workers = num_of_workers
# pool_stats = Pool(num_of_workers)
# pool_features = Pool(num_of_workers)
queue = Queue()
number_of_splits = 2

stats_tuple_lists_array = []

features_array = []
j = 0


for i in tqdm(range(number_of_batches), desc="Weights calc"):
    # Извлечение весов attention
    attention_w = grab_attention_weights(model, tokenizer, batched_sentences[i], max_tokens_amount, device)
    adj_matricies.append(attention_w)
    
    if (i+1) % DUMP_SIZE == 0 or i == number_of_batches - 1: # dumping
        adj_matricies = np.concatenate(adj_matricies, axis=1)
        adj_matricies = np.swapaxes(adj_matricies, axis1=0, axis2=1) # sample X layer X head X n_token X n_token
        filename = r_file + "_part" + str(ceil(i/DUMP_SIZE)) + "of" + str(number_of_files) + '.npy'
        # np.save(filename, adj_matricies)
        adj_filenames.append(filename)
        
        # Вычисление признаков
        ntokens = ntokens_array[j*batch_size*DUMP_SIZE : (j+1)*batch_size*DUMP_SIZE]
        splitted = split_matricies_and_lengths(adj_matricies, ntokens, num_of_workers)
        args = [(m, thresholds_array, ntokens, stats_name.split("_"), stats_cap) for m, ntokens in splitted]
        # stats_tuple_lists_array_part = pool_stats.starmap(
            # count_top_stats, args
        # )
        # pool_stats.close()
        # pool_stats.join()
        with multiprocessing.Pool(processes=num_of_workers) as pool:
            stats_tuple_lists_array_part = pool.starmap(count_top_stats, args)

        stats_tuple_lists_array.append(np.concatenate([_ for _ in stats_tuple_lists_array_part], axis=3))
        
        
        # Сохранение признаков
        # print(args)
        
        #ЭТО НУЖНО ВЫНЕСТИ ИЗ IF 
        # Вычисление баркодов
        barcodes = defaultdict(list)
        splitted = split_matricies_and_lengths(adj_matricies, ntokens, number_of_splits)
    
        barcodes_part =  get_only_barcodes(adj_matricies, ntokens, dim, lower_bound)
        barcodes = unite_barcodes(barcodes, barcodes_part)

        part = filename.split('_')[-1].split('.')[0]
        save_barcodes(barcodes, barcodes_file + '_' + part + '.json')
        
        # Удаление файла с весами attention матриц после использования
        # os.remove(filename)

        # batch_size = adj_matricies.shape[0]
        sentences = data[col_with_text].values[j*batch_size:(j+1)*batch_size]
        splitted_indexes = np.array_split(np.arange(batch_size), num_of_workers)
        splitted_list_of_ids = [
            get_list_of_ids(sentences[indx], tokenizer)
            for indx in tqdm(splitted_indexes, desc=f"Calculating token ids on iter {j} from {len(adj_filenames)}")
        ]
        splitted_adj_matricies = [adj_matricies[indx] for indx in splitted_indexes]

        args = [(m, feature_list, list_of_ids) for m, list_of_ids in zip(splitted_adj_matricies, splitted_list_of_ids)]

        # features_array_part = pool_features.starmap(
        #     calculate_features_t, args
        # )
        # pool_features.close()
        # pool_features.join()
        with multiprocessing.Pool(processes=num_of_workers) as pool:
            features_array_part = pool.starmap(calculate_features_t, args)
        features_array.append(np.concatenate([_ for _ in features_array_part], axis=3))
         # Очистка для следующей итерации
        adj_matricies = []
        j+=1

stats_tuple_lists_array_concatenated = np.concatenate(stats_tuple_lists_array, axis=3)
np.save(stats_file, stats_tuple_lists_array_concatenated)        
features_array = np.concatenate(features_array, axis=3)
np.save(output_dir+'features/' + attention_name + "_template.npy", features_array)

ripser_feature_names=[
    'h0_s',
    'h0_e',
    'h0_t_d',
    'h0_n_d_m_t0.75',
    'h0_n_d_m_t0.5',
    'h0_n_d_l_t0.25',
    'h1_t_b',
    'h1_n_b_m_t0.25',
    'h1_n_b_l_t0.95',
    'h1_n_b_l_t0.70',
    'h1_s',
    'h1_e',
    'h1_v',
    'h1_nb'
]
adj_filenames = [
    output_dir + 'barcodes/' + filename
    for filename in os.listdir(output_dir + 'barcodes/') if r_file.split('/')[-1] == filename.split('_part')[0]
]
adj_filenames = sorted(adj_filenames, key = lambda x: int(x.split('_')[-1].split('of')[0][4:].strip()))
# print(adj_filenames)

features_array = []

for filename in tqdm(adj_filenames, desc='Calculating ripser++ features'):
    barcodes = json.load(open(filename))
    # print(f"Barcodes loaded from: {filename}", flush=True)
    features_part = []
    for layer in barcodes:
        features_layer = []
        for head in barcodes[layer]:
            ref_barcodes = reformat_barcodes(barcodes[layer][head])
            features = ripser_count.count_ripser_features(ref_barcodes, ripser_feature_names)
            features_layer.append(features)
        features_part.append(features_layer)
    features_array.append(np.asarray(features_part))

# ripser_file = output_dir + 'features/' + subset + "_all_heads_" + str(len(layers_of_interest)) + "_layers" \
#              + "_MAX_LEN_" + str(max_tokens_amount) + \
#              "_" + model_path.split("/")[-1] + "_ripser" + '.npy'
features = np.concatenate(features_array, axis=2)
np.save(ripser_file, features)


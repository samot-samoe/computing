import re
import subprocess
from math import ceil
import numpy as np
import pandas as pd
# from tqdm.notebook import tqdm
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

model_path = tokenizer_path = 'ai-forever/ruBert-base'

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




# for i in tqdm(range(number_of_batches), desc="Weights calc"):
#     attention_w = grab_attention_weights(model, tokenizer, batched_sentences[i], max_tokens_amount, device)
#     # sample X layer X head X n_token X n_token
#     adj_matricies.append(attention_w)
#     if (i+1) % DUMP_SIZE == 0: # dumping
#         print(f'Saving: shape {adj_matricies[0].shape}')
#         adj_matricies = np.concatenate(adj_matricies, axis=1)
#         print("Concatenated")
#         adj_matricies = np.swapaxes(adj_matricies, axis1=0, axis2=1) # sample X layer X head X n_token X n_token
#         filename = r_file + "_part" + str(ceil(i/DUMP_SIZE)) + "of" + str(number_of_files) + '.npy'
#         print(f"Saving weights to : {filename}")
#         adj_filenames.append(filename)
#         np.save(filename, adj_matricies)
#         adj_matricies = []

# if len(adj_matricies):
#     filename = r_file + "_part" + str(ceil(i/DUMP_SIZE)) + "of" + str(number_of_files) + '.npy'
#     print(f'Saving: shape {adj_matricies[0].shape}')
#     adj_matricies = np.concatenate(adj_matricies, axis=1)
#     print("Concatenated")
#     adj_matricies = np.swapaxes(adj_matricies, axis1=0, axis2=1) # sample X layer X head X n_token X n_token
#     print(f"Saving weights to : {filename}")
#     np.save(filename, adj_matricies)

# adj_filenames = [
#     output_dir + 'attentions/' + filename
#     for filename in os.listdir(output_dir + 'attentions/') if r_file in (output_dir + 'attentions/' + filename)
# ]
# # sorted by part number
# adj_filenames = sorted(adj_filenames, key = lambda x: int(x.split('_')[-1].split('of')[0][4:].strip()))



# for i, filename in enumerate(tqdm(adj_filenames, desc='Вычисление признаков')):
#     adj_matricies = np.load(filename, allow_pickle=True)
#     ntokens = ntokens_array[i*batch_size*DUMP_SIZE : (i+1)*batch_size*DUMP_SIZE]
#     splitted = split_matricies_and_lengths(adj_matricies, ntokens, num_of_workers)
#     args = [(m, thresholds_array, ntokens, stats_name.split("_"), stats_cap) for m, ntokens in splitted]
#     stats_tuple_lists_array_part = pool.starmap(
#         count_top_stats, args
#     )
#     stats_tuple_lists_array.append(np.concatenate([_ for _ in stats_tuple_lists_array_part], axis=3))

# stats_tuple_lists_array = np.concatenate(stats_tuple_lists_array, axis=3)
# np.save(stats_file, stats_tuple_lists_array)


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
    print("layers and heads got")
    for (layer, head) in itertools.product(layers, heads):
        matricies = adj_matricies[:, layer, head, :, :]
        print('matricies got')
        barcodes[(layer, head)] = ripser_count.get_barcodes(matricies, ntokens_array, dim, lower_bound, (layer, head))
        print("barcodes got")
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

def subprocess_wrap(queue, function, args):
    queue.put(function(*args))
#     print("Putted in Queue")
    queue.close()
    exit()

feature_list=['self', 'beginning', 'prev', 'next', 'comma', 'dot']
# num_of_workers = num_of_workers
pool = Pool(num_of_workers)
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
        stats_tuple_lists_array_part = pool.starmap(
            count_top_stats, args
        )
        stats_tuple_lists_array.append(np.concatenate([_ for _ in stats_tuple_lists_array_part], axis=3))
        
        # Сохранение признаков
        stats_tuple_lists_array_concatenated = np.concatenate(stats_tuple_lists_array, axis=3)
        np.save(stats_file, stats_tuple_lists_array_concatenated)

        # Вычисление баркодов
        barcodes = defaultdict(list)
        splitted = split_matricies_and_lengths(adj_matricies, ntokens, number_of_splits)
        # for matricies, ntokens in tqdm(splitted, leave=False):
            # p = Process(
            #     target=subprocess_wrap,
            #     args=(
            #         queue,
            #         get_only_barcodes,
            #         (matricies, ntokens, dim, lower_bound)
            #     )
            # )
            # p.start()
            # barcodes_part = queue.get() # block until putted and get barcodes from the queue
            # p.join() # release resources
            # p.close() # releasing resources of ripser
        print(len(ntokens))
        print(adj_matricies.shape)
        barcodes_part =  get_only_barcodes(adj_matricies, ntokens, dim, lower_bound)
        barcodes = unite_barcodes(barcodes, barcodes_part)
        part = filename.split('_')[-1].split('.')[0]
        save_barcodes(barcodes, barcodes_file + '_' + part + '.json')
        
        # Удаление файла с весами attention матриц после использования
        # os.remove(filename)

        # Очистка для следующей итерации
        adj_matricies = []
        j+=1

# for i in tqdm(range(number_of_batches), desc="Weights calc"):
#     # Извлечение весов attention
#     attention_w = grab_attention_weights(model, tokenizer, batched_sentences[i], max_tokens_amount, device)
#     adj_matricies.append(attention_w)
    
#     if (i+1) % DUMP_SIZE == 0 or i == number_of_batches - 1: # dumping
        
#         adj_matricies = np.concatenate(adj_matricies, axis=1)
#         adj_matricies = np.swapaxes(adj_matricies, axis1=0, axis2=1) # sample X layer X head X n_token X n_token
#         filename = r_file + "_part" + str(ceil(i/DUMP_SIZE)) + "of" + str(number_of_files) + '.npy'
#         # np.save(filename, adj_matricies)
#         # adj_filenames.append(filename)
        
#         # Вычисление признаков
#         ntokens = ntokens_array[j*batch_size*DUMP_SIZE : (j+1)*batch_size*DUMP_SIZE]
#         # print(f'нтокенс_array {ntokens }')
#         # print(adj_matricies.shape) #(80, 12, 12, 128, 128)
#         splitted = split_matricies_and_lengths(adj_matricies, ntokens, num_of_workers)
#         args = [(m, thresholds_array, ntokens, stats_name.split("_"), stats_cap) for m, ntokens in splitted]
#         stats_tuple_lists_array_part = pool.starmap(
#             count_top_stats, args
#         )
#         stats_tuple_lists_array.append(np.concatenate([_ for _ in stats_tuple_lists_array_part], axis=3))
        
#         # Сохранение признаков
#         stats_tuple_lists_array_concatenated = np.concatenate(stats_tuple_lists_array, axis=3)
#         np.save(stats_file, stats_tuple_lists_array_concatenated)

#         print(adj_matricies)
#         # Вычисление баркодов
#         barcodes = defaultdict(list)
#         splitted = split_matricies_and_lengths(adj_matricies, ntokens, number_of_splits)
#         # print(len(splitted))
#         for matricies, ntokens in tqdm(splitted, leave=False):
#             # print(len(matricies)) #13
#             # print(len(ntokens)) #13
#             # p = Process(
#             #     target=subprocess_wrap,
#             #     args=(
#             #         queue,
#             #         get_only_barcodes,
#             #         (matricies, ntokens, dim, lower_bound)
#             #     )
#             # )
#             # print("Process created")
#             # p.start()
#             # barcodes_part = queue.get() # block until putted and get barcodes from the queue
#             # print("Features got.")
#             # p.join() # release resources
#             # print("The process is joined.")
#             # p.close() # releasing resources of ripser
#             # print("The proccess is closed.")
#             barcodes_part = get_only_barcodes(matricies, ntokens, dim, lower_bound)

#             barcodes = unite_barcodes(barcodes, barcodes_part)
#             print(barcodes)
#         part = filename.split('_')[-1].split('.')[0]
#         save_barcodes(barcodes, barcodes_file + '_' + part + '.json')
        
#         # Вычисление дополнительных признаков
#         # print(adj_matricies.shape[0])
#         # batch_size = adj_matricies.shape[0]
#         sentences = data[col_with_text].values[j*batch_size:(j+1)*batch_size]
#         splitted_indexes = np.array_split(np.arange(batch_size), num_of_workers)
#         splitted_list_of_ids = [
#             get_list_of_ids(sentences[indx], tokenizer)
#             for indx in tqdm(splitted_indexes, desc=f"Calculating token ids on iter {i} from {number_of_batches}")
#         ]
#         splitted_adj_matricies = [adj_matricies[indx] for indx in splitted_indexes]

#         args = [(m, feature_list, list_of_ids) for m, list_of_ids in zip(splitted_adj_matricies, splitted_list_of_ids)]

#         features_array_part = pool.starmap(
#             calculate_features_t, args
#         )
#         features_array.append(np.concatenate([_ for _ in features_array_part], axis=3))
        
#         # Удаление файла с весами attention матриц после использования
#         # os.remove(filename)

#     # Очистка для следующей итерации
#         adj_matricies = []
#         j+=1

# # Сохранение дополнительных признаков после цикла
# features_array = np.concatenate(features_array, axis=3)
# np.save(output_dir + attention_name + "_template.npy", features_array)
# pool.close()


# def worker_func(args):
#     queue, func, params = args
#     result = func(*params)
#     queue.put(result)

# if __name__ == '__main__':
#     pool = Pool(num_of_workers)
#     queue = Queue()
#     number_of_splits = 2

#     stats_tuple_lists_array = []
#     features_array = []
#     j = 0
#     for i in tqdm(range(number_of_batches), desc="Weights calc"):
#         # Извлечение весов attention
#         attention_w = grab_attention_weights(model, tokenizer, batched_sentences[i], max_tokens_amount, device)
#         adj_matricies.append(attention_w)
        
#         if (i+1) % DUMP_SIZE == 0 or i == number_of_batches - 1: # dumping
#             j+=1
#             adj_matricies = np.concatenate(adj_matricies, axis=1)
#             adj_matricies = np.swapaxes(adj_matricies, axis1=0, axis2=1) # sample X layer X head X n_token X n_token
#             filename = r_file + "_part" + str(ceil(i/DUMP_SIZE)) + "of" + str(number_of_files) + '.npy'
            
#             # Вычисление признаков
#             ntokens = ntokens_array[j*batch_size*DUMP_SIZE : (j+1)*batch_size*DUMP_SIZE]
#             print(adj_matricies.shape) #(4, 12, 12, 128, 128)
#             splitted = split_matricies_and_lengths(adj_matricies, ntokens, num_of_workers)
#             args = [(queue, count_top_stats, (m, thresholds_array, ntokens, stats_name.split("_"), stats_cap)) for m, ntokens in splitted]
#             pool.map(worker_func, args)
            
#             stats_tuple_lists_array_part = [queue.get() for _ in range(len(args))]
#             stats_tuple_lists_array.append(np.concatenate([_ for _ in stats_tuple_lists_array_part], axis=3))
            
#             # Сохранение признаков
#             stats_tuple_lists_array_concatenated = np.concatenate(stats_tuple_lists_array, axis=3)
#             np.save(stats_file, stats_tuple_lists_array_concatenated)

#             # Вычисление баркодов
#             barcodes = defaultdict(list)
#             args = [(queue, get_only_barcodes, (matricies, ntokens, dim, lower_bound)) for matricies, ntokens in splitted]
#             pool.map(worker_func, args)

#             for _ in range(len(args)):
#                 barcodes_part = queue.get()
#                 barcodes = unite_barcodes(barcodes, barcodes_part)

#             part = filename.split('_')[-1].split('.')[0]
#             save_barcodes(barcodes, barcodes_file + '_' + part + '.json')
            
#             # Вычисление дополнительных признаков
#             sentences = data[col_with_text].values[j*batch_size:(j+1)*batch_size]
#             splitted_indexes = np.array_split(np.arange(batch_size), num_of_workers)
#             splitted_list_of_ids = [
#                 get_list_of_ids(sentences[indx], tokenizer)
#                 for indx in tqdm(splitted_indexes, desc=f"Calculating token ids on iter {i} from {number_of_batches}")
#             ]
#             splitted_adj_matricies = [adj_matricies[indx] for indx in splitted_indexes]

#             args = [(queue, calculate_features_t, (m, feature_list, list_of_ids)) for m, list_of_ids in zip(splitted_adj_matricies, splitted_list_of_ids)]
#             pool.map(worker_func,args)

#             features_array_part=[queue.get() for _ in range(len(args))]
#             features_array.append(np.concatenate([_ for _ in features_array_part], axis=3))
            
#         adj_matricies=[]

#     features_array=np.concatenate(features_array,axis=3)
#     np.save(output_dir+attention_name+"_template.npy",features_array)
#     pool.close()


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



features_array = []

for filename in tqdm(adj_filenames, desc='Calculating ripser++ features'):
    barcodes = json.load(open(filename))
    print(f"Barcodes loaded from: {filename}", flush=True)
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
np.save(ripser_file, features)

# num_of_workers = 1
# pool = Pool(num_of_workers)
# feature_list = ['self', 'beginning', 'prev', 'next', 'comma', 'dot']
# features_array = []

# for i, filename in tqdm(list(enumerate(adj_filenames)), desc='Features calc'):
#     adj_matricies = np.load(filename, allow_pickle=True)
#     batch_size = adj_matricies.shape[0]
#     sentences = texts['Text'].values[i*batch_size:(i+1)*batch_size]
#     splitted_indexes = np.array_split(np.arange(batch_size), num_of_workers)
#     splitted_list_of_ids = [
#         get_list_of_ids(sentences[indx], tokenizer)
#         for indx in tqdm(splitted_indexes, desc=f"Calculating token ids on iter {i} from {len(adj_filenames)}")
#     ]
#     splitted_adj_matricies = [adj_matricies[indx] for indx in splitted_indexes]

#     args = [(m, feature_list, list_of_ids) for m, list_of_ids in zip(splitted_adj_matricies, splitted_list_of_ids)]

#     features_array_part = pool.starmap(
#         calculate_features_t, args
#     )
#     features_array.append(np.concatenate([_ for _ in features_array_part], axis=3))
# features_array = np.concatenate(features_array, axis=3)
# np.save(output_dir + attention_name + "_template.npy", features_array)
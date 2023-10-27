subset = "train_atd"           # .csv file with the texts, for which we count topological features
input_dir = "tda/tda4atd/input_dir/"  # Name of the directory with .csv file
output_dir = "tda/tda4atd/output_dir/"
csv_location = 'tda/tda4atd/train_x1000.csv'
device = 'cuda'
model_name = 'ai-forever/ruBert-base'
# attention_dir = '/content/drive/MyDrive/tda/testing/attentions/'
attention_name = 'train_atd_all_heads_12_layers_MAX_LEN_128_ruBert-base'
col_with_text = 'Text' # name of column with text in df
batch_size = 4 #batch size
dump_size = 20  # number of batches to be saved as barcode
num_of_workers = 2

import numpy as np
from multiprocessing import Process, Queue
import itertools
from collections import defaultdict
# import ripser_count
import json

def cutoff_matrix(matrix, ntokens):
    """Return normalized submatrix of first n_tokens"""
    matrix = matrix[:ntokens, :ntokens]
    matrix /= matrix.sum(axis=1, keepdims=True)
    return matrix

def function_for_v(list_of_v_degrees_of_graph):
    return sum(map(lambda x: np.sqrt(x*x), list_of_v_degrees_of_graph))

def split_matricies_and_lengths(adj_matricies, ntokens_array, num_of_workers):
    splitted_adj_matricies = np.array_split(adj_matricies, num_of_workers)
    splitted_ntokens = np.array_split(ntokens_array, num_of_workers)
    assert all([len(m)==len(n) for m, n in zip(splitted_adj_matricies, splitted_ntokens)]), "Split is not valid!"
    return zip(splitted_adj_matricies, splitted_ntokens)





def reformat_barcodes(barcodes):
    """Return barcodes to their original format"""
    formatted_barcodes = []
    for barcode in barcodes:
        formatted_barcode = {}
        for dim in barcode:
            formatted_barcode[int(dim)] = np.asarray(
                [(b, d) for b,d in barcode[dim]], dtype=[('birth', '<f4'), ('death', '<f4')]
            )
        formatted_barcodes.append(formatted_barcode)
    return formatted_barcodes







def matrix_distance(matricies, template, broadcast=True):
    """
    Calculates the distance between the list of matricies and the template matrix.
    Args:

    -- matricies: np.array of shape (n_matricies, dim, dim)
    -- template: np.array of shape (dim, dim) if broadcast else (n_matricies, dim, dim)

    Returns:
    -- diff: np.array of shape (n_matricies, )
    """
    diff = np.linalg.norm(matricies-template, ord='fro', axis=(1, 2))
    div = np.linalg.norm(matricies, ord='fro', axis=(1, 2))**2
    if broadcast:
        div += np.linalg.norm(template, ord='fro')**2
    else:
        div += np.linalg.norm(template, ord='fro', axis=(1, 2))**2
    return diff/np.sqrt(div)

def attention_to_self(matricies):
    """
    Calculates the distance between input matricies and identity matrix,
    which representes the attention to the same token.
    """
    _, n, m = matricies.shape
    assert n == m, f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.eye(n)
    return matrix_distance(matricies, template_matrix)

def attention_to_next_token(matricies):
    """
    Calculates the distance between input and E=(i, i+1) matrix,
    which representes the attention to the next token.
    """
    _, n, m = matricies.shape
    assert n == m, f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.triu(np.tri(n, k=1, dtype=matricies.dtype), k=1)
    return matrix_distance(matricies, template_matrix)

def attention_to_prev_token(matricies):
    """
    Calculates the distance between input and E=(i+1, i) matrix,
    which representes the attention to the previous token.
    """
    _, n, m = matricies.shape
    assert n == m, f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.triu(np.tri(n, k=-1, dtype=matricies.dtype), k=-1)
    return matrix_distance(matricies, template_matrix)

def attention_to_beginning(matricies):
    """
    Calculates the distance between input and E=(i+1, i) matrix,
    which representes the attention to [CLS] token (beginning).
    """
    _, n, m = matricies.shape
    assert n == m, f"Input matrix has shape {n} x {m}, but the square matrix is expected"
    template_matrix = np.zeros((n, n))
    template_matrix[:, 0] = 1.0
    return matrix_distance(matricies, template_matrix)

def attention_to_ids(matricies, list_of_ids, token_id):
    """
    Calculates the distance between input and ids matrix,
    which representes the attention to some particular tokens,
    which ids are in the `list_of_ids` (commas, periods, separators).
    """

    batch_size, n, m = matricies.shape
    EPS = 1e-7
    assert n == m, f"Input matrix has shape {n} x {m}, but the square matrix is expected"
#     assert len(list_of_ids) == batch_size, f"List of ids length doesn't match the dimension of the matrix"
    template_matrix = np.zeros_like(matricies)
    ids = np.argwhere(list_of_ids == token_id)
    if len(ids):
        batch_ids, row_ids = zip(*ids)
        template_matrix[np.array(batch_ids), :, np.array(row_ids)] = 1.0
        template_matrix /= (np.sum(template_matrix, axis=-1, keepdims=True) + EPS)
    return matrix_distance(matricies, template_matrix, broadcast=False)

def count_template_features(matricies, feature_list=['self', 'beginning', 'prev', 'next', 'comma', 'dot'], ids=None):
    features = []
    comma_id = 1010
    dot_id = 1012
    for feature in feature_list:
        if feature == 'self':
            features.append(attention_to_self(matricies))
        elif feature == 'beginning':
            features.append(attention_to_beginning(matricies))
        elif feature == 'prev':
            features.append(attention_to_prev_token(matricies))
        elif feature == 'next':
            features.append(attention_to_next_token(matricies))
        elif feature == 'comma':
            features.append(attention_to_ids(matricies, ids, comma_id))
        elif feature == 'dot':
            features.append(attention_to_ids(matricies, ids, dot_id))
    return np.array(features)

def calculate_features_t(adj_matricies, template_features, ids=None):
    """Calculate template features for adj_matricies"""
    features = []
    for layer in range(adj_matricies.shape[1]):
        features.append([])
        for head in range(adj_matricies.shape[2]):
            matricies = adj_matricies[:, layer, head, :, :]
            lh_features = count_template_features(matricies, template_features, ids) # samples X n_features
            features[-1].append(lh_features)
    return np.asarray(features) # layer X head X n_features X samples


import sys
import itertools
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import random
import logging
import os
from functools import reduce
import operator
from collections import Counter
import joblib
from datetime import datetime

from tqdm import tqdm
from collections import OrderedDict
from sklearn.base import ClassifierMixin
from sklearn.metrics import log_loss
from torch import softmax, log_softmax
from sklearn.preprocessing import StandardScaler

from pgmpy.readwrite import BIFWriter, BIFReader


def powerset(iterable, lim=3):
    s = list(iterable)
    return [tuple(sorted(i)) for i in itertools.chain.from_iterable(itertools.combinations(s, r)
                                                                    for r in range(lim + 1))]

def subset(iterable, n_elements):
    s = list(iterable)
    return [tuple(sorted(i)) for i in itertools.chain.from_iterable(itertools.combinations(s, r)
                                                                    for r in range(n_elements, n_elements + 1))]

def match_label(target_domains, label_name):
    found_key, found_label_name = None, None
    for key, value in target_domains.items(): 
        if label_name in value:
            found_key, found_label_name = key, label_name
            break
        else:
            continue 
    return found_key, found_label_name

def gen_pairwise_combinations(candidates):
    pairwise_combinations = []
    for i in range(len(candidates) - 1):
        for j in range(i + 1, len(candidates)):
            candidate_prev, candidate_cur = candidates[i], candidates[j]
            if len(set(candidate_prev).intersection(set(candidate_cur))) == len(candidate_prev) - 1:
                pairwise_combinations.append(tuple(sorted(set(candidate_prev + candidate_cur))))
    return list(set(pairwise_combinations))

def get_bn(model_path):
    reader = BIFReader(model_path)
    bn = reader.get_model()
    reader = BIFReader(model_path)
    bn = reader.get_model()
    return bn

def split_dict_by_key_range(data, key_range):
    keys_to_extract = set(map(str, key_range))  # Convert range keys to strings
    dict1 = {k: data[k] for k in keys_to_extract if k in data}  # Extracted keys
    dict2 = {k: data[k] for k in data if k not in keys_to_extract}  # Remaining keys
    return dict1, dict2

def extract_missing_positions(test_data_loader):
    """
    Detect missing positions in each instance and concatenate results.
    
    Args:
        test_data_loader: Object containing 'labels' and 'all_labels' as NumPy arrays.
    
    Returns:
        A 2D NumPy array where each row represents:
            [instance_index, missing_var_index_1, missing_var_index_2, ...]
    """
    all_labels = np.array(test_data_loader.all_labels)
    # labels contains missing values as -1
    labels = np.array(test_data_loader.labels)

    # Check if shapes match
    assert all_labels.shape == labels.shape, "Shape mismatch between all_labels and labels!"

    num_instances, _ = all_labels.shape  # Get shape of the dataset
    missing_data_list = []  # Store missing data information

    for i in range(num_instances):  # Loop through each instance
        missing_indices = np.where(all_labels[i] != labels[i])[0]  # Find missing positions

        if len(missing_indices) > 0:  # If there are missing values
            missing_data_list.append(np.concatenate(([i], missing_indices)))  
            # Store instance index followed by missing positions

    # Convert list to structured 2D NumPy array (rows = instances, cols = missing positions)
    if len(missing_data_list) > 0:
        missing_data_array = np.array(missing_data_list, dtype=object)  # Use dtype=object for variable-length rows
    else:
        missing_data_array = np.empty((0, 0))  # Return empty array if no missing values

    return missing_data_array

def train_tabular_base_clfs(model, x_train, y_train, learning_rate=None, n_epochs=None):
    if isinstance(model, torch.nn.Module):
        if learning_rate is None or n_epochs is None:
            raise ValueError('Not specified training hyper-parameters')
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=0.1)
        for epoch in range(1, n_epochs + 1):
            model.train()
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            cll_train = -loss.item()
            loss.backward()
            optimizer.step()
    elif isinstance(model, ClassifierMixin):

        model.fit(x_train, y_train)

    elif isinstance(model, list):
        return
    else:
        raise NotImplementedError

def compute_cll_tabular_clfs(model, x_train, y_train, weight=None):
    if isinstance(model, torch.nn.Module):
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        model.eval()
        with torch.no_grad():
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            return -loss.item()
    elif isinstance(model, ClassifierMixin):
        pred_probs = model.predict_proba(x_train)

        if weight is None:
            cll = -log_loss(y_train, pred_probs, normalize=False, 
                            sample_weight=weight)
        else:
            cll = -log_loss(y_train, pred_probs, normalize=False)
        return cll
    elif isinstance(model, list):
        pred_probs = np.zeros((len(y_train), len(model)))
        pred_probs[np.arange(len(y_train))] = model
        return -log_loss(y_train, pred_probs, normalize=False)
    else:
        raise NotImplementedError

def compute_cll_img_clfs(model, train_data_loader, loss_function='CrossEntropy'):
    # test base learner 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    if loss_function == 'BCELoss':
        pos_weight = torch.tensor([0.2])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_function == 'WeightCrossEntropy':
        train_data = train_data_loader.dataset
        labels = train_data.img_labels
        binary = torch.unique(labels).size(0) == 2
        if binary == True: 
            weights = cal_local_weight(labels).astype(np.float32)
            weights = torch.tensor(weights, device=device)
        else:
            weights = None
        criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction='sum')
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        
    model.eval()
    cll = 0.0
    with torch.no_grad():
        for x_batch, y_batch in train_data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            cll += -loss.item()
    # model.to(torch.device('cpu'))
    cll = 0.0 if cll is None else cll
    return cll

def predict_proba_base_clfs(model, x_test):
    """
    Predicts class probabilities based on the model type.
    Supports PyTorch models, scikit-learn classifiers, and lists.
    """
    
    if isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device 
            x_test = x_test.to(device)
            probs = softmax(model(x_test), dim=1)  # Ensure dim=1 for multi-class
            # print('model 1!')
            return probs.detach().cpu().numpy()

    elif isinstance(model, ClassifierMixin):  
        x = x_test.reshape(1, -1) if x_test.ndim == 1 else x_test  # Reshape if 1D
        # print('model 2!')
        return model.predict_proba(x)

    elif isinstance(model, list):
        # print('model 3!')   
        return np.array(model).reshape(1, -1)  # Ensure it returns a proper NumPy array
    else:
        raise NotImplementedError("Model type not supported")

def predict_log_proba_base_clfs(model, x_test):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    if isinstance(model, torch.nn.Module):
        model.eval()
        with torch.no_grad():
            probs = log_softmax(model(x_test), dim=1)
            return probs.detach().numpy()
    elif isinstance(model, ClassifierMixin):
        x = x_test.reshape(1, -1) if len(x_test.shape) == 1 else x_test
        return model.predict_log_proba(x)
    elif isinstance(model, list):
        return np.log(model)
    else:
        raise NotImplementedError

def save_results_csv(file_name='predictions_results.csv', column_names=None, *args):
    """
    Save multiple inference results to a CSV file with flexible input.

    Parameters:
    - file_name: Name of the output CSV file (default: 'predictions_results.csv')
    - column_names: List of column names for the CSV file (default: None).
                    If not provided, columns will be named 'Column_1', 'Column_2', etc.
    - *args: Variable length argument list of prediction arrays or lists.

    Returns:
    - None, saves the DataFrame to a CSV file.
    """
    # Ensure all inputs are 1-dimensional arrays/lists
    flattened_args = [np.array(arg).flatten() for arg in args]

    # Check if all arrays have the same length
    lengths = [len(arr) for arr in flattened_args]
    if len(set(lengths)) != 1:
        print("Error: Prediction arrays are not of the same length.")
        return

    # If column_names is None, create default column names
    if column_names is None:
        column_names = [f"Column_{i+1}" for i in range(len(flattened_args))]
    elif len(column_names) != len(flattened_args):
        print("Error: Number of column names must match the number of input arrays.")
        return

    # Create a dictionary for the DataFrame
    results_dict = {col: flattened_args[i] for i, col in enumerate(column_names)}

    # Create a DataFrame
    results_df = pd.DataFrame(results_dict)

    # Save DataFrame to CSV
    results_df.to_csv(file_name, index=False)
    print(f"Results saved to {file_name}")

def write_bn(bn, path):

    """
    Wrapper function for writing a BayesNet
    object to file

    Arguments
    ---------
    *bn* : a BayesNet object

    *path* : a string
        The path, absolute or relative. MUST contain
        the extension: '.bn' only support right now

    - Creates a new file on the user's local system

    Notes
    -----
    - Should add support for '.bif' and others

    """
    if '.bn' in path:
        write_json(bn, path)
    else:
        print("File Extension not supported")

def write_json(bn, path):
    """
    Write a BayesNet object to a json format file

    Arguments:
        1. *filename* - the path/name of the file to which the function will write the BKB object.
    
    """
    bn_dict = OrderedDict([('V',bn.V),('E',bn.E),('F',bn.F)])
    with open(path, 'w') as outfile:
        json.dump(bn_dict, outfile,indent=2)

def read_local_scores(f,verbose=False):
    '''Read local scores from a named file, standard input or a file object,
    and return a dictionary dkt where dkt[child][parentset] is the local score 
    for the family child<-parentset

    The file is assumed to be in "Jaakkola" format.

    Args:
        f (str/file object) : The file containing the local scores. 

    Returns: 
     dict : Dictionary containing local scores
    '''

    if type(f) == str:
        if f == '-':
            f = sys.stdin
        else:
            f = open(f)

    family_scores = {}
    n = int(f.readline())
    if verbose:
        print('Problem has {0} variables'.format(n), file=sys.stderr)
    fields = f.readline().rstrip().split()
    def init(fields):
        # no check that there are only two fields
        return fields[0], int(fields[1]), 0, {} 
    current_variable, nscores, i, this_dkt = init(fields)
    for line in f:
        fields = line.rstrip().split()
        if i < nscores:
            # don't bother checking that fields[1] correctly specifies
            # the number of parents
            this_dkt[frozenset(fields[2:])] = float(fields[0])
            i += 1
        else:
            family_scores[current_variable] = this_dkt
            current_variable, nscores, i, this_dkt = init(fields)
    f.close()
    family_scores[current_variable] = this_dkt
    if verbose:
        print('Scores read in', file=sys.stderr)
    return family_scores

def cal_global_weight(data):
    n_class = data.shape[1]
    no_samples = sum(np.bincount(data[:, i], minlength=2)[1] for i in range(n_class))
    weight_class = np.empty(shape=n_class)
    # Set up weight for demographic Information
    weight_demo = np.array([1, 1])
    for i in range(n_class):
        data_class = data[:, i]
        weight_class[i] = no_samples / (n_class * np.bincount(data_class)[1])
    weight_class = np.append(weight_class, weight_demo)
    return weight_class

def cal_local_weight(data):
    # Binary value
    n_class = int(2) 
    no_samples = data.shape
    weight_class = np.empty(shape=n_class)
    weight_class = no_samples / (n_class * np.bincount(data))
    return weight_class

def create_evidence(label_nodes, label_domains, evidence_nodes=None):
    """
    Create an evidence dictionary for multiple evidence nodes, with the real observed state.
    
    Parameters:
    - label_nodes: A list or dictionary containing observed values of evidence nodes.
    - label_domains: A dictionary where keys are node names and values are the valid domain for each node.
    - evidence_nodes: A list of node names for which evidence should be created.

    Returns:
    - evidences: A dictionary where keys are node names, and values are the corresponding observed states, or None if invalid.
    """
    evidences = {}
    
    if evidence_nodes is not None:
        for node in evidence_nodes:
            value = label_nodes[int(node)].item()
            # value = label_nodes[i].item()
            # Check if the value is np.nan and assign None if true
            if isinstance(value, float) and np.isnan(value):
                evidences[node] = None
            elif value in label_domains[node]:
                # Valid value within the domain for the node
                evidences[node] = value
            else:
                # Invalid value or node not in domain
                evidences[node] = None
    else:
        evidences = None
    
    return evidences

def check_missing_node(dict_data):
    # dict_data : dict type indicate the ground true values (values) of nodes (keys)
    # Return a list of keys with None as their value, or None if no such keys exist
    result = [key for key, value in dict_data.items() if value is None]
    return result if result else None

def missing_data_per_column(data, missing_fraction=0.3):
    """
    Introduces missing data where each column has 30% missing values.
    """
    data_with_missing = data.astype(float)
    rows, cols = data.shape

    for col in range(cols):  # Process each column separately
        num_missing = int(rows * missing_fraction)  # 30% of rows per column
        missing_indices = np.random.choice(rows, num_missing, replace=False)
        data_with_missing[missing_indices, col] = np.nan  # Set NaN in the column

    return data_with_missing

def missing_data_total(data, missing_fraction=0.3):
    """
    Introduces missing data in missing_fraction% of all elements (not per column).
    """
    data_with_missing = data.astype(float)
    total_elements = data.size  # Total number of elements in the array
    num_missing = int(total_elements * missing_fraction)  # Number of missing values

    # Flatten data, randomly choose indices to replace with NaN
    flat_data = data_with_missing.flatten()
    missing_indices = np.random.choice(total_elements, num_missing, replace=False)
    flat_data[missing_indices] = np.nan

    return flat_data.reshape(data.shape)  # Reshape back to original

def missing_data_generalized(data, missing_fraction=0.3):
    """
    Introduces random missing data into a NumPy array.
    
    Parameters:
    - data: 2D NumPy array of demographic entries.
    - missing_fraction: Fraction of values to replace with NaN.
    
    Returns:
    - Modified NumPy array with random NaN values introduced.
    """
    # Create a copy of the data to avoid modifying the original array
    data_with_missing = data.astype(float)  # Convert to float to support NaN
    
    # Generate a mask of random True/False values
    mask = np.random.rand(*data.shape) < missing_fraction

    # Replace selected elements with NaN
    data_with_missing[mask] = np.nan
    # data_with_missing[mask] = torch.nan
    
    return data_with_missing

def fix_proba_output(probs, clf_classes , expected_classes):
    """Ensures predict_proba output matches expected classes.
        - expected_classes : True set of classes 
        - probs: shape - (n_samples, num_learned_classes)
        - clf.classes_ storing only the classes seen during training 
    """
    full_probs = np.zeros((probs.shape[0], len(expected_classes)))  # Create full-size matrix
    for idx, c in enumerate(clf_classes):
        full_probs[:, expected_classes.tolist().index(c)] = probs[:, idx]  # Map correct probabilities
    return full_probs

def extract_missing_positions(test_data_loader, is_dis_predict=False):
    """
    Detect missing positions in each instance and return a NumPy array.
    
    Args:
        all_labels: True value labels (no missing values).
        labels: Missing value labels (contains NaN values).
        'labels' and 'all_labels' as NumPy arrays.
        is_dis_predict: To evaluate discreate features or not (default: False).
    
    Returns:
        A NumPy array of shape (N, 2), where each row contains:
            [instance_index, missing_var_index]
    """
    # Convert inputs to NumPy arrays if they aren't already
    if is_dis_predict == False:
        no_dis = int(len(test_data_loader.discrete_feature_names))
        all_labels = np.asarray(test_data_loader.all_labels[:, :-no_dis])
        labels = np.asarray(test_data_loader.labels[:, :-no_dis])
    else:
        all_labels = np.asarray(test_data_loader.all_labels)
        labels = np.asarray(test_data_loader.labels)

    # Ensure input shapes match
    if all_labels.shape != labels.shape:
        raise ValueError("Shape mismatch between all_labels and labels!")

    # Find all missing positions
    instance_indices, variable_indices = np.where(all_labels != labels)

    # Stack the indices into a 2D NumPy array of shape (N, 2)
    missing_data_array = np.column_stack((instance_indices, variable_indices))

    return missing_data_array

def extract_missing_values_per_instance(test_data_loader, *y_pred):
    """
    Extracts missing values per instance and stores them in NumPy arrays.

    Args:
        test_data_loader includes 'labels' and 'all_labels' as NumPy arrays.
        *y_pred: Variable number of NumPy arrays of shape (num_instances, num_classes).

    Returns:
        A tuple containing:
        - y_true_miss: NumPy array of missing true values.
        - List of NumPy arrays for each y_pred input (missing values only).
    """
    # Extract missing positions
    missing_indices = extract_missing_positions(test_data_loader)
    num_instances = test_data_loader.labels.shape[0]
    y_true = test_data_loader.all_labels

    # Initialize lists to store missing values per instance
    y_true_miss = [[] for _ in range(num_instances)]
    y_pred_miss = [[[] for _ in range(len(y_pred))] for _ in range(num_instances)]  # Nested list for multiple predictions

    # Populate missing value lists
    for instance_idx, var_idx in missing_indices:
        y_true_miss[instance_idx].append(int(y_true[instance_idx, var_idx]))
        for pred_idx, y_p in enumerate(y_pred):  # Iterate over each prediction array
            y_pred_miss[instance_idx][pred_idx].append(y_p[instance_idx, var_idx])

    # Convert lists to NumPy arrays and filter out empty lists
    y_true_miss = np.array([np.array(arr) for arr in y_true_miss if arr], dtype=object)
    y_pred_miss = [np.array([np.array(arr) for arr in preds if arr], dtype=object) for preds in zip(*y_pred_miss)]

    return (y_true_miss, *y_pred_miss)

def merge_random_triple(disc_domains, disc_features, disc_names):
    """
    Merge random triples of discrete features into new combined nodes.
    Specifed for Thyroid dataset containing 22 discrete features ('7': 6 outcomes, others are binary)
    => Merge three random nodes into a new node except node 7
    Parameters:
        disc_domains (dict): Dictionary mapping feature names to their domains.
        disc_features (np.array): Feature matrix where columns correspond to `disc_names`.
        disc_names (list): List of feature names.

    Returns:
        disc_domains (dict): Updated dictionary with merged feature domains.
        merged_features (np.array): Updated feature matrix with merged nodes.
        disc_names (list): Updated list of remaining feature names.
    """
    # disc_features = disc_features[:100, :]
    ori_features = disc_features.copy()
    ori_domains = disc_domains.copy()
    disc_frame = {}
    if '7' in disc_names:
        disc_names.remove('7')  
    # Perform merging only on the original nodes
    while len(disc_names) >= 3:
        random.shuffle(disc_names)
        node1, node2, node3 = disc_names.pop(0), disc_names.pop(0), disc_names.pop(0)

        # Create merged node name
        new_node = f"{node1}_{node2}_{node3}"

        # Merge domains
        domain1 = np.array(ori_domains[node1], dtype=int)
        domain2 = np.array(ori_domains[node2], dtype=int)
        domain3 = np.array(ori_domains[node3], dtype=int)
        new_domain = [tuple(map(int, x)) for x in itertools.product(domain1, domain2, domain3)]

        # Merge features
        idx1 = list(ori_domains.keys()).index(node1)
        idx2 = list(ori_domains.keys()).index(node2)
        idx3 = list(ori_domains.keys()).index(node3)
        feat1 = np.array(ori_features[:, idx1], dtype=int)
        feat2 = np.array(ori_features[:, idx2], dtype=int)
        feat3 = np.array(ori_features[:, idx3], dtype=int)
        feat = list(zip(feat1, feat2, feat3))

        # Encode features
        label_mapping = {label: idx for idx, label in enumerate(new_domain)}
        encoded_data = np.array([label_mapping.get(f, -1) for f in feat]) 

        # Update domains
        disc_domains[new_node] = np.array(list(label_mapping.values()), dtype=int)
        # disc_domains[new_node] = new_domain
        del disc_domains[node1], disc_domains[node2], disc_domains[node3]

        # Store merged features
        disc_frame[new_node] = encoded_data

    # Handle remaining node : node '7'
    last_node = '7'
    disc_domains[last_node] = np.array(disc_domains[last_node], dtype=int)
    # np.array(ori_features[:, last_node], dtype=int)
    last_idx = list(ori_domains.keys()).index(last_node)
    disc_frame[last_node] = np.array(ori_features[:, last_idx], dtype=int)
    # disc_frame[last_node] = ori_features[:, last_idx]

    # Convert feature dictionary to correctly ordered array
    disc_names = sorted(disc_domains.keys())  # Keep a consistent order
    merged_features = np.column_stack([disc_frame[name] for name in disc_names])

    return disc_domains, merged_features, disc_names


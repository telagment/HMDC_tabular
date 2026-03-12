import os
import sys

sys.path.append('..')

import csv
import pickle
# import datetime
import numpy as np
import argparse
import logging
import time
import pandas as pd

from sklearn.model_selection import KFold
from data import TabularDataLoader_HMDC
from classifier import TabularClassifier_HMDC
from utils import merge_random_triple

np.random.seed(42)

DATASETS = [
    'Adult',
    'Default',
    'Thyroid',
]

BASELEARNERS = [
    'lr',
    'nb',
    'rf',
    'svm',
]

DATASETS_DIR = 'MDC_data'

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Training HMDC models on tabular datasets')
    parser.add_argument('-dataset', choices=DATASETS, default='Adult', type=str, help='Dataset')
    parser.add_argument('--baselearner', choices=BASELEARNERS, default='lr', type=str, help='Base Learner')
    parser.add_argument('--palim', type=int, default=2, help='The maximum number of parents for each node')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--pruning-score', type=str, choices=['BIC', 'AIC'], default='BIC', help='Pruning score for learning')
    parser.add_argument('--grouping', action=argparse.BooleanOptionalAction, default=False, help='Whether or not to group discrete features')
    parser.add_argument('--output', type=str, default='experiments_tabular/HMDC_training', help='Output path')
    args = parser.parse_args()

    dataset = args.dataset
    base = args.baselearner
    palim = args.palim
    n_folds = args.n_folds
    output_path = args.output
    pruning_score = args.pruning_score
    grouping = args.grouping
    model_dir = os.path.join(output_path, pruning_score, base, dataset)
    os.makedirs(model_dir, exist_ok=True)

    if not(model_dir is None):
        name = base + '_' + dataset 
        log_file = os.path.join(model_dir, name + '.log')
        logging.basicConfig(level=logging.INFO, filename=log_file, filemode="w")
    else:
        raise Exception('Can not find Result path!')
    
    logging.info('{}{}{}{}'.format('Base learner: ', base, 'Dataset:', dataset))
    logging.info('{}{}{}{}'.format('No folds : ', n_folds, 'Palim : ', palim))
    logging.info('{}{}{}{}'.format('Pruning score : ', pruning_score, 'Grouping : ', grouping))

    mat_path = os.path.join(DATASETS_DIR, dataset, dataset) + '.mat'
    data_loader = TabularDataLoader_HMDC(mat_path=mat_path, dataset=dataset)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Merge random pairs of discrete features to reduce the number of discrete features
    if dataset == 'Thyroid': 
        disc_domains, disc_features, disc_names = merge_random_triple(data_loader.discrete_feature_domains, 
                                                                    data_loader.discrete_features, 
                                                                    data_loader.discrete_feature_names) 
        logging.info(f"Thyroid discrete features group {(disc_names)}")
        # saving disc-names
        with open(os.path.join(model_dir, "disc_names.txt"), 'w') as f:
            for item in disc_names:
                f.write("%s\n" % item)

        # 2. Map the new node names to the index-based names (using their index in disc_names)
        no_y = len(data_loader.label_names)
        indexed_disc_names = [str(i+no_y) for i in range(len(disc_names))]

        # 3. Create new disc_domains with indexed names, maintaining the correct order
        ordered_disc_domains = {indexed_disc_names[i]: disc_domains[node] for i, node in enumerate(disc_names)}
   
        # Update data_loader with correctly ordered data
        data_loader.discrete_feature_domains = ordered_disc_domains
        data_loader.discrete_features = disc_features
        data_loader.discrete_feature_names = indexed_disc_names
        with open(os.path.join(model_dir, "data_loader.pkl"), 'wb') as f:
                    pickle.dump(data_loader, f)
    if dataset == 'Adult' and grouping == True: 
        data_loader.update(data_loader, node_id=3) 

    saving_dir = os.path.join(model_dir,'data')
    os.makedirs(saving_dir, exist_ok=True)

    for i, (train_indices, test_indices) in enumerate(kf.split(data_loader.labels)):
        saving_fold = os.path.join(saving_dir, str(i))
        os.makedirs(saving_fold, exist_ok=True)

        data_splits = os.path.join(saving_fold, "dataset.npz")
        np.savez(data_splits, train_indices=train_indices, test_indices=test_indices)

        train_data_loader = data_loader.create_sub_data_loader(train_indices)
            
        # GBNC
        time_time = time.time()
        hmdc = TabularClassifier_HMDC(palim=palim, base=base, result_path=saving_fold)
        hmdc.fit(train_data_loader, pruning_score=pruning_score, save_baselearner=True)

        hmdc.learn_structure(pruning_score=pruning_score)
        logging.info(f"Time to fit and learn structure: {time.time() - time_time}") 
        time_time = time.time()



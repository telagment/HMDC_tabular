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
    parser.add_argument('-dataset', choices=DATASETS, default='Thyroid', type=str, help='Dataset')
    parser.add_argument('--baselearner', choices=BASELEARNERS, default='lr', type=str, help='Base Learner')
    parser.add_argument('--palim', type=int, default=0, help='The maximum number of parents for each node')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, help='Whether or not to plot the BN structure')
    parser.add_argument('--is-missing', action=argparse.BooleanOptionalAction, default=True, help='Whether or not missing data appear in inference')
    parser.add_argument('--n-folds', type=int, default=10, help='Number of folds')
    parser.add_argument('--output', type=str, default='experiments_tabular/HMDC_training', help='Output path')
    args = parser.parse_args()

    dataset = args.dataset
    base = args.baselearner
    palim = args.palim
    disclim = args.disclim
    is_plot = args.plot
    n_folds = args.n_folds
    output_path = args.output
    base_dir = os.path.join(output_path, base)
    os.makedirs(base_dir, exist_ok=True)
    dataset_dir = os.path.join(base_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)


    model_dir = dataset_dir
    os.makedirs(model_dir, exist_ok=True)

    if not(model_dir is None):
        name = base + '_' + dataset 
        log_file = os.path.join(model_dir, name + '.log')
        logging.basicConfig(level=logging.INFO, filename=log_file, filemode="w")
    else:
        raise Exception('Can not find Result path!')
    
    logging.info('{}{}{}{}{}{}{}{}'.format('Base learner: ', base, 'Dataset:', dataset))
    logging.info('{}{}{}{}'.format('No folds : ', n_folds, 'Palim : ', palim))

    mat_path = os.path.join(DATASETS_DIR, dataset, dataset) + '.mat'
    data_loader = TabularDataLoader_HMDC(mat_path=mat_path, dataset=dataset)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    h_hls_opt, h_zos_opt, s_hls_opt, s_zos_opt = [], [], [], []
    h_hls_ave, h_zos_ave, s_hls_ave, s_zos_ave = [], [], [], []
    h_hls_ref, h_zos_ref, s_hls_ref, s_zos_ref = [], [], [], []
    imp_h_hls, imp_s_hls, imp_h_zos, imp_s_zos = [], [], [], []
    imp_h_hls_02, imp_s_hls_02, imp_h_zos_02, imp_s_zos_02 = [], [], [], []
    ran_zeros, ran_hams = [], []
    imp_zeros, imp_hams = [], []
    br_hls, br_zos = [], []
    ran_deltas, imp_deltas, br_deltas = [], [], []
    opt_deltas, ave_deltas, ref_deltas = [], [], []
    train_splits = []
    test_splits = []
    
    # Merge random pairs of discrete features to reduce the number of discrete features
    if dataset == 'Thyroid': 
        disc_domains, disc_features, disc_names = merge_random_triple(data_loader.discrete_feature_domains, 
                                                                    data_loader.discrete_features, 
                                                                    data_loader.discrete_feature_names) 
            
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
        gbnc = TabularClassifier_HMDC(palim=palim, base=base, result_path=saving_fold)
        gbnc.fit(train_data_loader, save_baselearner=True)

        gbnc.learn_structure()
        logging.info(f"Time to fit and learn structure: {time.time() - time_time}") 
        time_time = time.time()



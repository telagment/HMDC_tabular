'''    
    This script is used to evaluate the performance of the hmdc model with missing data in the inference step.
    The struture is follow the constrant that X_c --> (X_d, Y)    
    Different from the previous script, this script will compare the performance of the hmdc model with the 
    random classifier, mode imputation, and hmdc imputation.
    - FIle: TO run save training results for making difference evaluations between different methods
'''

import os
import sys

sys.path.append('..')

import csv
import pickle
import datetime
import numpy as np
import argparse
import logging
import time
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.dummy import DummyClassifier
from scipy import stats
from skmultilearn.problem_transform import BinaryRelevance

from data import TabularDataLoader_HMDC
from classifier import TabularClassifier_HMDC
from metrics import zero_one_score_mdc, hamming_score_mdc_variate, deltas_distance
from utils import extract_missing_values_per_instance, merge_random_triple

np.random.seed(42)

DATASETS = [
    'Adult',
    'Default',
    'Thyroid',
]

BASELEARNERS = [
    'lr',
    'lr_valid',
    'nb',
    'rf',
    'svm',
]

DATASETS_DIR = 'MDC_data'

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Mixed Data Experiments with Logistic Regression')
    parser.add_argument('-dataset', choices=DATASETS, default='Adult', type=str, help='Dataset')
    parser.add_argument('--baselearner', choices=BASELEARNERS, default='lr', type=str, help='Base Learner')
    parser.add_argument('--palim', type=int, default=2, help='The maximum number of parents for each node')
    # parser.add_argument('--dis-missing', type=float, default=0.5, help='The percentage of missing discrete features ')
    # parser.add_argument('--class-missing', type=float, default=0.5, help='The percentage of missing class variables')
    parser.add_argument('--n-chains', default=10, type=int, help='Number of randomly generated classifier chains')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, help='Whether or not to plot the BN structure')
    parser.add_argument('--is-missing', action=argparse.BooleanOptionalAction, default=True, help='Whether or not missing data appear in inference')
    parser.add_argument('--n-folds', type=int, default=10, help='Number of folds')
    parser.add_argument('--output', type=str, default='experiments_tabular/HMDC_prediction', help='Output path')
    args = parser.parse_args()

    dataset = args.dataset
    base = args.baselearner
    palim = args.palim
    disclim = args.disclim
    is_plot = args.plot
    n_chains = args.n_chains
    n_folds = args.n_folds
    output_path = args.output
    is_missing = args.is_missing

    dis_missing_list = [0.3, 0.8]
    class_missing_list = [0.3, 0.7, 0.8, 0.9]

    base_dir = os.path.join(output_path, base)
    os.makedirs(base_dir, exist_ok=True)
    dataset_dir = os.path.join(base_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    load_dir = 'experiments_tabular/HMDC_inference/' + str(base) + '/' + str(dataset)

    for dis_missing in dis_missing_list:
        for class_missing in class_missing_list:
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
            imp_hmdc_deltas_h, imp_hmdc_deltas_s = [], []
            opt_deltas_h, opt_deltas_s = [], []
            ave_deltas_h, ave_deltas_s = [], []
            ref_deltas_h, ref_deltas_s = [], []
            train_splits = []
            test_splits = []
            print(f"Processing dis_missing: {dis_missing}, class_missing: {class_missing}")
            name = str(class_missing) + '_' + str(dis_missing)
            load_case_dir = os.path.join(load_dir, name)
            model_dir = os.path.join(dataset_dir, name)
            os.makedirs(model_dir, exist_ok=True)
            
            log_file = os.path.join(model_dir, 'log_file.log')
            logging.basicConfig(level=logging.INFO, filename=log_file, filemode="w")

            logging.info(f'python mixed_data_missing_inference_HMDC_Imb_Inference_loop.py -dataset {dataset} --baselearner {base} \
                    --palim {palim} --disclim {disclim} --dis-missing {dis_missing} --class-missing {class_missing} \
                    --n-chains {n_chains} --n-folds {n_folds} --output {output_path} --is-missing {is_missing}')
        
            test_dir = os.path.join(load_case_dir, 'test_data')

            for i in range(n_folds):
                prediction_dir = os.path.join(test_dir, str(i) + '/predictions.npz')
                
                data = np.load(prediction_dir, allow_pickle=True)

                # To load it back
                with open(os.path.join(test_dir, str(i) + "/test_data_loader.pkl"), 'rb') as f:
                    test_data_loader = pickle.load(f)

                # Create missing indicator: 1 if missing (-1), 0 otherwise
                missing_indicator = (test_data_loader.labels == -1).astype(int)

                y_pred_ran = data['y_pred_ran']
                y_pred_imp = data['y_pred_imp']
                br_pred = data['br_pred']
                hmdc_h_imp = data['hmdc_h_imp']
                hmdc_s_imp = data['hmdc_s_imp']
                hmdc_h_pred_opt = data['hmdc_h_pred_opt']
                hmdc_s_pred_opt = data['hmdc_s_pred_opt']
                hmdc_h_pred_ave = data['hmdc_h_pred_ave']
                hmdc_s_pred_ave = data['hmdc_s_pred_ave'] 
                hmdc_s_pred_ref = data['hmdc_s_pred_ref']
                hmdc_h_pred_ref = data['hmdc_h_pred_ref']

                y_true = test_data_loader.all_labels
                
                (   y_true,
                    y_pred_ran,
                    y_pred_imp,
                    br_pred,
                    hmdc_h_imp,
                    hmdc_s_imp,
                    hmdc_h_pred_opt,
                    hmdc_s_pred_opt,
                    hmdc_h_pred_ave,
                    hmdc_s_pred_ave,
                    hmdc_s_pred_ref,
                    hmdc_h_pred_ref,
                ) = extract_missing_values_per_instance(
                    test_data_loader,
                    y_pred_ran,
                    y_pred_imp,
                    br_pred,
                    hmdc_h_imp,
                    hmdc_s_imp,
                    hmdc_h_pred_opt,
                    hmdc_s_pred_opt,
                    hmdc_h_pred_ave,
                    hmdc_s_pred_ave,
                    hmdc_s_pred_ref,
                    hmdc_h_pred_ref,
                )

                # Random Classifier
                ran_zero = zero_one_score_mdc(y_true=y_true, y_pred=y_pred_ran)
                ran_ham = hamming_score_mdc_variate(y_true=y_true, y_pred=y_pred_ran)
                
                # Mode Imputation
                imp_zero = zero_one_score_mdc(y_true=y_true, y_pred=y_pred_imp)
                imp_ham = hamming_score_mdc_variate(y_true=y_true, y_pred=y_pred_imp)
                
                # Binary Relevance
                br_hl = hamming_score_mdc_variate(y_true=y_true, y_pred=br_pred)
                br_zo = zero_one_score_mdc(y_true=y_true, y_pred=br_pred)
                
                imp_h_hl = hamming_score_mdc_variate(y_true=y_true, y_pred=hmdc_h_imp)
                imp_s_zo = zero_one_score_mdc(y_true=y_true, y_pred=hmdc_s_imp)

                h_hl_opt = hamming_score_mdc_variate(y_true=y_true, y_pred=hmdc_h_pred_opt)
                s_zo_opt = zero_one_score_mdc(y_true=y_true, y_pred=hmdc_s_pred_opt)

                h_hl_ave = hamming_score_mdc_variate(y_true=y_true, y_pred=hmdc_h_pred_ave)
                s_zo_ave = zero_one_score_mdc(y_true=y_true, y_pred=hmdc_s_pred_ave)

                h_hl_ref = hamming_score_mdc_variate(y_true=y_true, y_pred=hmdc_h_pred_ref)
                s_zo_ref = zero_one_score_mdc(y_true=y_true, y_pred=hmdc_s_pred_ref)

                ran_zeros.append(ran_zero)
                ran_hams.append(ran_ham)

                imp_zeros.append(imp_zero)
                imp_hams.append(imp_ham)

                br_hls.append(br_hl)
                br_zos.append(br_zo)

                imp_h_hls.append(imp_h_hl)
                imp_s_zos.append(imp_s_zo)
        
                h_hls_opt.append(h_hl_opt)
                s_zos_opt.append(s_zo_opt)


                h_hls_ave.append(h_hl_ave)
                s_zos_ave.append(s_zo_ave)


                h_hls_ref.append(h_hl_ref)
                s_zos_ref.append(s_zo_ref)

            # Prepare data for CSV
            data = [
                ["Random Classifier", "Hamming", np.mean(ran_hams) * 100, np.std(ran_hams) * 100],
                ["Random Classifier", "0/1 Score", np.mean(ran_zeros) * 100, np.std(ran_zeros) * 100],

                ["Mode Imputation", "Hamming", np.mean(imp_hams) * 100, np.std(imp_hams) * 100],
                ["Mode Imputation", "0/1 Score", np.mean(imp_zeros) * 100, np.std(imp_zeros) * 100],

                ["Binary Relevance", "Hamming", np.mean(br_hls) * 100, np.std(br_hls) * 100],
                ["Binary Relevance", "0/1 Score", np.mean(br_zos) * 100, np.std(br_zos) * 100],

                ["hmdc Imputation", "Hamming", np.mean(imp_h_hls) * 100, np.std(imp_h_hls) * 100],
                ["hmdc Imputation", "0/1 Score", np.mean(imp_s_zos) * 100, np.std(imp_s_zos) * 100],

                ["hmdc Optimistic", "Hamming", np.mean(h_hls_opt) * 100, np.std(h_hls_opt) * 100],
                ["hmdc Optimistic", "0/1 Score", np.mean(s_zos_opt) * 100, np.std(s_zos_opt) * 100],

                ["hmdc Average", "Hamming", np.mean(h_hls_ave) * 100, np.std(h_hls_ave) * 100],
                ["hmdc Average", "0/1 Score", np.mean(s_zos_ave) * 100, np.std(s_zos_ave) * 100],

                ["hmdc Reference", "Hamming", np.mean(h_hls_ref) * 100, np.std(h_hls_ref) * 100],
                ["hmdc Reference", "0/1 Score", np.mean(s_zos_ref) * 100, np.std(s_zos_ref) * 100],
            ]

            # Convert to DataFrame
            df = pd.DataFrame(data, columns=["Category", "Metric", "Mean (%)", "Std Dev (%)"])

            # Split into separate DataFrames
            df_hamming = df[df["Metric"] == "Hamming"]
            df_subset = df[df["Metric"] == "0/1 Score"]

            # Save to separate CSV files
            hamming_csv = os.path.join(model_dir, "results_hamming.csv")
            subset_csv = os.path.join(model_dir, "results_subset.csv")

            df_hamming.to_csv(hamming_csv, index=False)
            df_subset.to_csv(subset_csv, index=False)
            
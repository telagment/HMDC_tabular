import os
import sys

sys.path.append('..')

import pickle
import numpy as np
import argparse
import logging
import time

from sklearn.dummy import DummyClassifier
from scipy import stats
from skmultilearn.problem_transform import BinaryRelevance

from data import TabularDataLoader_HMDC
from classifier import TabularClassifier_HMDC
# from metrics import zero_one_score_mdc, hamming_score_mdc_variate, deltas_distance
# from utils import extract_missing_values_per_instance, merge_random_triple

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

DATASETS_DIR = '/MDC_data'


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Mixed Data Experiments with Logistic Regression')
    parser.add_argument('-dataset', choices=DATASETS, default='Adult', type=str, help='Dataset')
    parser.add_argument('--baselearner', choices=BASELEARNERS, default='lr', type=str, help='Base Learner')
    parser.add_argument('--palim', type=int, default=2, help='The maximum number of parents for each node')
    # parser.add_argument('--dis-missing', type=float, default=0.8, help='The percentage of missing discrete features ')
    # parser.add_argument('--class-missing', type=float, default=0.8, help='The percentage of missing class variables')
    parser.add_argument('--is-missing', action=argparse.BooleanOptionalAction, default=True, help='Whether or not missing data appear in inference')
    parser.add_argument('--n-folds', type=int, default=10, help='Number of folds')
    parser.add_argument('--output', type=str, default='experiments_tabular/HMDC_inference_rest', help='Output path')
    args = parser.parse_args()

    dataset = args.dataset
    base = args.baselearner
    palim = args.palim
    disclim = args.disclim
    n_folds = args.n_folds
    output_path = args.output
    is_missing = args.is_missing

    dis_missing_list = [0.3, 0.8]
    class_missing_list = [0.3, 0.7, 0.8, 0.9]

    base_dir = os.path.join(output_path, base)
    os.makedirs(base_dir, exist_ok=True)
    dataset_dir = os.path.join(base_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    data_loader_dir = 'experiments_tabular/HMDC_training/' + str(base) + '/' + str(dataset)
    mat_path = os.path.join(DATASETS_DIR, dataset, dataset) + '.mat'
    if dataset == 'Thyroid': 
        with open(os.path.join(data_loader_dir, "data_loader.pkl"), 'rb') as f:
            data_loader = pickle.load(f)
    else:
        data_loader = TabularDataLoader_HMDC(mat_path=mat_path, dataset=dataset)

    load_dir = 'experiments_tabular/HMDC_training/' + str(base) + '/' + str(dataset) + '/data'
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
            train_splits = []
            test_splits = []
            model_dir = os.path.join(dataset_dir, str(class_missing) + '_' + str(dis_missing))
            os.makedirs(model_dir, exist_ok=True)

            saving_dir = os.path.join(model_dir, 'test_data')
            os.makedirs(saving_dir, exist_ok=True)
            
            log_file = os.path.join(model_dir, 'log_file.log')
            logging.basicConfig(level=logging.INFO, filename=log_file, filemode="w")

            logging.info(f'python mixed_data_missing_inference_HMDC_Imb_Inference_loop.py -dataset {dataset} --baselearner {base} \
                    --palim {palim} --disclim {disclim} --dis-missing {dis_missing} --class-missing {class_missing} \
                    --n-folds {n_folds} --output {output_path} --is-missing {is_missing}')
            
            for i in range(n_folds):
                saving_fold = os.path.join(saving_dir, str(i))
                os.makedirs(saving_fold, exist_ok=True)
                load_fold = os.path.join(load_dir, str(i))
                data_splits = np.load(os.path.join(load_fold, "dataset.npz"))
                train_indices = data_splits["train_indices"]
                test_indices = data_splits["test_indices"]

                train_data_loader = data_loader.create_sub_data_loader(train_indices)
                test_data_loader = data_loader.create_sub_data_loader(test_indices, missing_class_var=class_missing, missing_dis_fea=dis_missing, is_missing=is_missing)

                with open(os.path.join(saving_fold, "test_data_loader.pkl"), 'wb') as f:
                    pickle.dump(test_data_loader, f)

                # Random prediction - For testing only
                # Create a random classifier (uniform guessing)
                random_clf = DummyClassifier(strategy="uniform", random_state=42)
                random_clf.fit(test_data_loader.continuous_features, test_data_loader.all_labels)
                y_pred_ran = random_clf.predict(test_data_loader.continuous_features)

                # # Imputation - Mode Imputeation
                most_frequent_classes = [stats.mode(train_data_loader.labels[:, col])[0][0] for col in range(train_data_loader.labels.shape[1])]
                y_pred_imp = test_data_loader.labels.copy()
                for col in range(test_data_loader.labels.shape[1]):
                    y_pred_imp[:, col] = np.where(test_data_loader.labels[:, col] == -1, most_frequent_classes[col], test_data_loader.labels[:, col])

            
                # Binary Relevance - BR
                base_clf = TabularClassifier_HMDC._create_base_clfs(base)
                br = BinaryRelevance(classifier=base_clf)
                br.fit(X=train_data_loader.continuous_features, y=train_data_loader.labels)
                br_pred = br.predict(X=test_data_loader.continuous_features)
                br_pred = br_pred.toarray()

                # GBNC
                time_time = time.time()
                gbnc = TabularClassifier_HMDC(palim=palim, base=base, result_path=model_dir)
                score_path = os.path.join(load_fold, "score")
                classifier_path = os.path.join(load_fold, "classifiers.pkl")
                # Load the learned structure and parameters
                gbnc.load_DAG(score_path, classifier_path, test_data_loader)

                gbnc.learn_structure()

                gbnc_s_imp, gbnc_h_imp = gbnc.inference_missing_imp(train_data_loader=train_data_loader, test_data_loader=test_data_loader)

                gbnc_s_pred_opt, gbnc_h_pred_opt = gbnc.inference_missing_opt(test_data_loader) 

                gbnc_s_pred_ave, gbnc_h_pred_ave = gbnc.inference_missing_ave(test_data_loader)

                gbnc_s_pred_ref, gbnc_h_pred_ref = gbnc.inference_missing_ref(test_data_loader=test_data_loader)

                np.savez(os.path.join(saving_fold, "predictions.npz"),
                            y_pred_ran=y_pred_ran,
                            y_pred_imp=y_pred_imp,
                            br_pred=br_pred,
                            gbnc_h_imp=gbnc_h_imp,
                            gbnc_s_imp=gbnc_s_imp,
                            gbnc_h_pred_opt=gbnc_h_pred_opt,
                            gbnc_s_pred_opt=gbnc_s_pred_opt,
                            gbnc_h_pred_ave=gbnc_h_pred_ave,
                            gbnc_s_pred_ave=gbnc_s_pred_ave,
                            gbnc_s_pred_ref=gbnc_s_pred_ref,
                            gbnc_h_pred_ref=gbnc_h_pred_ref,
                        )
    

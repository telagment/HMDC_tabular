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

DATASETS_DIR = 'MDC_data'


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Mixed Data Experiments with Logistic Regression')
    parser.add_argument('-dataset', choices=DATASETS, default='Adult', type=str, help='Dataset')
    parser.add_argument('--baselearner', choices=BASELEARNERS, default='lr', type=str, help='Base Learner')
    parser.add_argument('--palim', type=int, default=2, help='The maximum number of parents for each node')
    # parser.add_argument('--dis-missing', type=float, default=0.8, help='The percentage of missing discrete features ')
    # parser.add_argument('--class-missing', type=float, default=0.8, help='The percentage of missing class variables')
    parser.add_argument('--is-missing', action=argparse.BooleanOptionalAction, default=True, help='Whether or not missing data appear in inference')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--pruning-score', type=str, choices=['BIC', 'AIC'], default='BIC', help='Pruning score for learning')
    parser.add_argument('--grouping', action=argparse.BooleanOptionalAction, default=False, help='Whether or not to group Adult discrete features')
    parser.add_argument('--load_dir', type=str, default='experiments_tabular/HMDC_training/BIC/lr/Adult/data', help='Output path')
    parser.add_argument('--output', type=str, default='experiments_tabular/HMDC_inference', help='Output path')
    args = parser.parse_args()

    dataset = args.dataset
    base = args.baselearner
    palim = args.palim
    n_folds = args.n_folds
    output_path = args.output
    is_missing = args.is_missing
    load_dir = args.load_dir 
    grouping = args.grouping
    pruning_score = args.pruning_score
    
    dis_missing_list = [0.3, 0.8, 1.0]
    class_missing_list = [0.3, 0.7, 0.8, 0.9, 1.0]

    dataset_dir = os.path.join(output_path, pruning_score, base, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    # data_loader_dir = 'experiments_tabular/HMDC_training/' + str(base) + '/' + str(dataset)
    mat_path = os.path.join(DATASETS_DIR, dataset, dataset) + '.mat'
    if dataset == 'Thyroid': 
        with open(os.path.join(load_dir, "data_loader.pkl"), 'rb') as f:
            data_loader = pickle.load(f)
    else:
        data_loader = TabularDataLoader_HMDC(mat_path=mat_path, dataset=dataset)
    
    if dataset == 'Adult' and grouping == True: 
        data_loader.update(data_loader, node_id=3) 

    # load_dir = 'experiments_tabular/HMDC_training/' + str(base) + '/' + str(dataset) + '/data'
    for dis_missing in dis_missing_list:
        for class_missing in class_missing_list:
            model_dir = os.path.join(dataset_dir, str(class_missing) + '_' + str(dis_missing))
            os.makedirs(model_dir, exist_ok=True)
            saving_dir = os.path.join(model_dir, 'test_data')
            os.makedirs(saving_dir, exist_ok=True)
            log_file = os.path.join(model_dir, 'log_file.log')

            logging.basicConfig(level=logging.INFO, filename=log_file, filemode="w")
            logging.info(f'python HMDC_Inference.py -dataset {dataset} --baselearner {base} \
                    --palim {palim} --dis-missing {dis_missing} --class-missing {class_missing} \
                    --n-folds {n_folds} --output {output_path} --is-missing {is_missing} --pruning_score {pruning_score}')
            
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

                # # Imputation - Mode Imputation
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

                # hmdc
                time_time = time.time()
                hmdc = TabularClassifier_HMDC(palim=palim, base=base, result_path=model_dir)
                score_path = os.path.join(load_fold, "score")
                classifier_path = os.path.join(load_fold, "classifiers.pkl")
                # Load the learned structure and parameters
                hmdc.load_DAG(score_path, classifier_path, test_data_loader)

                hmdc.learn_structure(pruning_score=pruning_score)

                hmdc_s_imp, hmdc_h_imp = hmdc.inference_missing_imp(train_data_loader=train_data_loader, test_data_loader=test_data_loader)

                hmdc_s_pred_opt, hmdc_h_pred_opt = hmdc.inference_missing_opt(test_data_loader) 

                hmdc_s_pred_ave, hmdc_h_pred_ave = hmdc.inference_missing_ave(test_data_loader)

                hmdc_s_pred_ref, hmdc_h_pred_ref = hmdc.inference_missing_ref(test_data_loader=test_data_loader)

                np.savez(os.path.join(saving_fold, "predictions.npz"),
                            y_true=test_data_loader.labels,
                            y_all_labels=test_data_loader.all_labels,
                            y_pred_ran=y_pred_ran,
                            y_pred_imp=y_pred_imp,
                            br_pred=br_pred,
                            hmdc_h_imp=hmdc_h_imp,
                            hmdc_s_imp=hmdc_s_imp,
                            hmdc_h_pred_opt=hmdc_h_pred_opt,
                            hmdc_s_pred_opt=hmdc_s_pred_opt,
                            hmdc_h_pred_ave=hmdc_h_pred_ave,
                            hmdc_s_pred_ave=hmdc_s_pred_ave,
                            hmdc_s_pred_ref=hmdc_s_pred_ref,
                            hmdc_h_pred_ref=hmdc_h_pred_ref,
                        )
    

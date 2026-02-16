sys.path.append('..')
import os
import sys
import pickle
import numpy as np
import argparse
import logging
import time
import pandas as pd

from metrics import deltas_distance


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
    parser.add_argument('--baselearner', choices=BASELEARNERS, default='rf', type=str, help='Base Learner')
    parser.add_argument('--palim', type=int, default=2, help='The maximum number of parents for each node')
    # parser.add_argument('--dis-missing', type=float, default=0.5, help='The percentage of missing discrete features ')
    # parser.add_argument('--class-missing', type=float, default=0.5, help='The percentage of missing class variables')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, help='Whether or not to plot the BN structure')
    parser.add_argument('--is-missing', action=argparse.BooleanOptionalAction, default=True, help='Whether or not missing data appear in inference')
    parser.add_argument('--n-folds', type=int, default=10, help='Number of folds')
    parser.add_argument('--output', type=str, default='experiments_tabular/HMDC_prediction_deltas_paper', help='Output path')
    args = parser.parse_args()

    dataset = args.dataset
    base = args.baselearner
    palim = args.palim
    disclim = args.disclim
    is_plot = args.plot
    n_folds = args.n_folds
    output_path = args.output
    is_missing = args.is_missing
    # dis_missing = args.dis_missing
    # class_missing = args.class_missing

    dis_missing_list = [0.3, 0.8]
    class_missing_list = [0.3, 0.7, 0.8, 0.9]

    base_dir = os.path.join(output_path, base)
    os.makedirs(base_dir, exist_ok=True)
    dataset_dir = os.path.join(base_dir, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    load_dir = 'experiments_tabular/HMDC_inference/' + str(base) + '/' + str(dataset)

    for dis_missing in dis_missing_list:
        for class_missing in class_missing_list:
            ran_deltas_max, imp_deltas_max, br_deltas_max = [], [], []
            imp_gbnc_deltas_h_max, imp_gbnc_deltas_s_max = [], []
            opt_deltas_h_max, opt_deltas_s_max = [], []
            ave_deltas_h_max, ave_deltas_s_max = [], []
            ref_deltas_h_max, ref_deltas_s_max = [], [] 
            ran_deltas_ave, imp_deltas_ave, br_deltas_ave = [], [], []
            imp_gbnc_deltas_h_ave, imp_gbnc_deltas_s_ave = [], []
            opt_deltas_h_ave, opt_deltas_s_ave = [], []
            ave_deltas_h_ave, ave_deltas_s_ave = [], []
            ref_deltas_h_ave, ref_deltas_s_ave = [], [] 

            train_splits = []
            test_splits = []

            name = str(class_missing) + '_' + str(dis_missing)
            load_case_dir = os.path.join(load_dir, name)
            model_dir = os.path.join(dataset_dir, name)
            os.makedirs(model_dir, exist_ok=True)
            
            log_file = os.path.join(model_dir, 'log_file.log')
            logging.basicConfig(level=logging.INFO, filename=log_file, filemode="w")

            logging.info(f'python mixed_data_missing_inference_HMDC_Imb_Inference_loop.py -dataset {dataset} --baselearner {base} \
                    --palim {palim} --disclim {disclim} --dis-missing {dis_missing} --class-missing {class_missing} \
                    --n-folds {n_folds} --output {output_path} --is-missing {is_missing}')
        
            test_dir = os.path.join(load_case_dir, 'test_data')

            for i in range(n_folds):
                prediction_dir = os.path.join(test_dir, str(i) + '/predictions.npz')
                
                data = np.load(prediction_dir, allow_pickle=True)

                # To load it back
                with open(os.path.join(test_dir, str(i) + "/test_data_loader.pkl"), 'rb') as f:
                    test_data_loader = pickle.load(f)
                no_y = len(test_data_loader.label_names) - len(test_data_loader.discrete_feature_names)
                # Create missing indicator: 1 if missing (-1), 0 otherwise
                missing_indicator = (test_data_loader.labels[:, :no_y] == -1).astype(int)

                y_pred_ran = data['y_pred_ran'][:, :no_y]
                y_pred_imp = data['y_pred_imp'][:, :no_y]
                br_pred = data['br_pred'][:, :no_y]
                gbnc_h_imp = data['gbnc_h_imp'][:, :no_y]
                gbnc_s_imp = data['gbnc_s_imp'][:, :no_y]
                gbnc_h_pred_opt = data['gbnc_h_pred_opt'][:, :no_y]
                gbnc_s_pred_opt = data['gbnc_s_pred_opt'][:, :no_y]
                gbnc_h_pred_ave = data['gbnc_h_pred_ave'][:, :no_y]
                gbnc_s_pred_ave = data['gbnc_s_pred_ave'][:, :no_y]
                gbnc_s_pred_ref = data['gbnc_s_pred_ref'][:, :no_y]
                gbnc_h_pred_ref = data['gbnc_h_pred_ref'][:, :no_y]

                y_true = test_data_loader.all_labels[:, :no_y]
                
                ran_delta = deltas_distance(y_true=y_true, y_pred=y_pred_ran, y_card=test_data_loader.label_domains, missing_indicator=missing_indicator)
                imp_delta = deltas_distance(y_true=y_true, y_pred=y_pred_imp, y_card=test_data_loader.label_domains, missing_indicator=missing_indicator)
                br_delta = deltas_distance(y_true=y_true, y_pred=br_pred, y_card=test_data_loader.label_domains, missing_indicator=missing_indicator)
                
                imp_gbnc_delta_h = deltas_distance(y_true=y_true, y_pred=gbnc_h_imp, y_card=test_data_loader.label_domains, missing_indicator=missing_indicator)
                
                opt_delta_h = deltas_distance(y_true=y_true, y_pred=gbnc_h_pred_opt, y_card=test_data_loader.label_domains, missing_indicator=missing_indicator)
                
                ave_delta_h = deltas_distance(y_true=y_true, y_pred=gbnc_h_pred_ave, y_card=test_data_loader.label_domains, missing_indicator=missing_indicator)
                
                ref_delta_h = deltas_distance(y_true=y_true, y_pred=gbnc_h_pred_ref, y_card=test_data_loader.label_domains, missing_indicator=missing_indicator)
                
                ran_deltas_max.append(ran_delta[0])
                imp_deltas_max.append(imp_delta[0])     
                br_deltas_max.append(br_delta[0])
                imp_gbnc_deltas_h_max.append(imp_gbnc_delta_h[0])
                opt_deltas_h_max.append(opt_delta_h[0])
                ave_deltas_h_max.append(ave_delta_h[0])
                ref_deltas_h_max.append(ref_delta_h[0])

                ran_deltas_ave.append(ran_delta[1])
                imp_deltas_ave.append(imp_delta[1])     
                br_deltas_ave.append(br_delta[1])  

                imp_gbnc_deltas_h_ave.append(imp_gbnc_delta_h[1])
                opt_deltas_h_ave.append(opt_delta_h[1])
                ave_deltas_h_ave.append(ave_delta_h[1])
                ref_deltas_h_ave.append(ref_delta_h[1])
            
            # Prepare data for CSV
            data = [
                # ["Random Classifier", "Deltas", np.mean(ran_deltas_max, axis=0), np.std(ran_deltas_max, axis=0),  np.mean(ran_deltas_ave, axis=0), np.std(ran_deltas_ave, axis=0)],               
                # ["Mode Imputation", "Deltas", np.mean(imp_deltas_max, axis=0), np.std(imp_deltas_max, axis=0),  np.mean(imp_deltas_ave, axis=0), np.std(imp_deltas_ave, axis=0)],
                ["Binary Relevance", "Deltas", np.mean(br_deltas_max, axis=0), np.std(br_deltas_max, axis=0),  np.mean(br_deltas_ave, axis=0), np.std(br_deltas_ave, axis=0)],
                ["GBNC Imputation", "Deltas", np.mean(imp_gbnc_deltas_h_max, axis=0), np.std(imp_gbnc_deltas_h_max, axis=0),  np.mean(imp_gbnc_deltas_h_ave, axis=0), np.std(imp_gbnc_deltas_h_ave, axis=0)],
                ["GBNC Optimal", "Deltas", np.mean(opt_deltas_h_max, axis=0), np.std(opt_deltas_h_max, axis=0),  np.mean(opt_deltas_h_ave, axis=0), np.std(opt_deltas_h_ave, axis=0)],
                ["GBNC Average", "Deltas", np.mean(ave_deltas_h_max, axis=0), np.std(ave_deltas_h_max, axis=0),  np.mean(ave_deltas_h_ave, axis=0), np.std(ave_deltas_h_ave, axis=0)],
                ["GBNC Reference", "Deltas", np.mean(ref_deltas_h_max, axis=0), np.std(ref_deltas_h_max, axis=0),  np.mean(ref_deltas_h_ave, axis=0), np.std(ref_deltas_h_ave, axis=0)],

            ]

            # Convert to DataFrame
            df = pd.DataFrame(data, columns=["Category", "Metric", "Deltas_max - Mean (%)", "Deltas_max - Std Dev (%)", "Deltas_ave - Mean (%)", "Deltas_ave - Std Dev (%)"])

            # Save to CSV
            csv_filename = os.path.join(model_dir, name + "deltas_results.csv")
            df.to_csv(csv_filename, index=False)

            # Log success
            logging.info(f"Results saved to {csv_filename}")
            print(f"Results saved to {csv_filename}")

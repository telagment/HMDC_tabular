import os
import logging
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image
from ast import literal_eval
from scipy.io import loadmat

from utils import missing_data_total

class TabularDataLoader_HMDC:
     # For tabular data with discrete and continuous features DAG: Xc->(Y + X_d)
    def __init__(self,
                 csv_path=None, arff_path=None, xlsx_path=None , n_labels=None,
                 mat_path=None, dataset = 'Adult'):
        if csv_path is not None and arff_path is not None and n_labels is not None:
            # print('Read data from the arff file ...')
            csv_file = open(csv_path, 'r')
            arff_file = open(arff_path, 'r')
            arff_lines = arff_file.readlines()
            continuous_feature_names = []
            discrete_attr_names = []
            discrete_attr_domains = {}
            for line in arff_lines:
                if '@attribute' in line or '@ATTRIBUTE' in line:
                    attributes = line.split()
                    if attributes[-1] == 'numeric':
                        continuous_feature_names.append(attributes[-2])
                    else:
                        discrete_attr_names.append(attributes[-2])
                        discrete_attr_domains[attributes[-2]] = literal_eval(attributes[-1])
                elif '@data' in line or '@DATA' in line:
                    break
            discrete_feature_names = discrete_attr_names[0:-n_labels]
            label_names = discrete_attr_names[-n_labels:]

            df = pd.read_csv(csv_file)

            self.labels = df.loc[:, label_names].values
            self.all_labels = df.loc[:, label_names].values
            self.label_names = [str(i) for i in range(len(label_names))]
            self.label_domains = {k: discrete_attr_domains[n] for k, n in zip(self.label_names, label_names)}
            self.continuous_features = df.loc[:, continuous_feature_names].values
            self.discrete_features = df.loc[:, discrete_feature_names].values
            # self.discrete_feature_names = [chr(i + 97) for i in range(len(discrete_feature_names))]
            self.discrete_feature_names = [str(i + int(len(label_names))) for i in range(len(discrete_feature_names))]
            self.discrete_feature_domains = {k: discrete_attr_domains[n] for k, n in zip(self.discrete_feature_names,
                                                                                         discrete_feature_names)}
        elif mat_path is not None:
            print('Read data from the mat file ...')
            mat_data = loadmat(mat_path)
            orig_data = mat_data['data']['orig'][0][0]
            norm_data = mat_data['data']['norm'][0][0]
            labels = mat_data['target']
            # matlab's index starts from 1
            continuous_feature_indices = mat_data['data_type']['c'][0][0][0] - 1
            non_ordinal_feature_indices = mat_data['data_type']['d_wo_o'][0][0] - 1
            ordinal_feature_indices = mat_data['data_type']['d_w_o'][0][0] - 1
            binary_feature_indices = mat_data['data_type']['b'][0][0] - 1
            discrete_feature_indices = np.concatenate((non_ordinal_feature_indices, ordinal_feature_indices,
                                                       binary_feature_indices), axis=None)
            self.labels = labels - 1
            self.all_labels = labels - 1
            self.label_names = [str(i) for i in range(self.labels.shape[1])]
            self.label_domains = {}
            for i in range(labels.shape[1]):
                self.label_domains[str(i)] = np.unique(self.labels[:, i])

            if discrete_feature_indices.shape[0] != 0:
                self.continuous_features = norm_data[:, continuous_feature_indices]
                self.discrete_features = orig_data[:, discrete_feature_indices]
                # To ensure all labels start from 0
                self.discrete_features = self.discrete_features - self.discrete_features.min(axis=0)
                if dataset == 'Default':
                    # Apply mapping: Only replace 4 → 3, keep others unchanged
                    self.discrete_features[:, 1:]  = np.where(self.discrete_features[:, 1:]  == 4, 3, self.discrete_features[:, 1:])
                # 1st way: Call discrete feature names as a, b, c, ...
                # self.discrete_feature_names = [chr(i + 97) for i in range(len(discrete_feature_indices))]
                # 2nd way: Call discrete feature names as number (1, 2, 3,... + number of class variables Y_K)
                # K = labels.shape[1]
                self.discrete_feature_names = [str(i + int(labels.shape[1])) for i in range(len(discrete_feature_indices))]
                self.discrete_feature_domains = {}
                for i, feature_idx in enumerate(discrete_feature_indices):
                    self.discrete_feature_domains[str(i + int(labels.shape[1]))] = np.unique(self.discrete_features[:, i]) 
                    # self.discrete_feature_domains[str(i + int(labels.shape[1]))] = np.unique(self.discrete_features[:, feature_idx]) 
                    # discrete_feature_domains[chr(i + 97)] = np.unique(orig_data[:, feature_idx]) - 1
            else:
                self.continuous_features = norm_data
                self.discrete_feature_names = None
                self.discrete_feature_domains = None
        else:
            self.continuous_features = None
            self.discrete_features = None
            self.discrete_feature_names = None
            self.discrete_feature_domains = None
            self.labels = None
            self.all_labels = None
            self.label_names = None
            self.label_domains = None
    
    def get_slices(self, node, parents, configuration, weight=None):
        return self.__get_slices(node, parents, configuration, weight)

    def __get_slices(self, node, parents, configuration, weight=None):
        node_idx = self.label_names.index(node)
        label_name_indices = []
        label_configuration = []
        for i, parent in enumerate(parents):
            label_name_indices.append(self.label_names.index(parent))
            label_configuration.append(configuration[i])

        indices = np.logical_and.reduce(self.labels[:, label_name_indices] == label_configuration, axis=-1)

        X = self.continuous_features[indices]
        y = self.labels[indices][:, node_idx]

        return (X, y) if weight is None else (X, y, weight[indices])

    def create_sub_data_loader(self, indices, missing_class_var=1, missing_dis_fea=1, is_missing=False, is_disc=True):
        data_loader = TabularDataLoader_HMDC() # Keeping original instantiation

        data_loader.continuous_features = self.continuous_features[indices]
        data_loader.discrete_feature_names = self.discrete_feature_names
        data_loader.discrete_feature_domains = self.discrete_feature_domains

        # Creating and Handling missing values
        if is_missing == True:
            data_loader.labels = missing_data_total(self.labels[indices], missing_fraction=missing_class_var)
            data_loader.labels = np.nan_to_num(data_loader.labels, nan=-1).astype(int)
            data_loader.discrete_features = missing_data_total(self.discrete_features[indices], missing_fraction=missing_dis_fea)
            data_loader.discrete_features = np.nan_to_num(data_loader.discrete_features, nan=-1).astype(int)
        else:
            data_loader.labels = self.labels[indices]
            data_loader.discrete_features = (
                                            self.discrete_features[indices].astype(int)
                                            if self.discrete_features is not None else None
                                        )

        # Include discrete features (X_d) as class variables/labels if True
        if is_disc == True:
            data_loader.all_labels = np.concatenate((self.labels[indices], self.discrete_features[indices]), axis=1)  # Store True values of missing data in discrete features
            data_loader.label_names = self.label_names + self.discrete_feature_names
            data_loader.label_domains = {**self.label_domains, **self.discrete_feature_domains}
            data_loader.labels = np.concatenate((data_loader.labels, data_loader.discrete_features), axis=1)      
        else:
            data_loader.all_labels = self.labels[indices] 
            data_loader.label_names = self.label_names
            data_loader.label_domains = self.label_domains
        return data_loader

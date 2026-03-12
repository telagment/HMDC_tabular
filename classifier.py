import itertools
import networkx as nx
import numpy as np
import os
import torch
import torch.nn as nn
import pandas as pd
import logging
from tqdm import tqdm
import itertools
import joblib
from concurrent.futures import as_completed, ThreadPoolExecutor

from data import TabularDataLoader_HMDC
from utils import read_local_scores
from utils import subset, create_evidence, check_missing_node
from utils import train_tabular_base_clfs
from utils import compute_cll_tabular_clfs
from utils import predict_proba_base_clfs
from utils import split_dict_by_key_range

from sklearn.base import ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from scipy.stats import mode


from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference.ExactInference import VariableElimination
from pygobnilp.gobnilp import Gobnilp
from scipy import stats


class DataProcessor:
    @staticmethod
    def _deduplicate_instances(x: np.ndarray, y: np.ndarray, weight: np.ndarray):
        """
        Removes duplicate (x_i, y_i) instances and sums corresponding weight_i.

        Parameters:
            x (np.ndarray): Feature matrix of shape (n_samples, n_features)
            y (np.ndarray): Label matrix of shape (n_samples, n_labels)
            weight (np.ndarray): Weight matrix of shape (n_samples)

        Returns:
            unique_x (np.ndarray): Deduplicated feature matrix
            unique_y (np.ndarray): Deduplicated label matrix
            new_weight (np.ndarray): Summed weights for duplicates
        """
        # Stack x and y to create unique identifiers for each instance
        xy = np.hstack((x, y))

        # Identify unique (x, y) pairs and get inverse indices
        unique_xy, inverse_indices = np.unique(xy, axis=0, return_inverse=True)

        # Split unique_xy back into x and y
        num_features = x.shape[1]
        unique_x = unique_xy[:, :num_features]
        unique_y = unique_xy[:, num_features:]

        # Initialize new weight matrix
        new_weight = np.ones(len(unique_x), dtype=float) 

        # Sum weights for duplicates
        for i in range(len(weight)):
            new_weight[inverse_indices[i]] += weight[i]

        return unique_x, unique_y, new_weight
    
    
    @staticmethod
    def _impute_labels_with_mode(labels):
        """
        Impute missing labels (-1) in a 2D array using mode per column.

        Args:
            labels (np.ndarray): Array of shape (n_samples, n_classes) with -1 as missing label marker.

        Returns:
            imputed_labels (np.ndarray): Labels with missing values imputed.
            missing_mask (np.ndarray): Boolean mask, True where original labels were missing.
        """
        labels = np.array(labels)
        missing_mask = (labels == -1)
        imputed_labels = labels.copy()

        for col in range(labels.shape[1]):
            valid_values = labels[labels[:, col] != -1, col]
            if len(valid_values) == 0:
                raise ValueError(f"All labels are missing in column {col}, cannot compute mode.")
            mode_val = stats.mode(valid_values, keepdims=False).mode
            imputed_labels[missing_mask[:, col], col] = mode_val

        return imputed_labels, missing_mask

    def update_dataloader(self, train_data_loader):
        """
        Update the labels of the given train_data_loader with imputed values.
        
        Args:
            train_data_loader: An object with a `labels` attribute (2D array-like).
        """
        imputed_labels, missing_mask = self._impute_labels_with_mode(train_data_loader.labels)
        train_data_loader.labels = imputed_labels
        train_data_loader.missing_mask = missing_mask

class TabularClassifier_HMDC:
    def __init__(self,
                 palim: int = 2,
                 apha: int = 1,
                #  penalty: str = None,
                 result_path: str = None,
                 base: str = None):
        self.palim = palim
        self.base = base
        self.apha = apha
        self.result_path = result_path

        self.label_domains = None
        self.discrete_feature_domains = None
        self.selected_discrete_features = None
        self.classifiers = None
        self.parent_dict = None
        self.scores = None
        self.__best_subset_scores = None
        self.bn = None

    def __init_structures(self, label_domains, discrete_feature_domains=None):
        self.label_domains = label_domains
        self.discrete_feature_domains = discrete_feature_domains
        if discrete_feature_domains:
            self.selected_discrete_features = {node: {} for node in list(label_domains.keys())}
        self.classifiers = {node: {} for node in list(label_domains.keys())}
        # A dictionary that scores[i][parents] stores the CLL score for node i given its parents
        self.scores = {node: {} for node in list(label_domains.keys())}
        self.__best_subset_scores = {node: {} for node in list(label_domains.keys())}
        self.bn = None

    def _build_instance_bn(self, test_feature, bn):
        """Build a Bayesian Network with CPDs based on the current test instance."""

        for child, parents in self.parent_dict.items():
            parent_set = frozenset(parents)
            parent_domains = [self.label_domains[p] for p in parents]
            configurations = list(itertools.product(*parent_domains))

            cp_matrix = np.empty((len(self.label_domains[child]), len(configurations)), dtype=np.float32)

            for j, config in enumerate(configurations):
                clf = self.classifiers[child][parent_set][tuple(config)]
                probs = predict_proba_base_clfs(clf, test_feature)

                if isinstance(clf, ClassifierMixin) and len(self.label_domains[child]) > len(clf.classes_):
                    corrected = np.zeros(len(self.label_domains[child]), dtype=np.float32)
                    for idx_c, c in enumerate(clf.classes_):
                        corrected[c] = probs[:, idx_c]
                    probs = corrected

                cp_matrix[:, j] = probs

            cpd = TabularCPD(
                variable=child,
                variable_card=len(self.label_domains[child]),
                values=cp_matrix,
                evidence=parents,
                evidence_card=[len(self.label_domains[p]) for p in parents]
            )
            bn.add_cpds(cpd)

        return bn
        
    def fit(self, train_data_loader: TabularDataLoader_HMDC, pruning_score: str = 'BIC', show_log=False, save_baselearner=False):
        self.__init_structures(train_data_loader.label_domains)
        all_nodes = list(self.label_domains.keys())
        candidate_child_nodes = list(self.label_domains.keys())
        stopped_parents = {node: [] for node in candidate_child_nodes}
        for n_parents in tqdm(range(self.palim + 1)):
            all_parents = {node: list(subset(list(set(all_nodes) - {node}), n_elements=n_parents))
                           for node in candidate_child_nodes}
            for node in list(all_parents.keys()):
                cur_parents = all_parents[node]
                is_stopped_cur_parents = [False for _ in range(len(cur_parents))]
                for parents_idx, parents in enumerate(cur_parents):
                    parent_set = frozenset(parents)
                    parent_domains = [self.label_domains[parent_id] for parent_id in parents]
                    configurations = list(itertools.product(*parent_domains))
                    # penalty = -0.5 * np.log(len(train_data_loader.continuous_features)) \
                    #           * len(configurations) * (len(self.label_domains[node]) - 1)
                    if pruning_score == 'AIC':
                        penalty = -len(configurations) * (len(self.label_domains[node]) - 1) 
                    else:  # pruning_score == 'BIC'
                        penalty = -0.5 * np.log(len(train_data_loader.continuous_features)) \
                                  * len(configurations) * (len(self.label_domains[node]) - 1) 
                    if show_log:
                        logging.info('{:<10}{:<5}'
                              '{:<10}{:<30}'
                              '{:<20}{:<20.3f}'.format('Node:', node,
                                                       'Parents:', '{}'.format(list(parents)),
                                                       'Penalty:', penalty))
                    if len(parents) != 0:
                        parent_subsets = list(subset(parents, len(parents) - 1))
                        if not set(parent_subsets).isdisjoint(set(stopped_parents[node])):
                            stopped_parents[node].append(parents)
                            is_stopped_cur_parents[parents_idx] = True
                            continue
                        subset_scores = [self.__best_subset_scores[node][frozenset(parent_subset)]
                                         for parent_subset in parent_subsets]
                        best_subset_score = max(subset_scores)
                        if best_subset_score >= penalty:
                            self.__best_subset_scores[node][parent_set] = best_subset_score
                            stopped_parents[node].append(parents)
                            is_stopped_cur_parents[parents_idx] = True
                            if show_log:
                                logging.info('{:<10}{:<5}'
                                      '{:<10}{:<30}'
                                      '{}'.format('Node:', node,
                                                  'Parents:', '{}'.format(list(parents)),
                                                  'Pruned and stop growing!'))
                            continue
                    config_clfs = dict()
                    scoring_function = penalty
                    for configuration in configurations:
                        configuration = tuple(configuration)
                        local_features, local_labels = train_data_loader.get_slices(node, parents, configuration)
                        if local_features.shape[0] == 0:
                            logging.info('Empty data!')
                            clf = [1 / len(self.label_domains[node])] * len(self.label_domains[node])
                            config_clfs[configuration] = clf
                            continue
                        if (local_features.std(axis=0) == 0).all(): 
                            logging.info("All features are constant!")
                            clf = [1 / len(self.label_domains[node])] * len(self.label_domains[node])
                            config_clfs[configuration] = clf
                            continue
                        if np.all(local_labels == local_labels[0]):
                            logging.info('All labels are the same!')
                            if not len(self.label_domains[node]):
                                print(f"Error: self.label_domains[{node}] is empty!")
                            else:
                                clf = [0] * len(self.label_domains[node])
                            clf[int(local_labels[0])] = 1
                            config_clfs[configuration] = clf
                            continue     
                        else:
                            clf = self._create_base_clfs(self.base)
                        if isinstance(clf, torch.nn.Module):
                            x_train, y_train = torch.Tensor(local_features), torch.Tensor(local_labels)
                            y_train = y_train.type(torch.LongTensor)
                            train_tabular_base_clfs(model=clf, x_train=x_train, y_train=y_train)
                        elif isinstance(clf, (ClassifierMixin, list)):
                            x_train, y_train = local_features, local_labels
                            train_tabular_base_clfs(model=clf, x_train=x_train, y_train=y_train)
                        else:
                            raise NotImplementedError
                        config_clfs[configuration] = clf
                        scoring_function += compute_cll_tabular_clfs(model=clf, x_train=x_train, y_train=y_train)
                    if show_log:
                        logging.info('{:<10}{:<5}'
                              '{:<10}{:<30}'
                              '{:<20}{:<20.3f}'.format('Node:', node,
                                                       'Parents:', '{}'.format(list(parents)),
                                                       'CLL Score:', scoring_function))
                    self.__best_subset_scores[node][parent_set] = scoring_function
                    if len(parents) != 0:
                        parent_subsets = list(subset(parents, len(parents) - 1))
                        subset_scores = [self.__best_subset_scores[node][frozenset(parent_subset)]
                                        for parent_subset in parent_subsets]
                        best_subset_score = max(subset_scores)
                        if best_subset_score >= scoring_function:
                            self.__best_subset_scores[node][parent_set] = best_subset_score
                            if show_log:
                                logging.info('{:<10}{:<5}'
                                      '{:<10}{:<30}'
                                      '{}'.format('Node:', node,
                                                  'Parents:', '{}'.format(list(parents)),
                                                  'Pruned!'))
                            continue
                    self.classifiers[node][parent_set] = config_clfs
                    self.scores[node][parent_set] = scoring_function
                if np.all(is_stopped_cur_parents):
                    if show_log:
                        logging.info('{:<10}{:<5}{}'.format('Node:', node, 'Stopped growing!'))
                    candidate_child_nodes.remove(node)
        if save_baselearner == True:
            filepath = os.path.join(self.result_path, 'classifiers.pkl')
            joblib.dump(self.classifiers, filepath)
            scorepath = os.path.join(self.result_path, 'scores.npy')
            np.save(scorepath, self.scores)

    def learn_structure(self, pruning_score='BIC'):
        m = Gobnilp()
        if not (self.result_path is None):
            output_path = os.path.join(self.result_path, 'score')
        else:
            output_path = None

        if pruning_score == 'AIC':
            str_score = 'DiscreteAIC'
        else:            
            str_score = 'DiscreteBIC'

        m.learn(local_scores_source=self.scores, score=str_score, palim=self.palim, output_scores=output_path)
        self.bn = m.learned_bn
        parent_dict = {node: [] for node in list(self.label_domains.keys())}

        for edge in list(self.bn.edges):
            parent_dict[edge[1]].append(edge[0])
        for k, v in parent_dict.items():
            parent_dict[k] = sorted(v)
        self.parent_dict = parent_dict
        # Prune the clfs
        for node in list(self.label_domains.keys()):
            for parent_set in list(self.classifiers[node].keys()):
                if frozenset(parent_dict[node]) != parent_set:
                    self.classifiers[node].pop(parent_set, None)
        
    def load_DAG(self, score_path, classifier_path, data_loader: TabularDataLoader_HMDC):
        self.scores = read_local_scores(score_path)
        self.classifiers = joblib.load(classifier_path)
        self.label_domains = data_loader.label_domains
        
    def inference(self, test_data_loader, is_evidence=False):
        # Defind Baysian Networks
        bn = BayesianNetwork(self.bn)

        test_features = test_data_loader.continuous_features

        pred_labels_01 = np.empty(shape=(len(test_features), len(self.label_domains)))
        pred_labels_h = np.empty(shape=(len(test_features), len(self.label_domains)))

        for i, test_feature in enumerate(tqdm(test_features)):
            # Build BN with CPDs for this test sample
            bn = self._build_instance_bn(test_feature, bn)
            assert bn.check_model()

            inferencer = VariableElimination(bn)
            
            if is_evidence:
                labels = test_data_loader.labels[i]
                evidence_nodes = test_data_loader.discrete_feature_names
                variable_nodes = [n for n in bn.nodes() if n not in evidence_nodes]
                evidences = create_evidence(label_nodes=labels, label_domains=self.label_domains, evidence_nodes=evidence_nodes)
            else:
                evidences = None
                variable_nodes = list(bn.nodes())
            '''Inference with Subset 0/1 Loss - calculate MAP/MPE (transform BOP problem to MPE problem)'''
            pred_results = inferencer.map_query(variables=variable_nodes, evidence=evidences, elimination_order="MinWeight")
            for v in pred_results.keys():
                pred_labels_01[i, int(v)] = pred_results[v] 

            '''Inference with Hamming Loss - Find K max marginal probability'''
            for k in variable_nodes:
                marg = inferencer.query(variables=[k], evidence=evidences)
                pred_labels_h[i, int(k)] = np.argmax(marg.values)
            bn.cpds = []
        
        return pred_labels_01, pred_labels_h


    def inference_imp(self, impu, test_data_loader): 
         
        ''' Subset 0/1 Loss => transform BOP to MPE problem (join probability)
            Hamming Loss =>  transform BOP to K Marginal problem

            loss= {'sub01' - Inference with Subset 0/1 Loss, 'hamming' - Inference with Hamming Loss, 'all'/other case - Inference with all}
            Inference with wrong evidence (impute evidence - 1. random predict 2. most frequent value)
            impu: impute evidence - y_1, ..., y_k, x_{d,1}, ..., x_{d,m})
        '''           
        # Defind Baysian Networks
        test_features = test_data_loader.continuous_features
        bn = BayesianNetwork(self.bn)

        pred_labels_01 = np.empty(shape=(len(test_features), len(self.label_domains)))
        pred_labels_h = np.empty(shape=(len(test_features), len(self.label_domains)))

        for i, test_feature in enumerate(tqdm(test_features)):
            bn = self._build_instance_bn(test_feature)
            assert bn.check_model()

            inferencer = VariableElimination(bn)
            # Make predictions based on the specified loss function
            labels = test_data_loader.labels[i]
            evidences = create_evidence(label_nodes=labels, label_domains=self.label_domains, evidence_nodes=list(bn.nodes()))
            no_class = int(test_data_loader.labels.shape[1]) - int(test_data_loader.discrete_features.shape[1])
            class_evidence, disc_evidence = split_dict_by_key_range(evidences, range(no_class))
            evidences = {
                        key: impu[i, int(key)] for key, _ in disc_evidence.items()
                    }      
            variable_nodes = list(class_evidence.keys())
           
            '''Inference with Subset 0/1 Loss - calculate MAP/MPE (transform BOP problem to MPE problem)'''
            pred_results = inferencer.map_query(variables=variable_nodes, evidence=evidences, elimination_order="MinWeight")
            for v in pred_results.keys():
                pred_labels_01[i, int(v)] = pred_results[v] 

            '''Inference with Hamming Loss - Find K max marginal probability'''
            for k in variable_nodes:
                pred_labels_h[i, int(k)] = inferencer.map_query(variables=[k], evidence=evidences, elimination_order="MinWeight")[k]
            bn.cpds = []
        return pred_labels_01, pred_labels_h 
        

    def inference_missing_imp(self, train_data_loader, test_data_loader):  
        ''' Subset 0/1 Loss => transform BOP to MPE problem (join probability)
            Hamming Loss =>  transform BOP to K Marginal problem

            loss= {'sub01' - Inference with Subset 0/1 Loss, 'hamming' - Inference with Hamming Loss, 'all'/other case - Inference with all}
            Inference with Missing data 
                - Missing at discreate features and classvariables 

                - Concept: Setting 1 
                    + impute missing discreate features by the most frequent value in training data
                    + missing class variables as discreate features (imputated missing discreate features + observed discreate features) are observed
        '''           
        # Defind Baysian Networks
        test_features = test_data_loader.continuous_features
        bn = BayesianNetwork(self.bn)

        pred_labels_01 = np.empty(shape=(len(test_features), len(self.label_domains)))
        pred_labels_h = np.empty(shape=(len(test_features), len(self.label_domains)))

        imputation = mode(train_data_loader.labels, axis=0).mode[0] 

        for i, test_feature in enumerate(tqdm(test_features)):
            disc_evidences = {}
            class_evidence = {}
            for child, parents in self.parent_dict.items():
                parent_set = frozenset(parents)
                parent_domains = [self.label_domains[parent_id] for parent_id in parents]
                configurations = list(itertools.product(*parent_domains))
                cp_matrix = np.empty(shape=(len(self.label_domains[child]), len(configurations)), dtype=np.float32)
                for j, configuration in enumerate(configurations):
                    clf = self.classifiers[child][parent_set][tuple(configuration)]
                    cond_probs = predict_proba_base_clfs(clf, test_feature)
                    if isinstance(clf, ClassifierMixin) and len(self.label_domains[child]) > len(clf.classes_):
                        corrected_cond_probs = np.zeros(shape=len(self.label_domains[child]), dtype=np.float32)
                        for idx_c, c in enumerate(clf.classes_):
                            corrected_cond_probs[c] = cond_probs[:, idx_c]
                        cond_probs = corrected_cond_probs
                    cp_matrix[:, j] = cond_probs
                cpd = TabularCPD(variable=child, variable_card=len(self.label_domains[child]), 
                                 values=cp_matrix, evidence=parents,
                                 evidence_card=[len(self.label_domains[parent]) for parent in parents])
                bn.add_cpds(cpd)

            # Check the model
            assert bn.check_model()
            inferencer = VariableElimination(bn)
            # Make predictions based on the specified loss function
            labels = test_data_loader.labels[i]
            all_evidences = create_evidence(label_nodes=labels, label_domains=self.label_domains, evidence_nodes=list(bn.nodes()))
            no_class = int(test_data_loader.labels.shape[1]) -int(test_data_loader.discrete_features.shape[1])
            class_evidence, disc_evidence = split_dict_by_key_range(all_evidences, range(no_class))
            missing_class = check_missing_node(dict_data=class_evidence)
            # missing_disc = check_missing_node(dict_data=disc_evidence)

            if missing_class != None:
                # Update the variable and envidence nodes
                disc_evidences = {
                                key: (imputation[int(key)] if value is None else value)
                                for key, value in disc_evidence.items()
                            } 
                class_evidence = {key: value for key, value in class_evidence.items() if value is not None}
                evidences = {**disc_evidences, **class_evidence}    
                variable_nodes = missing_class
            else:
                continue

            '''Inference with Subset 0/1 Loss - calculate MAP/MPE (transform BOP problem to MPE problem)'''
            pred_results = inferencer.map_query(variables=variable_nodes, evidence=evidences, elimination_order="MinWeight", show_progress=False)
            for v in pred_results.keys():
                pred_labels_01[i, int(v)] = pred_results[v] 

            '''Inference with Hamming Loss - Find K max marginal probability'''
            for k in variable_nodes:
                pred_labels_h[i, int(k)] = inferencer.map_query(variables=[k], evidence=evidences, elimination_order="MinWeight", show_progress=False)[k]
            
            bn.cpds = []

        return pred_labels_01, pred_labels_h 
    
    def inference_missing_opt(self, test_data_loader):  
        # Optimistic approach
        ''' Subset 0/1 Loss => transform BOP to MPE problem (join probability)
            Hamming Loss =>  transform BOP to K Marginal problem

            loss= {'sub01' - Inference with Subset 0/1 Loss, 'hamming' - Inference with Hamming Loss, 'all'/other case - Inference with all}
            is_DeEnvi = False/True - be Inference with demographic information (Age/Gender(Sex)) or not

            Inference with Missing data 
                - Missing at discreate features and classvariables 
                - Concept: Setting 1 - Predict the missing discreate feature + missing class variables
        '''           
        # Defind Baysian Networks
        test_features = test_data_loader.continuous_features
        bn = BayesianNetwork(self.bn)

        pred_labels_01 = np.empty(shape=(len(test_features), len(self.label_domains)))
        pred_labels_h = np.empty(shape=(len(test_features), len(self.label_domains)))
        no_class = int(test_data_loader.labels.shape[1]) -int(test_data_loader.discrete_features.shape[1])
        
        for i, test_feature in enumerate(tqdm(test_features)):
            labels = test_data_loader.labels[i]

            # Check for missing class variables before doing any heavy work
            evidences = create_evidence(labels, test_data_loader.label_domains, list(self.bn.nodes()))
            class_evidence, _ = split_dict_by_key_range(evidences, range(no_class))
            missing_nodes = check_missing_node(class_evidence)

            if not missing_nodes:
                continue # No missing class variables 

            # Build BN with CPDs for this test sample
            bn = self._build_instance_bn(test_feature, bn)
            assert bn.check_model()

            inferencer = VariableElimination(bn)

            # Step 4: Update evidence and variable nodes
            variable_nodes = check_missing_node(evidences)
            evidences = {k: v for k, v in evidences.items() if v is not None}
            
            '''Inference with Subset 0/1 Loss - calculate MAP/MPE (transform BOP problem to MPE problem)'''

            pred_results = inferencer.map_query(variables=variable_nodes, evidence=evidences, elimination_order="MinWeight", show_progress=False)
            for v in pred_results.keys():
                pred_labels_01[i, int(v)] = pred_results[v] 

            '''Inference with Hamming Loss - Find K max marginal probability'''
            for k in variable_nodes:
                marg = inferencer.query(variables=[k], evidence=evidences)
                pred_labels_h[i, int(k)] = np.argmax(marg.values)
                # pred_labels_h[i, int(k)] = inferencer.map_query(variables=[k], evidence=evidences, elimination_order="MinWeight")[k]
            bn.cpds = []

        return pred_labels_01, pred_labels_h 
    
    def inference_missing_ave(self, test_data_loader):  
        # Averaging approach
        ''' Subset 0/1 Loss => transform BOP to MPE problem (join probability)
            Hamming Loss =>  transform BOP to K Marginal problem

            loss= {'sub01' - Inference with Subset 0/1 Loss, 'hamming' - Inference with Hamming Loss, 'all'/other case - Inference with all}
            is_DeEnvi = False/True - be Inference with demographic information (Age/Gender(Sex)) or not

            Inference with Missing data 
                - Missing at discreate features + class variables
                - Concept: Setting 2 - Predict only class variables missing
        '''           
        # Defind Baysian Networks
        test_features = test_data_loader.continuous_features
        bn = BayesianNetwork(self.bn)

        pred_labels_01 = np.empty(shape=(len(test_features), len(self.label_domains)))
        pred_labels_h = np.empty(shape=(len(test_features), len(self.label_domains)))
        no_class = int(test_data_loader.labels.shape[1]) -int(test_data_loader.discrete_features.shape[1])

        for i, test_feature in enumerate(tqdm(test_features)):
            # Get the labels for this test instance
            labels = test_data_loader.labels[i]

            # Check for missing class variables before doing any heavy work
            evidences = create_evidence(labels, test_data_loader.label_domains, list(self.bn.nodes()))
            class_evidence, _ = split_dict_by_key_range(evidences, range(no_class))
            missing_nodes = check_missing_node(class_evidence)

            if not missing_nodes:
                continue # No missing class variables — skip this sample

            # Build BN with CPDs for this test sample
            bn = self._build_instance_bn(test_feature, bn)
            assert bn.check_model()

            inferencer = VariableElimination(bn)

            # Update evidence and variable nodes
            evidences = {k: v for k, v in evidences.items() if v is not None}
            variable_nodes = check_missing_node(class_evidence)

            '''Inference with Subset 0/1 Loss - calculate MAP/MPE (transform BOP problem to MPE problem)'''
            pred_results = inferencer.map_query(variables=variable_nodes, evidence=evidences, elimination_order="MinWeight", show_progress=False)
            for v in pred_results.keys():
                pred_labels_01[i, int(v)] = pred_results[v] 

            '''Inference with Hamming Loss - Find K max marginal probability'''
            for k in variable_nodes:
                marg = inferencer.query(variables=[k], evidence=evidences)
                pred_labels_h[i, int(k)] = np.argmax(marg.values)
                # pred_labels_h[i, int(k)] = inferencer.map_query(variables=[k], evidence=evidences, elimination_order="MinWeight")[k]
            bn.cpds = []

        return pred_labels_01, pred_labels_h 

    def inference_missing_ref(self, test_data_loader):  
        ''' Subset 0/1 Loss => transform BOP to MPE problem (join probability)
            Hamming Loss =>  transform BOP to K Marginal problem

            Inference with Missing data 
                - Missing at discreate features and classvariables 
                - Concept: Setting 3 - Predict the missing class variables with true evidence of discreate features and obsered class variables
        '''           
        # Defind Baysian Networks
        test_features = test_data_loader.continuous_features
        bn = BayesianNetwork(self.bn)

        pred_labels_01 = np.empty(shape=(len(test_features), len(self.label_domains)))
        pred_labels_h = np.empty(shape=(len(test_features), len(self.label_domains)))
        no_class = int(test_data_loader.labels.shape[1]) -int(test_data_loader.discrete_features.shape[1])

        for i, test_feature in enumerate(tqdm(test_features)):
            variable_nodes, evidences = [], {}

            # Get the labels for this test instance
            labels = test_data_loader.labels[i]

            # Check for missing class variables before doing any heavy work
            evidences = create_evidence(labels, test_data_loader.label_domains, list(self.bn.nodes()))
            class_evidence, _ = split_dict_by_key_range(evidences, range(no_class))
            missing_nodes = check_missing_node(class_evidence)

            if not missing_nodes:
                continue # No missing class variables — skip this sample

           # Build BN with CPDs for this test sample
            bn = self._build_instance_bn(test_feature, bn)
            assert bn.check_model()
            
            inferencer = VariableElimination(bn)

            # Update evidence and variable nodes
            class_evidences = {key: value for key, value in class_evidence.items() if value is not None}
            disc_evidences = create_evidence(label_nodes=test_data_loader.all_labels[i], 
                                                label_domains=test_data_loader.label_domains, 
                                                evidence_nodes=test_data_loader.discrete_feature_names)
            evidences = {**disc_evidences, **class_evidences}
            variable_nodes = missing_nodes
                
            '''Inference with Subset 0/1 Loss - calculate MAP/MPE (transform BOP problem to MPE problem)'''
            pred_results = inferencer.map_query(variables=variable_nodes, evidence=evidences, elimination_order="MinWeight", show_progress=False)
            for v in pred_results.keys():
                pred_labels_01[i, int(v)] = pred_results[v] 

            '''Inference with Hamming Loss - Find K max marginal probability'''
            for k in variable_nodes:
                marg = inferencer.query(variables=[k], evidence=evidences)
                pred_labels_h[i, int(k)] = np.argmax(marg.values)
                # pred_labels_h[i, int(k)] = inferencer.map_query(variables=[k], evidence=evidences, elimination_order="MinWeight")[k]
            bn.cpds = []

        return pred_labels_01, pred_labels_h      
       
    @staticmethod
    def _train_single_config(args):
        # print(f"[PID {os.getpid()}] Processing configuration: {args[2]}")
        node, parents, configuration, label_domains, train_data_loader, base_model, weight = args

        configuration = tuple(configuration)

        if weight is None:
            local_features, local_labels = train_data_loader.get_slices(node, parents, configuration)
            local_weights = None
        else:
            local_features, local_labels, local_weights = train_data_loader.get_slices(node, parents, configuration, weight)

        if local_features.shape[0] == 0:
            clf = [1 / len(label_domains[node])] * len(label_domains[node])
            return configuration, clf, 0

        if (local_features.std(axis=0) == 0).all():
            clf = [1 / len(label_domains[node])] * len(label_domains[node])
            return configuration, clf, 0

        if np.all(local_labels == local_labels[0]):
            clf = [0] * len(label_domains[node])
            label_idx = int(local_labels[0]) if isinstance(local_labels[0], (int, np.integer)) \
                        else list(label_domains[node]).index(local_labels[0])
            clf[label_idx] = 1
            return configuration, clf, 0

        clf = TabularClassifier_HMDC._create_base_clfs(base_model)
        
        if isinstance(clf, torch.nn.Module):
            x_train, y_train = torch.Tensor(local_features), torch.Tensor(local_labels).type(torch.LongTensor)
            train_tabular_base_clfs(model=clf, x_train=x_train, y_train=y_train)
        elif isinstance(clf, (ClassifierMixin, list)):
            x_train, y_train = local_features, local_labels
            train_tabular_base_clfs(model=clf, x_train=x_train, y_train=y_train)
        else:
            raise NotImplementedError

        cll_score = compute_cll_tabular_clfs(model=clf, x_train=x_train, y_train=y_train, weight=local_weights)
        return configuration, clf, cll_score
    
    @staticmethod
    def _create_base_clfs(base):
        # print('base:', base)
        if base == 'lr':
            return SGDClassifier(loss='log_loss', n_jobs=-1)
        elif base == 'nb':
            return GaussianNB(var_smoothing=1e-5)
        elif base == 'rf':
            return RandomForestClassifier(n_jobs=-1)
        elif base == 'svm':
            return SVC(probability=True)
        else:
            raise NotImplementedError
        
    @staticmethod
    def _create_gpu_base_clfs(base):
        if base == 'lr':
            return LogisticRegressionModel()
        elif base == 'nb':
            return GaussianNB(var_smoothing=1e-5)
        elif base == 'rf':
            return RandomForestClassifier(n_jobs=-1)
        elif base == 'svm':
            return SVC(probability=True)
        else:
            raise NotImplementedError
        
    @staticmethod
    def _inference_worker(i, test_feature, bn, variable_nodes, build_instance_bn_func):

        # build instance BN with provided function
        bn = build_instance_bn_func(test_feature, bn)
        assert bn.check_model()

        inferencer = VariableElimination(bn)

        # Subset 0/1 Loss inference
        pred_results = inferencer.map_query(
            variables=variable_nodes,
            elimination_order="MinWeight", 
            show_progress=False
        )
        pred_label_01 = {int(v): pred_results[v] for v in pred_results.keys()}

        # Hamming Loss inference (max marginals)
        pred_label_h = {}
        for k in variable_nodes:
            marg = inferencer.query(variables=[k])
            pred_label_h[int(k)] = np.argmax(marg.values)

        # Clean CPDs if needed
        bn.cpds = []

        return i, pred_label_01, pred_label_h

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
                

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

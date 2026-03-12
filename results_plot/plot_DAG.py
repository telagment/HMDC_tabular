
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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

import os
from pygobnilp.gobnilp import Gobnilp
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import glob

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

def save_dag_from_scores(score_path, output_path, pruning_score='BIC', palim=2, image_format='png', prog='dot'):
    """
    Learn a Bayesian Network from a local score file and save the DAG as an image.

    Args:
        score_path (str): Path to the Gobnilp local score file.
        output_path (str): File path to save the output DAG image.
        palim (int): Parent limit (maximum number of parents per node).
        image_format (str): Image format for output (e.g., 'png', 'pdf').
        prog (str): Layout program to use with Graphviz (e.g., 'dot', 'neato').
    """
    # Read scores and learn structure
    local_scores = read_local_scores(score_path)
    if pruning_score == 'AIC':
        str_score = 'DiscreteAIC'
    else:            
        str_score = 'DiscreteBIC'
    m = Gobnilp()
    m.learn(local_scores_source=local_scores, score=str_score, palim=palim)
    bn = m.learned_bn

    # Convert to AGraph and save
    A = to_agraph(bn)
    A.draw(output_path, format=image_format, prog=prog)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Mixed Data Experiments with Logistic Regression')
    parser.add_argument('-dataset', choices=DATASETS, default='Adult', type=str, help='Dataset')
    parser.add_argument('--baselearner', choices=BASELEARNERS, default='lr', type=str, help='Base Learner')
    parser.add_argument('--palim', type=int, default=2, help='The maximum number of parents for each node')
    # parser.add_argument('--dis-missing', type=float, default=0.5, help='The percentage of missing discrete features ')
    # parser.add_argument('--class-missing', type=float, default=0.5, help='The percentage of missing class variables')
    parser.add_argument('--pruning-score', type=str, choices=['BIC', 'AIC'], default='BIC', help='Pruning score for learning')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, help='Whether or not to plot the BN structure')
    parser.add_argument('--n-folds', type=int, default=10, help='Number of folds')
    parser.add_argument('--output', type=str, default='experiments_tabular/HMDC_prediction_Images', help='Output path')
    args = parser.parse_args()

    dataset = args.dataset
    base = args.baselearner
    palim = args.palim
    is_plot = args.plot
    n_folds = args.n_folds
    output_path = args.output
    pruning_score = args.pruning_score
    
    dis_missing_list = [0.3, 0.8]
    class_missing_list = [0.3, 0.7, 0.8, 0.9]

    base_dir = os.path.join(output_path, base)
    os.makedirs(base_dir, exist_ok=True)
    saving_dir = os.path.join(base_dir, dataset)
    os.makedirs(saving_dir, exist_ok=True)

    result_dir = 'experiments_tabular/HMDC_inference/' + str(base) + '/' + str(dataset)
    
    for dis_missing in dis_missing_list:
        for class_missing in class_missing_list:
            name = str(class_missing) + '_' + str(dis_missing)
            score_dir = os.path.join(result_dir, name, "score")
            save_dag_from_scores(score_dir, os.path.join(saving_dir, str(base) + '_' + str(dataset) + '_' + name + '.png'))
    
    print("Finished saving all DAG images")



        

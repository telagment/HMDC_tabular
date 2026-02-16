# Hybrid Multi-Dimensional Classification
Hybrid Multi-Dimensional Classification with mixed data for making inference in the presence of unobserved discrete features and unobserved class variable values at prediction time.

This project includes/adapts portions of code from ... . We sincerely thank the authors for their contribution.

## Requirements

The required packages are listed in `requirements.txt`.

## Data

Please check the experiment section of our paper for a collection of datasets used in the experiments
Link Data: https://palm.seu.edu.cn/zhangml/

## Usage

To run experiments on mixed data sets with missing value at prediction time, you can call
options:

  -dataset                        Dataset - 'Adult','Default','Thyroid'
  --baselearner                   Base classifier {lr,nb,rf}
  --palim PALIM                   The maximum number of parents for each node
  --dis-missing DIS_MISS          The percentage of missing discrete features [0,1]
  --class-missing CLASS_MISS      The percentage of missing class variables [0,1]
  --plot, --no-plot               Whether or not to plot the BN structure
  --n-folds N_FOLDS               Number of folds
  --output OUTPUT                 Output path
```

```shell
- Navigate to code folder
cd HMDC_tabular
```

```shell
- Training HDMC
python HDMC_training.py -dataset Adult --palim 2  --baselearner lr --n-folds 10
```

```shell
- Inference HDMC
python HDMC_Inference.py -dataset Adult --palim 2  --baselearner lr --n-folds 10
```

```shell
- Prediction HDMC: hamming and subset 0/1 Scores
python HDMC_prediction.py -dataset Adult --palim 2 --baselearner lr --n-folds 10
```

```shell
- Prediction HDMC: Balanced Accuracy (BA) 
python HDMC_prediction_BA.py -dataset Adult --palim 2 --baselearner lr --n-folds 10
```

```shell
- For the Thyroid dataset – comment out the HMDC optimistic method due to storage constraints.

```

```shell
- Visualize Score
usage: main.py [-dataset {Adult,'Default','Thyroid'}] [--base {lr,nb,rf}] [--palim PALIM] [--n-folds N_FOLDS] [--output OUTPUT] [--class-missing CLASS_MISS] [--dis-missing DIS_MISS]

## Folder Structure
project-name/
├── MDC_data/
│   
├── experiments_tabular/
├── results_plot/
├── README.md
├── ...
└── requirements.txt



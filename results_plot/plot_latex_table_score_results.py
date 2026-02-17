import os
import sys
import argparse
import pandas as pd
import numpy as np

np.random.seed(42)

DATASETS = ['Adult', 'Default', 'Thyroid']
BASELEARNERS = ['lr', 'nb', 'rf', 'svm']


def parse_array(x):
    """Parse array-like strings into Python lists of floats."""
    if isinstance(x, str):
        x = x.strip("[]")
        return [float(i) for i in x.split()] if x else []
    return x


def format_val(mean, std):
    """Format a mean ± std value."""
    return f"{mean:.2f} ± {std:.2f}"

def build_latex_table(df_dict, dataset, base, palim):
    """
    Build LaTeX table in the exact format of your original example.
    df_dict[(dis_missing, class_missing)] = DataFrame of results
    """
    cat_map = {
        "Random Classifier": "RandC",
        "Mode Imputation": "MI",
        "Binary Relevance": "BR",
        "hmdc Imputation": "HMDC-MI",
        "hmdc Optimistic": "HMDC-OP",
        "hmdc Average": "HMDC-AV",
        "hmdc Reference": "RefC",
    }
    # Metric name mapping
    metric_map = {
        "Hamming": "Hamming Score",
        "0/1 Score": "Subset 0/1 Score"
    }

    ref_df = next(iter(df_dict.values()))
    categories = ref_df["Category"].unique()
    metrics = ref_df["Metric"].unique()

    latex = []
    caption = f"{dataset} - {base.upper()} - Palim = {palim}"
    label = f"tab:detailed results {dataset} - {base.upper()}"
    latex.append(f"% {dataset} - {base.upper()}")
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\scriptsize")
    latex.append(f"\\caption{{{caption}}}\\label{{{label}}}")
    latex.append("\\vspace{1.5em}")
    latex.append("\\resizebox{\\textwidth}{!}{")
    latex.append("\\begin{tabular}{l| l || l l l l || l l l l}")
    latex.append("\\hline")
    latex.append(
        "\\multirow{2}{*}{Category} & \\multirow{2}{*}{Metric} "
        "& \\multicolumn{4}{c||}{Missing rate on discrete features = 80\\%} "
        "& \\multicolumn{4}{c}{Missing rate on discrete features = 30\\%} \\rule[-1ex]{0pt}{3.5ex} \\\\"
    )
    latex.append("\\cline{3-6} \\cline{7-10}")
    latex.append("& & 30\\% & 70\\% & 80\\% & 90\\% & 30\\% & 70\\% & 80\\% & 90\\% \\\\")
    latex.append("\\hline")

    for cat in categories:
        cat_label = cat_map.get(cat, cat)
        latex.append(f"\\multirow{{2}}{{*}}{{{cat_label}}} ")
        metrics_list = list(metrics)

        for metric in metrics:
            metric_label = metric_map.get(metric, metric)
            row = []
            row.append(f"& {metric_label} ")

            # Order: discrete 0.8 (left block), then discrete 0.3 (right block)
            for dis in [0.8, 0.3]:
                for cls in [0.3, 0.7, 0.8, 0.9]:
                    df = df_dict.get((dis, cls))
                    if df is not None:
                        vals = df[(df["Category"] == cat) & (df["Metric"] == metric)]
                        if not vals.empty:
                            mean = vals["Mean (%)"].values[0]
                            std = vals["Std Dev (%)"].values[0]
                            row.append("& " + format_val(mean, std) + " ")
                        else:
                            row.append("& -- ")
                    else:
                        row.append("& -- ")
            row.append("\\rule[-1ex]{0pt}{3.5ex}\\\\")
            latex.append("".join(row))
        latex.append("\\hline")

    latex.append("\\end{tabular}")
    latex.append("}")
    latex.append("\\end{table}")
    return "\n".join(latex)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate LaTeX table for Mixed Data Experiments')
    parser.add_argument('-dataset', choices=DATASETS, default='Adult', type=str)
    parser.add_argument('--baselearner', choices=BASELEARNERS, default='lr', type=str)
    parser.add_argument('--palim', type=int, default=2)
    parser.add_argument('--output', type=str, default='experiments_tabular/HMDC_score_latex')
    # parser.add_argument('--loss', choices=['hamming', 'subset'], default='hamming', type=str)
    args = parser.parse_args()

    dataset = args.dataset
    base = args.baselearner
    palim = args.palim
    output_path = args.output
    # loss = args.loss

    # Missingness configurations
    dis_missing_list = [0.3, 0.8]      # discrete features missing percentages
    class_missing_list = [0.3, 0.7, 0.8, 0.9]  # class variable missing percentages

    result_dir = f'experiments_tabular/HMDC_prediction/{base}/{dataset}'

    df_dict = {}
    for dis_missing in dis_missing_list:
        for class_missing in class_missing_list:
            name = f"{class_missing}_{dis_missing}"
            csv_filename = os.path.join(result_dir, name, "results.csv")
            if not os.path.exists(csv_filename):
                print(f"CSV not found: {csv_filename}")
                continue
            df = pd.read_csv(csv_filename)
            df_dict[(dis_missing, class_missing)] = df

    if not df_dict:
        print("No CSVs found. Cannot generate LaTeX table.")
        sys.exit(1)

    latex_code = build_latex_table(df_dict, dataset, base, palim)

    os.makedirs(output_path, exist_ok=True)
    tex_file = os.path.join(output_path, f"{dataset}_{base}.tex")
    with open(tex_file, "w") as f:
        f.write(latex_code)

    print(f"LaTeX table saved to {tex_file}")





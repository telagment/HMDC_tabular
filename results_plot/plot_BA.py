import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
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


def parse_array(x):
    """Parse array-like strings into Python lists of floats"""
    if isinstance(x, str):
        x = x.strip("[]")  # remove brackets
        return [float(i) for i in x.split()] if x else []
    return x

def _safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")

def df_to_png_barplot(df, dataset, base, name, metric, figure_dir, palim, ylim=None, dpi=200):
    """
    Save expanded DataFrame as a PNG grouped bar chart.

    df format:
      - first column: 'Category'
      - remaining columns: label series (e.g. Y1, Y2, ...)
    """
    cat_map = {
        "Random Classifier": "Ran",
        "Binary Relevance": "BR",
        "Mode Imputation": "MI",
        "hmdc Imputation": "HMDC-MI",
        "hmdc Optimal": "HMDC-OP",
        "hmdc Average": "HMDC-AV",
        "hmdc Reference": "RefC",
    }

    # categories on x-axis (methods)
    methods_full = df["Category"].tolist()
    methods = [cat_map.get(m, m[:3]) for m in methods_full]

    label_names = list(df.columns[1:])
    n_methods = len(methods)
    n_series = len(label_names)

    # x positions for grouped bars
    x = np.arange(n_methods)
    group_width = 0.8
    bar_w = group_width / max(n_series, 1)

    fig_w = max(6.0, 0.9 * n_methods)  # scale a bit with number of methods
    fig_h = 3.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    for j, label in enumerate(label_names):
        # center the group around x
        offset = (j - (n_series - 1) / 2) * bar_w
        y = df[label].to_numpy(dtype=float)
        ax.bar(x + offset, y, width=bar_w, label=f"${label}$")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)

    if ylim is not None:
        ax.set_ylim(*ylim)

    # Caption-like title (same info you used in LaTeX)
    dis_missing, class_missing = name.split("_")
    caption = f"{float(class_missing)*100:.0f}% class and {float(dis_missing)*100:.0f}% dis. feat. missing"
    ax.set_title(caption)

    ax.legend(ncol=min(4, n_series), fontsize=9, frameon=False)

    os.makedirs(figure_dir, exist_ok=True)
    metric_tag = _safe_filename(metric)
    png_filename = os.path.join(
        figure_dir, f"{dataset}_{name}_{base}_palim{palim}_{metric_tag}.png"
    )
    fig.tight_layout()
    fig.savefig(png_filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"PNG figure saved as {png_filename}")

def df_to_latex_tikz(df, dataset, base, name, metric, figure_dir, palim):
    """
    Convert expanded DataFrame into LaTeX TikZ/pgfplots bar chart
    following the requested LaTeX style.
    """
    # Map methods to short labels
    cat_map = {
        "Mode Imputation": "MI",
        "Binary Relevance": "BR",
        "hmdc Imputation": "HMDC-MI",
        "hmdc Optimal": "HMDC-OP",
        "hmdc Average": "HMDC-AV",
        "hmdc Reference": "RefC",
    }

    label_names = df.columns[1:]  # skip 'Category'
    coords = {label: [] for label in label_names}

    # Build coordinates for each label
    for _, row in df.iterrows():
        method = row["Category"]
        short = cat_map.get(method, method[:3])
        for label in label_names:
            coords[label].append(f"({short},{row[label]:.2f})")

    # Start LaTeX figure
    tikz = []
    tikz.append(r"\begin{tikzpicture}")
    tikz.append(r"\begin{axis}[")
    tikz.append(r"    width=\linewidth,")
    tikz.append(r"    height=0.5\linewidth,")
    tikz.append(r"    ybar,")
    tikz.append(r"    bar width=6pt,")
    tikz.append(r"    symbolic x coords={BR, HMDC-MI, HMDC-OP, HMDC-AV, RefC},")
    tikz.append(r"    xtick=data,")
    tikz.append(r"    xticklabels={")
    tikz.append(r"        {\tiny BR},")
    tikz.append(r"        {\tiny \shortstack{HMDC-\\MI}},")
    tikz.append(r"        {\tiny \shortstack{HMDC-\\OP}},")
    tikz.append(r"        {\tiny \shortstack{HMDC-\\AV}},")
    tikz.append(r"        {\tiny RefC}")
    tikz.append(r"    },")
    tikz.append(r"    x tick label style={rotate=45, anchor=east},")
    tikz.append(r"    ymin=0, ymax=1,")
    tikz.append(r"    ymajorgrids=true,")
    tikz.append(r"    legend style={")
    tikz.append(r"        at={(0.5,1)},")
    tikz.append(r"        anchor=north east,")
    tikz.append(r"        legend columns=4,")
    tikz.append(r"        font=\scriptsize")
    tikz.append(r"    },")
    tikz.append(r"]")

    # colors = ["blue", "orange", "green", "red", "purple", "cyan", "yellow"]
    colors = ["blue", "orange", "green", "red"]


    # Add data series
    for i, label in enumerate(label_names):
        tikz.append(f"\\addplot+[fill={colors[i % len(colors)]}] coordinates {{")
        tikz.append("    " + " ".join(coords[label]))
        tikz.append("};")

    # Legend (formatted like example)
    legend_labels = [f"${label}$" for label in label_names]
    tikz.append(f"\\legend{{{','.join(legend_labels)}}}")
    tikz.append(r"\end{axis}")
    tikz.append(r"\end{tikzpicture}")
    tikz.append(r"\vspace{-1em}")

    # Caption format: “Adult dataset with 90% missing class variables and 30% missing features.”
    class_missing, dis_missing = name.split('_')
    class_missing_str = f"{float(class_missing)*100:.0f}\\%"
    dis_missing_str = f"{float(dis_missing)*100:.0f}\\%"
    tikz.append(
        f"\\caption{{{class_missing_str} class and {dis_missing_str} dis. feat. missing}}"
    )


    # Label
    tikz.append(f"\\label{{fig:acc_{dataset}_{name}_{base}_{palim}}}")

    # Save LaTeX file
    os.makedirs(figure_dir, exist_ok=True)
    tex_filename = os.path.join(figure_dir, f"{dataset}_{name}_{base}.tex")
    with open(tex_filename, "w") as f:
        f.write("\n".join(tikz))


    print(f"LaTeX TikZ figure saved as {tex_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mixed Data Experiments with Logistic Regression (LaTeX export)')
    parser.add_argument('-dataset', choices=DATASETS, default='Adult', type=str, help='Dataset')
    parser.add_argument('--baselearner', choices=BASELEARNERS, default='lr', type=str, help='Base Learner')
    parser.add_argument('--palim', type=int, default=2, help='The maximum number of parents for each node')
    parser.add_argument('--result_dir', type=str, default='experiments_tabular/HMDC_prediction/BA', help='Output path')
    parser.add_argument('--output', type=str, default='experiments_tabular/HMDC_prediction/', help='Output path')
    args = parser.parse_args()

    dataset = args.dataset
    base = args.baselearner
    output_path = args.output
    result_dir = args.result_dir

    # Missingness configurations
    dis_missing_list = [0.3, 0.8]
    class_missing_list = [0.3, 0.7, 0.8, 0.9]

    base_dir = os.path.join(output_path, base)
    os.makedirs(base_dir, exist_ok=True)
    saving_dir = os.path.join(base_dir, dataset)
    os.makedirs(saving_dir, exist_ok=True)

    # result_dir = f'experiments_tabular/HMDC_prediction/BA/{base}/{dataset}'

    for dis_missing in dis_missing_list:
        for class_missing in class_missing_list:
            name = f"{dis_missing}_{class_missing}"
            # name = f"{class_missing}_{dis_missing}"
            csv_filename = os.path.join(result_dir, name, f"{name}deltas_results.csv")

            if not os.path.exists(csv_filename):
                print(f"CSV not found: {csv_filename}")
                continue

            # Load CSV
            df = pd.read_csv(csv_filename)

            # Parse array columns
            for col in ["Deltas_max - Mean (%)", "Deltas_max - Std Dev (%)",
                        "Deltas_ave - Mean (%)", "Deltas_ave - Std Dev (%)",
                        "Acc_min - Mean (%)", "Acc_min - Std Dev (%)"]:
                if col in df.columns:
                    df[col] = df[col].apply(parse_array)

            # Choose metric
            metric = "Deltas_ave - Mean (%)"  # default (↑)
            # metric = "Acc_min - Mean (%)"  # default (↑)
            # metric = "Deltas_max - Mean (%)"  # alternative (↓)

            # Expand into columns
            no_labels = len(df[metric][0])
            label_names = [f"Y{i+1}" for i in range(no_labels)]
            expanded = pd.DataFrame(df[metric].tolist(), columns=label_names)
            df_expanded = pd.concat([df["Category"], expanded], axis=1)

            # Export LaTeX/TikZ
            # df_to_latex_tikz(df_expanded, dataset, base, name, metric, saving_dir, args.palim)
            df_to_png_barplot(df_expanded, dataset, base, name, metric, saving_dir, args.palim)



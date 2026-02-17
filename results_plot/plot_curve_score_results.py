import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

DATASETS = ['Adult', 'Default', 'Thyroid']
BASELEARNERS = ['lr', 'nb', 'rf', 'svm']


def format_val(mean, std):
    """Format a mean ± std value."""
    return f"{mean:.2f}±{std:.2f}"


def build_latex_curve(dataset, dis_missing, metric_name, ymin, ymax, results_dict):
    """
    Generate a LaTeX TikZ figure comparing methods across class-missing rates.

    Parameters
    ----------
    dataset : str
        Dataset name (for comment/title)
    dis_missing : float
        Discrete missing rate (e.g., 0.3)
    metric_name : str
        Metric name (e.g. 'Hamming' or 'Subset Accuracy')
    ymin, ymax : int
        Axis limits
    results_dict : dict
        Mapping of method name -> list of (class_missing_rate, score)
    """

    color_map = {
        "Mode Imputation": "red",
        "Binary Relevance": "black",
        "hmdc Imputation": "magenta",
        "hmdc Optimistic": "blue",
        "hmdc Average": "green!60!black",
        "hmdc Reference": "gray",
    }

    marker_map = {
        "Mode Imputation": "square*, thin",
        "Binary Relevance": "x, mark options={scale=1.5}, thin",
        "hmdc Imputation": "triangle*, thick",
        "hmdc Optimistic": "diamond*, thick",
        "hmdc Average": "pentagon*, thin",
        "hmdc Reference": "*, thin",
    }

    latex = []
    latex.append("\\begin{tikzpicture}")
    latex.append(f"    % {int(dis_missing * 100)} dis - {metric_name}")
    latex.append("    \\begin{axis}[")
    latex.append("        scale=0.45,")
    latex.append("        symbolic x coords={30,70,80,90},")
    latex.append("        xtick=data,")
    latex.append(f"        ymin={ymin}, ymax={ymax},")
    latex.append("        ytick={20,30,...,100},")
    latex.append("        enlarge x limits=+0.01,")
    latex.append("        ymajorgrids=true,")
    latex.append("        bar width=0.3cm,")
    latex.append("        xmajorgrids=true,")
    latex.append("    ]")

    for method, coords in results_dict.items():
        color = color_map.get(method, method)
        marker = marker_map.get(method, method)
        coord_str = " ".join([f"({x}, {y:.5f})" for x, y in coords])
        latex.append(f"        % {method}")
        latex.append(f"        \\addplot[mark={marker}, {color}, thin] coordinates {{")
        latex.append(f"            {coord_str}")
        latex.append("        };")
        latex.append("")

    latex.append("    \\end{axis}")
    latex.append("\\end{tikzpicture}")
    return "\n".join(latex)

def draw_png_curve(dataset, dis_missing, metric_name, ymin, ymax, results_dict, png_path):
    """
    Draw and save a PNG curve plot comparing methods across class-missing rates.

    Parameters
    ----------
    dataset : str
    dis_missing : float
    metric_name : str
    ymin, ymax : float
    results_dict : dict[str, list[tuple[int, float]]]
        method -> list of (class_missing_rate_percent, score)
    png_path : str
        Output file path ending with .png
    """
    # Matplotlib-friendly styling
    color_map = {
        "Mode Imputation": "red",
        "Binary Relevance": "black",
        "hmdc Imputation": "magenta",
        "hmdc Optimistic": "blue",
        "hmdc Average": "green",
        "hmdc Reference": "gray",
    }
    marker_map = {
        "Mode Imputation": "s",      # square
        "Binary Relevance": "x",     # x
        "hmdc Imputation": "^", # triangle up
        "hmdc Optimistic": "D",      # diamond
        "hmdc Average": "p",         # pentagon
        "hmdc Reference": "o",       # circle
    }
    lw_map = {
        "Mode Imputation": 1.0,
        "Binary Relevance": 1.0,
        "hmdc Imputation": 2.0,
        "hmdc Optimistic": 2.0,
        "hmdc Average": 1.2,
        "hmdc Reference": 1.0,
    }

    # Collect and sort x values (percent labels like 30,70,80,90)
    all_x = sorted({x for coords in results_dict.values() for (x, _) in coords})
    if not all_x:
        print(f"[PNG] No points to plot for dis_missing={dis_missing}")
        return

    fig, ax = plt.subplots(figsize=(6, 4))

    for method, coords in results_dict.items():
        # Sort by x to ensure lines connect correctly
        coords_sorted = sorted(coords, key=lambda t: t[0])
        xs = [x for x, _ in coords_sorted]
        ys = [y for _, y in coords_sorted]

        ax.plot(
            xs, ys,
            label=method,
            color=color_map.get(method, "black"),
            marker=marker_map.get(method, "o"),
            linewidth=lw_map.get(method, 1.2),
            markersize=5,
        )

    ax.set_ylim(ymin, ymax)
    ax.set_xticks(all_x)
    ax.set_xlabel("Class-missing rate (%)")
    ax.set_ylabel(metric_name)
    ax.grid(True, which="major", axis="both")
    ax.set_title(f"{dataset} — {int(dis_missing * 100)}% discrete missing — {metric_name}")

    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

# -------------------- Main --------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate LaTeX table and TikZ plots for Mixed Data Experiments')
    parser.add_argument('-dataset', choices=DATASETS, default='Adult', type=str)
    parser.add_argument('--baselearner', choices=BASELEARNERS, default='lr', type=str)
    parser.add_argument('--palim', type=int, default=2)
    parser.add_argument('--output', type=str, default='experiments_tabular/HMDC_curve_score_plots')
    parser.add_argument('--loss', choices=['hamming', 'subset'], default='hamming', type=str)
    args = parser.parse_args()

    dataset = args.dataset
    base = args.baselearner
    palim = args.palim
    output_path = args.output
    loss = args.loss

    # Missingness configurations
    dis_missing_list = [0.3, 0.8]
    class_missing_list = [0.3, 0.7, 0.8, 0.9]

    result_dir = f'experiments_tabular/HMDC_prediction/{base}/{dataset}'

    os.makedirs(output_path, exist_ok=True)

    # Build TikZ plot for each discrete missing rate
    for dis_missing in dis_missing_list:
        results_dict = {}

        for class_missing in class_missing_list:
            name = f"{class_missing}_{dis_missing}"
            csv_filename = os.path.join(result_dir, name, f"results_{loss}.csv")
            if not os.path.exists(csv_filename):
                print(f"CSV not found: {csv_filename}")
                continue

            df = pd.read_csv(csv_filename)
            for _, row in df.iterrows():
                cat = row['Category']
                mean_val = row['Mean (%)']
                if cat not in results_dict:
                    results_dict[cat] = []
                results_dict[cat].append((int(class_missing * 100), mean_val))

        if not results_dict:
            print(f"No data found for discrete missing rate = {dis_missing}")
            continue

        png_filename = os.path.join(output_path, f"{dataset}_{base}_{int(dis_missing * 100)}dis_{loss}.png")
        draw_png_curve(dataset, dis_missing, loss.capitalize(), 0, 100, results_dict, png_filename)
        print(f"PNG plot saved to {png_filename}")

        # latex_code = build_latex_curve(dataset, dis_missing, loss.capitalize(), 0, 100, results_dict)

        # tex_filename = os.path.join(output_path, f"{dataset}_{base}_{int(dis_missing * 100)}dis_{loss}.tex")
        # with open(tex_filename, "w") as f:
        #     f.write(latex_code)

        # print(f"LaTeX TikZ plot saved to {tex_filename}")

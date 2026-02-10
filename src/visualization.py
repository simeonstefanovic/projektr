import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd
import numpy as np
import os
from scipy import stats

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

FIGSIZE_SINGLE = (10, 6)
FIGSIZE_WIDE = (14, 6)
FIGSIZE_TALL = (10, 10)
FIGSIZE_GRID = (16, 12)
DPI = 150

COLORS = {
    "MA1": "#2ecc71",
    "MA2": "#3498db",
    "passed": "#27ae60",
    "failed": "#e74c3c",
}


def get_exam_labels_by_position(n_exams, course):
    """
    Get exam labels based on position and course type.

    MA1 order (winter semester course):
    1. Kontinuirana nastava
    2. Zimski rok
    3. Ljetni rok
    4. Jesenski rok
    5+. Dekanski rok 1, 2, ...

    MA2 order (summer semester course):
    1. Kontinuirana nastava
    2. Ljetni rok
    3. Jesenski rok
    4+. Dekanski rok 1, 2, ...
    """
    labels = []

    if course == "MA1":
        base_labels = ["Kont.", "Zimski rok", "Ljetni rok", "Jesenski rok"]
        for i in range(n_exams):
            if i < len(base_labels):
                labels.append(base_labels[i])
            else:
                dekanski_num = i - len(base_labels) + 1
                labels.append(f"Dekanski rok {dekanski_num}")
    else:  # MA2
        base_labels = ["Kont.", "Ljetni rok", "Jesenski rok"]
        for i in range(n_exams):
            if i < len(base_labels):
                labels.append(base_labels[i])
            else:
                dekanski_num = i - len(base_labels) + 1
                labels.append(f"Dekanski rok {dekanski_num}")

    return labels


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_figure(fig, filename, output_dir):
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  - {filename}... saved")


def plot_pass_rate_by_year(stats, output_dir):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    years = []
    for course in ["MA1", "MA2"]:
        years = sorted(stats["single_course"][course].keys())
        rates = [stats["single_course"][course][y]["pass_rate"] * 100 for y in years]
        ax.plot(
            years,
            rates,
            marker="o",
            linewidth=2,
            label=course,
            color=COLORS[course],
            markersize=8,
        )

    ax.set_xlabel("Akademska godina")
    ax.set_ylabel("Prolaznost (%)")
    ax.set_title("Ukupna prolaznost MA1 i MA2 po godinama")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(years)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    save_figure(fig, "pass_rate_trend.png", output_dir)


def plot_enrollment_trend(stats, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    for course in ["MA1", "MA2"]:
        years = sorted(stats["single_course"][course].keys())
        totals = [stats["single_course"][course][y]["total_students"] for y in years]
        passed = [stats["single_course"][course][y]["passed_students"] for y in years]
        failed = [stats["single_course"][course][y]["failed_students"] for y in years]

        ax = axes[0] if course == "MA1" else axes[1]

        ax.bar(years, passed, label="Položili", color=COLORS["passed"])
        ax.bar(years, failed, bottom=passed, label="Pali", color=COLORS["failed"])

        ax.set_xlabel("Akademska godina")
        ax.set_ylabel("Broj studenata")
        ax.set_title(f"{course} - Broj studenata po godinama")
        ax.legend()
        ax.set_xticks(years)

    plt.tight_layout()
    save_figure(fig, "enrollment_passed_failed.png", output_dir)


def plot_grade_distribution_combined(processed, output_dir):
    for course in ["MA1", "MA2"]:
        years = sorted(processed[course].keys())
        n_years = len(years)

        fig, axes = plt.subplots(
            2, (n_years + 1) // 2, figsize=(4 * ((n_years + 1) // 2), 8)
        )
        axes = axes.flatten()

        i = 0
        for i, year in enumerate(years):
            df = processed[course][year]
            passed_df = df[df["passed"]]
            grades = passed_df["final_grade"].value_counts().sort_index()

            ax = axes[i]
            bars = ax.bar(
                [2, 3, 4, 5],
                [
                    grades.get(2, 0),
                    grades.get(3, 0),
                    grades.get(4, 0),
                    grades.get(5, 0),
                ],
                color=COLORS[course],
                edgecolor="black",
            )
            ax.set_xlabel("Ocjena")
            ax.set_ylabel("Broj studenata")
            ax.set_title(f"{year}")
            ax.set_xticks([2, 3, 4, 5])
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            axes[j].set_visible(False)

        fig.suptitle(
            f"{course} - Distribucija ocjena po godinama",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        save_figure(fig, f"grade_distribution_{course}_all.png", output_dir)


def plot_points_by_exam_period(processed, output_dir):
    from src.processing import get_exam_columns

    for course in ["MA1", "MA2"]:
        for year, df in processed[course].items():
            exams = get_exam_columns(df)
            n_exams = len(exams)
            labels = get_exam_labels_by_position(n_exams, course)

            n_cols = min(3, n_exams)
            n_rows = (n_exams + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
            if n_exams == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            axes_flat = axes.flatten()

            for i, (name, points_col, prolaz_col, time_col) in enumerate(exams):
                ax = axes_flat[i]

                attempted = df[df[points_col] > 0][points_col]
                passed = (
                    df[df[prolaz_col]][points_col]
                    if prolaz_col in df.columns
                    else pd.Series()
                )
                failed = (
                    df[(df[points_col] > 0) & (~df[prolaz_col])][points_col]
                    if prolaz_col in df.columns
                    else pd.Series()
                )

                if len(failed) > 0:
                    ax.hist(
                        failed,
                        bins=15,
                        alpha=0.7,
                        label="Pali",
                        color=COLORS["failed"],
                        edgecolor="black",
                    )
                if len(passed) > 0:
                    ax.hist(
                        passed,
                        bins=15,
                        alpha=0.7,
                        label="Prošli",
                        color=COLORS["passed"],
                        edgecolor="black",
                    )

                ax.set_xlabel("Bodovi")
                ax.set_ylabel("Broj studenata")
                ax.set_title(labels[i], fontsize=10)
                ax.legend(fontsize=8)
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            for j in range(n_exams, len(axes_flat)):
                axes_flat[j].set_visible(False)

            fig.suptitle(
                f"{course} {year} - Distribucija bodova po roku",
                fontsize=12,
                fontweight="bold",
            )
            plt.tight_layout()
            save_figure(fig, f"points_by_exam_{course}_{year}.png", output_dir)


def plot_pass_rate_by_exam_period(stats, output_dir):
    for course in ["MA1", "MA2"]:
        years = sorted(stats["pass_by_exam"][course].keys())

        year_groups = []
        for i in range(0, len(years), 4):
            year_groups.append(years[i : i + 4])

        n_groups = len(year_groups)
        fig, axes = plt.subplots(n_groups, 2, figsize=(14, 6 * n_groups))
        if n_groups == 1:
            axes = axes.reshape(1, -1)

        all_rates = []
        all_cum_rates = []
        for year in years:
            exam_data = stats["pass_by_exam"][course][year]
            for e in exam_data.keys():
                all_rates.append(exam_data[e]["rate"] * 100)
                all_cum_rates.append(exam_data[e]["cumulative_rate"] * 100)

        rate_min = max(0, min(all_rates) - 5) if all_rates else 0
        rate_max = min(100, max(all_rates) + 5) if all_rates else 100
        cum_min = max(0, min(all_cum_rates) - 5) if all_cum_rates else 0
        cum_max = min(100, max(all_cum_rates) + 5) if all_cum_rates else 100

        for group_idx, year_group in enumerate(year_groups):
            ref_year = year_group[0]
            n_exams = len(stats["pass_by_exam"][course][ref_year])
            labels = get_exam_labels_by_position(n_exams, course)

            ax1 = axes[group_idx, 0]
            for year in year_group:
                exam_data = stats["pass_by_exam"][course][year]
                exams = list(exam_data.keys())
                rates = [exam_data[e]["rate"] * 100 for e in exams]
                ax1.plot(
                    range(len(exams)),
                    rates,
                    marker="o",
                    label=str(year),
                    linewidth=2,
                    markersize=8,
                )

            ax1.set_xlabel("Rok")
            ax1.set_ylabel("Prolaznost na roku (%)")
            year_range = (
                f"{year_group[0]}-{year_group[-1]}"
                if len(year_group) > 1
                else str(year_group[0])
            )
            ax1.set_title(f"{course} ({year_range}) - Prolaznost po pojedinom roku")
            ax1.set_xticks(range(len(labels)))
            ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            ax1.legend(fontsize=9)
            ax1.set_ylim(rate_min, rate_max)
            ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax1.grid(True, alpha=0.3)

            ax2 = axes[group_idx, 1]
            for year in year_group:
                exam_data = stats["pass_by_exam"][course][year]
                exams = list(exam_data.keys())
                cum_rates = [exam_data[e]["cumulative_rate"] * 100 for e in exams]
                ax2.plot(
                    range(len(exams)),
                    cum_rates,
                    marker="s",
                    label=str(year),
                    linewidth=2,
                    markersize=8,
                )

            ax2.set_xlabel("Rok")
            ax2.set_ylabel("Kumulativna prolaznost (%)")
            ax2.set_title(f"{course} ({year_range}) - Kumulativna prolaznost")
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            ax2.legend(fontsize=9)
            ax2.set_ylim(cum_min, cum_max)
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_figure(fig, f"pass_rate_by_exam_{course}.png", output_dir)


def plot_attempts_distribution(stats, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_GRID)

    for idx, course in enumerate(["MA1", "MA2"]):
        all_passed = {}
        all_failed = {}

        for year, dist in stats["attempts_dist"][course].items():
            for k, v in dist.items():
                all_passed[k] = all_passed.get(k, 0) + v

        for year, dist in stats["failed_attempts_dist"][course].items():
            for k, v in dist.items():
                all_failed[k] = all_failed.get(k, 0) + v

        ax_passed = axes[0, idx]
        if all_passed:
            keys: list[int | str] = sorted(
                [k for k in all_passed.keys() if isinstance(k, int)]
            )
            if "5+" in all_passed:
                keys.append("5+")
            values = [all_passed[k] for k in keys]

            ax_passed.bar(
                range(len(keys)), values, color=COLORS["passed"], edgecolor="black"
            )
            ax_passed.set_xticks(range(len(keys)))
            ax_passed.set_xticklabels([str(k) for k in keys])
        ax_passed.set_xlabel("Broj pokušaja")
        ax_passed.set_ylabel("Broj studenata")
        ax_passed.set_title(f"{course} - Položili: pokušaji do prolaska")
        ax_passed.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax_failed = axes[1, idx]
        if all_failed:
            keys_failed: list[int | str] = sorted(
                [k for k in all_failed.keys() if isinstance(k, int)]
            )
            if "5+" in all_failed:
                keys_failed.append("5+")
            values = [all_failed[k] for k in keys_failed]

            ax_failed.bar(
                range(len(keys_failed)),
                values,
                color=COLORS["failed"],
                edgecolor="black",
            )
            ax_failed.set_xticks(range(len(keys_failed)))
            ax_failed.set_xticklabels([str(k) for k in keys_failed])
        ax_failed.set_xlabel("Broj izlazaka")
        ax_failed.set_ylabel("Broj studenata")
        ax_failed.set_title(f"{course} - Pali: broj izlazaka")
        ax_failed.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    save_figure(fig, "attempts_distribution_all.png", output_dir)


def plot_grade_heatmap_combined(stats, output_dir):
    years = sorted(stats["grade_matrix"].keys())
    n_years = len(years)

    cols = min(4, n_years)
    rows = (n_years + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for i, year in enumerate(years):
        row, col = i // cols, i % cols
        ax = axes[row, col]

        matrix = stats["grade_matrix"][year]
        sns.heatmap(
            matrix, annot=True, fmt="d", cmap="YlGnBu", ax=ax, cbar=False, square=True
        )
        ax.set_xlabel("MA2 Ocjena")
        ax.set_ylabel("MA1 Ocjena")
        ax.set_title(f"{year}")

    for i in range(len(years), rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].set_visible(False)

    fig.suptitle(
        "Matrica ocjena MA1 vs MA2 po godinama", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    save_figure(fig, "grade_matrix_all.png", output_dir)


def plot_scatter_points_combined(merged, stats, output_dir):
    years = sorted(merged.keys())
    n_years = len(years)

    cols = min(4, n_years)
    rows = (n_years + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for i, year in enumerate(years):
        row, col = i // cols, i % cols
        ax = axes[row, col]

        df = merged[year]
        both = df[df["both_passed"]].copy()
        both = both.dropna(subset=["ma1_points", "ma2_points"])

        if len(both) < 2:
            ax.text(
                0.5,
                0.5,
                "Nedovoljno podataka",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{year}")
            continue

        ax.scatter(
            both["ma1_points"],
            both["ma2_points"],
            alpha=0.5,
            edgecolors="black",
            linewidths=0.3,
            s=30,
        )

        x = both["ma1_points"].values
        y = both["ma2_points"].values
        linreg = stats_module.linregress(x, y)
        slope: float = linreg[0]  # type: ignore
        intercept: float = linreg[1]  # type: ignore
        r_value: float = linreg[2]  # type: ignore
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "r-", linewidth=2, label=f"R²={r_value**2:.2f}")

        ax.set_xlabel("MA1 Bodovi")
        ax.set_ylabel("MA2 Bodovi")
        ax.set_title(f"{year}")
        ax.legend(fontsize=8)

    for i in range(len(years), rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].set_visible(False)

    fig.suptitle("Korelacija bodova MA1 vs MA2", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, "scatter_points_all.png", output_dir)


def plot_ma1_predicts_ma2(stats, output_dir):
    years = sorted(stats["ma1_predicts_ma2"].keys())

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    ax1 = axes[0]
    for year in years:
        pred = stats["ma1_predicts_ma2"][year]
        if pred is None:
            continue
        grades = sorted(pred.keys())
        pass_rates = [pred[g]["ma2_pass_rate"] * 100 for g in grades]
        ax1.plot(grades, pass_rates, marker="o", label=str(year), linewidth=2)

    ax1.set_xlabel("Ocjena na MA1")
    ax1.set_ylabel("Vjerojatnost prolaska MA2 (%)")
    ax1.set_title("MA1 ocjena → MA2 prolaznost")
    ax1.set_xticks([2, 3, 4, 5])
    ax1.legend(fontsize=8)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = axes[1]
    for year in years:
        pred = stats["ma1_predicts_ma2"][year]
        if pred is None:
            continue
        grades = sorted(
            [g for g in pred.keys() if pred[g]["avg_ma2_grade"] is not None]
        )
        avg_grades = [pred[g]["avg_ma2_grade"] for g in grades]
        ax2.plot(grades, avg_grades, marker="s", label=str(year), linewidth=2)

    ax2.set_xlabel("Ocjena na MA1")
    ax2.set_ylabel("Prosječna ocjena na MA2")
    ax2.set_title("MA1 ocjena → MA2 prosječna ocjena")
    ax2.set_xticks([2, 3, 4, 5])
    ax2.set_yticks([2, 3, 4, 5])
    ax2.legend(fontsize=8)

    plt.tight_layout()
    save_figure(fig, "ma1_predicts_ma2.png", output_dir)


def plot_covid_comparison(stats, output_dir):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    covid = stats["covid_impact"]

    x = np.arange(3)
    width = 0.35

    ma1_rates = [
        (covid["MA1"]["pre_covid_pass_rate"] or 0) * 100,
        (covid["MA1"]["covid_pass_rate"] or 0) * 100,
        (covid["MA1"]["post_covid_pass_rate"] or 0) * 100,
    ]

    ma2_rates = [
        (covid["MA2"]["pre_covid_pass_rate"] or 0) * 100,
        (covid["MA2"]["covid_pass_rate"] or 0) * 100,
        (covid["MA2"]["post_covid_pass_rate"] or 0) * 100,
    ]

    ax.bar(x - width / 2, ma1_rates, width, label="MA1", color=COLORS["MA1"])
    ax.bar(x + width / 2, ma2_rates, width, label="MA2", color=COLORS["MA2"])

    ax.set_xlabel("Razdoblje")
    ax.set_ylabel("Prolaznost (%)")
    ax.set_title("Usporedba prolaznosti: Pre-COVID vs COVID vs Post-COVID")
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Pre-COVID\n(2018)", "COVID\n(2019-2020)", "Post-COVID\n(2021-2024)"]
    )
    ax.legend()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    save_figure(fig, "covid_comparison.png", output_dir)


def plot_ma2_before_ma1(stats, output_dir):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    years = sorted(stats["correlation"].keys())
    counts = [stats["correlation"][y]["ma2_before_ma1"] for y in years]

    ax.bar(years, counts, color="orange", edgecolor="black")
    ax.set_xlabel("Akademska godina")
    ax.set_ylabel("Broj studenata")
    ax.set_title("Studenti koji su položili MA2 prije MA1")
    ax.set_xticks(years)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    save_figure(fig, "ma2_before_ma1.png", output_dir)


def plot_most_common_pass_exam(stats, output_dir):
    """Show which exam period students most commonly pass on."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    for idx, course in enumerate(["MA1", "MA2"]):
        ax = axes[idx]

        # Aggregate by position (index) across all years
        all_pass_by_position = {}
        max_exams = 0

        for year, year_stats in stats["single_course"][course].items():
            exam_dict = year_stats.get("pass_by_exam", {})
            exam_list = list(exam_dict.items())
            max_exams = max(max_exams, len(exam_list))

            for pos, (exam_name, count) in enumerate(exam_list):
                all_pass_by_position[pos] = all_pass_by_position.get(pos, 0) + count

        if all_pass_by_position:
            labels = get_exam_labels_by_position(max_exams, course)
            positions = sorted(all_pass_by_position.keys())
            counts = [all_pass_by_position[p] for p in positions]
            display_labels = [
                labels[p] if p < len(labels) else f"Rok {p+1}" for p in positions
            ]

            bars = ax.bar(
                range(len(positions)), counts, color=COLORS[course], edgecolor="black"
            )
            ax.set_xticks(range(len(positions)))
            ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=9)

            max_idx = counts.index(max(counts))
            bars[max_idx].set_color("#e74c3c")
            bars[max_idx].set_edgecolor("black")

        ax.set_xlabel("Rok")
        ax.set_ylabel("Broj studenata koji su položili")
        ax.set_title(f"{course} - Na kojem roku studenti najčešće prolaze")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    save_figure(fig, "most_common_pass_exam.png", output_dir)


def plot_rejection_analysis(processed, stats, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_GRID)

    for idx, course in enumerate(["MA1", "MA2"]):
        ax = axes[0, idx]

        improved = 0
        worsened = 0
        same = 0

        for year, df in processed[course].items():
            rejected = df[df["rejected_grade"]]
            improved += (rejected["grade_change"] > 0).sum()
            worsened += (rejected["grade_change"] < 0).sum()
            same += (rejected["grade_change"] == 0).sum()

        values = [improved, same, worsened]
        labels = ["Poboljšana", "Ista", "Pogoršana"]
        colors_rej = ["#27ae60", "#f39c12", "#e74c3c"]

        ax.bar(labels, values, color=colors_rej, edgecolor="black")
        ax.set_xlabel("Promjena ocjene")
        ax.set_ylabel("Broj studenata")
        ax.set_title(f"{course} - Dinamika ocjena nakon odbijanja (ista godina)")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Cross-year rejections (students who passed exam but re-enrolled next year)
        ax2 = axes[1, idx]
        cross_year = stats.get("cross_year_rejections", {}).get(course, {})

        if cross_year:
            year_pairs = sorted(cross_year.keys())
            counts = [cross_year[yp]["count"] for yp in year_pairs]

            ax2.bar(range(len(year_pairs)), counts, color="#8e44ad", edgecolor="black")
            ax2.set_xticks(range(len(year_pairs)))
            ax2.set_xticklabels(year_pairs, rotation=45, ha="right", fontsize=9)

            total = sum(counts)
            ax2.annotate(
                f"Ukupno: {total}",
                xy=(0.98, 0.95),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        ax2.set_xlabel("Prijelaz godine")
        ax2.set_ylabel("Broj studenata")
        ax2.set_title(f"{course} - Prošli ispit, odbili, upisali sljedeću godinu")
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    save_figure(fig, "rejection_analysis.png", output_dir)


def plot_failed_analysis(processed, stats, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_GRID)

    for idx, course in enumerate(["MA1", "MA2"]):
        years = sorted(processed[course].keys())

        ax1 = axes[0, idx]
        never_tried = [
            stats["single_course"][course][y]["failed_never_tried"] for y in years
        ]
        tried_failed = [
            stats["single_course"][course][y]["failed_students_with_attempts"]
            for y in years
        ]

        ax1.bar(years, tried_failed, label="Izašli, ali pali", color="#e67e22")
        ax1.bar(
            years,
            never_tried,
            bottom=tried_failed,
            label="Nikad nisu izašli",
            color="#95a5a6",
        )
        ax1.set_xlabel("Akademska godina")
        ax1.set_ylabel("Broj studenata")
        ax1.set_title(f"{course} - Struktura padova")
        ax1.legend(fontsize=8)
        ax1.set_xticks(years)
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax2 = axes[1, idx]
        pass_thresholds = [
            stats["single_course"][course][y]["pass_threshold"] for y in years
        ]
        ax2.plot(years, pass_thresholds, marker="o", linewidth=2, color=COLORS[course])
        ax2.set_xlabel("Akademska godina")
        ax2.set_ylabel("Prag prolaznosti (bodovi)")
        ax2.set_title(f"{course} - Detektirani prag prolaznosti")
        ax2.set_xticks(years)
        ax2.set_ylim(35, 55)
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    save_figure(fig, "failed_analysis.png", output_dir)


def plot_correlation_trend(stats, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    years = sorted(stats["correlation"].keys())

    ax1 = axes[0]
    correlations = [
        stats["correlation"][y]["pearson_points"]
        for y in years
        if stats["correlation"][y]["pearson_points"]
    ]
    valid_years = [y for y in years if stats["correlation"][y]["pearson_points"]]

    if correlations:
        ax1.plot(valid_years, correlations, marker="o", linewidth=2, color="purple")
    ax1.set_xlabel("Akademska godina")
    ax1.set_ylabel("Pearsonov koeficijent korelacije")
    ax1.set_title("Korelacija bodova MA1 i MA2")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.set_xticks(valid_years)

    ax2 = axes[1]
    r_squared = [
        stats["correlation"][y]["r_squared"]
        for y in years
        if stats["correlation"][y]["r_squared"]
    ]

    if r_squared:
        ax2.plot(valid_years, r_squared, marker="s", linewidth=2, color="darkgreen")
    ax2.set_xlabel("Akademska godina")
    ax2.set_ylabel("R² (koeficijent determinacije)")
    ax2.set_title("Kvaliteta regresije MA1 → MA2")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    ax2.set_xticks(valid_years)

    plt.tight_layout()
    save_figure(fig, "correlation_trend.png", output_dir)


stats_module = stats


def plot_summary_dashboard(stats, output_dir):
    """Create a comprehensive summary dashboard with key statistics."""
    fig = plt.figure(figsize=(20, 16))

    # Overall layout: 3 rows x 4 columns
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    # 1. Pass rate comparison (top-left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, 0:2])
    years = sorted(stats["single_course"]["MA1"].keys())
    ma1_rates = [stats["single_course"]["MA1"][y]["pass_rate"] * 100 for y in years]
    ma2_rates = [stats["single_course"]["MA2"][y]["pass_rate"] * 100 for y in years]

    x = np.arange(len(years))
    width = 0.35
    ax1.bar(
        x - width / 2,
        ma1_rates,
        width,
        label="MA1",
        color=COLORS["MA1"],
        edgecolor="black",
    )
    ax1.bar(
        x + width / 2,
        ma2_rates,
        width,
        label="MA2",
        color=COLORS["MA2"],
        edgecolor="black",
    )
    ax1.axhline(
        y=np.mean(ma1_rates),
        color=COLORS["MA1"],
        linestyle="--",
        alpha=0.7,
        label=f"MA1 prosjek ({np.mean(ma1_rates):.1f}%)",
    )
    ax1.axhline(
        y=np.mean(ma2_rates),
        color=COLORS["MA2"],
        linestyle="--",
        alpha=0.7,
        label=f"MA2 prosjek ({np.mean(ma2_rates):.1f}%)",
    )
    ax1.set_xlabel("Godina")
    ax1.set_ylabel("Prolaznost (%)")
    ax1.set_title("Prolaznost po godinama", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 100)

    # 2. Grade distribution pie charts (top-right, 2 subplots)
    for idx, course in enumerate(["MA1", "MA2"]):
        ax = fig.add_subplot(gs[0, 2 + idx])
        grades = {2: 0, 3: 0, 4: 0, 5: 0}
        for year in stats["single_course"][course].keys():
            for grade, count in stats["single_course"][course][year][
                "grade_distribution"
            ].items():
                if grade in grades:
                    grades[grade] += count

        colors_pie = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
        labels = [
            f"2 ({grades[2]})",
            f"3 ({grades[3]})",
            f"4 ({grades[4]})",
            f"5 ({grades[5]})",
        ]
        ax.pie(
            grades.values(),
            labels=labels,
            colors=colors_pie,
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title(f"{course} - Distribucija ocjena", fontweight="bold")

    # 3. MA1 predicts MA2 (middle-left)
    ax3 = fig.add_subplot(gs[1, 0:2])
    agg_pred = {
        2: {"total": 0, "passed": 0},
        3: {"total": 0, "passed": 0},
        4: {"total": 0, "passed": 0},
        5: {"total": 0, "passed": 0},
    }
    for year in stats["ma1_predicts_ma2"].keys():
        pred = stats["ma1_predicts_ma2"][year]
        if pred:
            for grade in [2, 3, 4, 5]:
                if grade in pred:
                    agg_pred[grade]["total"] += pred[grade]["total"]
                    agg_pred[grade]["passed"] += pred[grade]["ma2_passed"]

    grades = [2, 3, 4, 5]
    rates = [
        (
            agg_pred[g]["passed"] / agg_pred[g]["total"] * 100
            if agg_pred[g]["total"] > 0
            else 0
        )
        for g in grades
    ]
    colors_bar = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
    bars = ax3.bar(grades, rates, color=colors_bar, edgecolor="black")
    ax3.set_xlabel("Ocjena iz MA1")
    ax3.set_ylabel("Prolaznost na MA2 (%)")
    ax3.set_title("Kako ocjena iz MA1 predviđa uspjeh na MA2", fontweight="bold")
    ax3.set_xticks(grades)
    ax3.set_ylim(0, 105)
    for bar, rate in zip(bars, rates):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # 4. COVID impact (middle-right)
    ax4 = fig.add_subplot(gs[1, 2:4])
    covid = stats["covid_impact"]
    periods = ["Pre-COVID\n(2018)", "COVID\n(2019-2020)", "Post-COVID\n(2021-2024)"]
    ma1_covid = [
        (covid["MA1"]["pre_covid_pass_rate"] or 0) * 100,
        (covid["MA1"]["covid_pass_rate"] or 0) * 100,
        (covid["MA1"]["post_covid_pass_rate"] or 0) * 100,
    ]
    ma2_covid = [
        (covid["MA2"]["pre_covid_pass_rate"] or 0) * 100,
        (covid["MA2"]["covid_pass_rate"] or 0) * 100,
        (covid["MA2"]["post_covid_pass_rate"] or 0) * 100,
    ]

    x = np.arange(len(periods))
    ax4.bar(
        x - width / 2,
        ma1_covid,
        width,
        label="MA1",
        color=COLORS["MA1"],
        edgecolor="black",
    )
    ax4.bar(
        x + width / 2,
        ma2_covid,
        width,
        label="MA2",
        color=COLORS["MA2"],
        edgecolor="black",
    )
    ax4.set_xlabel("Razdoblje")
    ax4.set_ylabel("Prolaznost (%)")
    ax4.set_title("Utjecaj COVID-19 na prolaznost", fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(periods)
    ax4.legend()
    ax4.set_ylim(0, 100)

    # 5. Correlation scatter (bottom-left, spans 2 columns)
    ax5 = fig.add_subplot(gs[2, 0:2])
    years_corr = sorted(stats["correlation"].keys())
    pearson_vals = [
        stats["correlation"][y]["pearson_points"]
        for y in years_corr
        if stats["correlation"][y]["pearson_points"]
    ]
    valid_years = [y for y in years_corr if stats["correlation"][y]["pearson_points"]]

    ax5.plot(
        valid_years,
        pearson_vals,
        marker="o",
        linewidth=2,
        markersize=10,
        color="purple",
    )
    ax5.fill_between(valid_years, pearson_vals, alpha=0.3, color="purple")
    ax5.axhline(
        y=np.mean(pearson_vals),
        color="red",
        linestyle="--",
        label=f"Prosjek: r={np.mean(pearson_vals):.3f}",
    )
    ax5.set_xlabel("Godina")
    ax5.set_ylabel("Pearsonov koeficijent korelacije")
    ax5.set_title("Korelacija bodova MA1 i MA2 kroz godine", fontweight="bold")
    ax5.set_ylim(0.4, 0.9)
    ax5.set_xticks(valid_years)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Key statistics summary (bottom-right)
    ax6 = fig.add_subplot(gs[2, 2:4])
    ax6.axis("off")

    # Calculate key stats
    total_ma1 = sum(
        stats["single_course"]["MA1"][y]["total_students"]
        for y in stats["single_course"]["MA1"]
    )
    total_ma2 = sum(
        stats["single_course"]["MA2"][y]["total_students"]
        for y in stats["single_course"]["MA2"]
    )
    passed_ma1 = sum(
        stats["single_course"]["MA1"][y]["passed_students"]
        for y in stats["single_course"]["MA1"]
    )
    passed_ma2 = sum(
        stats["single_course"]["MA2"][y]["passed_students"]
        for y in stats["single_course"]["MA2"]
    )

    grade_trans = stats.get("grade_transition", {})
    dropout = stats.get("dropout", {})
    perfect = stats.get("perfect_scores", {})
    stat_tests = stats.get("statistical_tests", {})

    summary_text = f"""
    KLJUČNE STATISTIKE (2018-2024)
    
    UKUPNI PODACI:
    • MA1: {total_ma1:,} studenata, {passed_ma1:,} položilo ({passed_ma1/total_ma1*100:.1f}%)
    • MA2: {total_ma2:,} studenata, {passed_ma2:,} položilo ({passed_ma2/total_ma2*100:.1f}%)
    
    STATISTIČKA ZNAČAJNOST:
    • MA2 je značajno teži (p < 0.001)
    • Razlika prolaznosti: {(passed_ma1/total_ma1 - passed_ma2/total_ma2)*100:.1f} postotnih bodova
    
    PRIJELAZ MA1 → MA2:
    • Poboljšali ocjenu: {grade_trans.get('improved', 0):,} ({grade_trans.get('improved_pct', 0):.1f}%)
    • Ista ocjena: {grade_trans.get('same', 0):,} ({grade_trans.get('same_pct', 0):.1f}%)
    • Pogoršali ocjenu: {grade_trans.get('dropped', 0):,} ({grade_trans.get('dropped_pct', 0):.1f}%)
    
    DROPOUT (nikad nisu izašli):
    • MA1: {dropout.get('MA1', {}).get('dropout_rate', 0):.1f}%
    • MA2: {dropout.get('MA2', {}).get('dropout_rate', 0):.1f}%
    
    SAVRŠENE OCJENE (100 bodova):
    • MA1: {perfect.get('MA1', 0)} studenata
    • MA2: {perfect.get('MA2', 0)} studenata
    """

    ax6.text(
        0.05,
        0.95,
        summary_text,
        transform=ax6.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    fig.suptitle(
        "SAŽETAK: Statistička analiza ispita MA1 i MA2 (2018-2024)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    save_figure(fig, "summary_dashboard.png", output_dir)


def plot_first_enrollment_pass_rate(stats, output_dir):
    """Plot first-enrollment continual pass rate vs all-students pass rate by year."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    fe = stats["first_enrollment"]

    for idx, course in enumerate(["MA1", "MA2"]):
        ax = axes[idx]
        years = sorted(fe[course].keys())

        # First-enrollment continual pass rate
        fe_cont_rates = [
            fe[course][y]["first_enrollment_continual_rate"] * 100 for y in years
        ]
        # First-enrollment overall pass rate (during that academic year)
        fe_overall_rates = [
            fe[course][y]["first_enrollment_pass_rate"] * 100 for y in years
        ]
        # All students continual pass rate (includes repeaters)
        all_cont_rates = [fe[course][y]["all_continual_rate"] * 100 for y in years]
        # All students overall pass rate
        all_overall_rates = [fe[course][y]["all_pass_rate"] * 100 for y in years]

        ax.plot(
            years,
            fe_cont_rates,
            marker="o",
            linewidth=2.5,
            markersize=8,
            label="Brucoši - kontinuirana",
            color="#2ecc71",
            linestyle="-",
        )
        ax.plot(
            years,
            fe_overall_rates,
            marker="s",
            linewidth=2.5,
            markersize=8,
            label="Brucoši - ukupna prolaznost",
            color="#27ae60",
            linestyle="--",
        )
        ax.plot(
            years,
            all_cont_rates,
            marker="^",
            linewidth=1.5,
            markersize=7,
            label="Svi - kontinuirana",
            color="#95a5a6",
            linestyle="-",
        )
        ax.plot(
            years,
            all_overall_rates,
            marker="d",
            linewidth=1.5,
            markersize=7,
            label="Svi - ukupna prolaznost",
            color="#7f8c8d",
            linestyle="--",
        )

        ax.set_xlabel("Akademska godina")
        ax.set_ylabel("Prolaznost (%)")
        ax.set_title(f"{course} - Prolaznost prvih upisa vs. svi studenti")
        ax.legend(fontsize=8, loc="best")
        ax.set_xticks(years)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    save_figure(fig, "first_enrollment_pass_rate.png", output_dir)


def plot_first_enrollment_detail(stats, output_dir):
    """Detailed bar chart: for each year, show first-enrollment numbers and rates."""
    fe = stats["first_enrollment"]

    for course in ["MA1", "MA2"]:
        years = sorted(fe[course].keys())
        n = len(years)

        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

        # Left: stacked bar - total first-enrollment students breakdown
        ax1 = axes[0]
        cont_pass = [fe[course][y]["passed_continual_first_enrollment"] for y in years]
        other_pass = [
            fe[course][y]["passed_first_enrollment"]
            - fe[course][y]["passed_continual_first_enrollment"]
            for y in years
        ]
        failed = [
            fe[course][y]["total_first_enrollment"]
            - fe[course][y]["passed_first_enrollment"]
            for y in years
        ]

        x = np.arange(n)
        ax1.bar(
            x,
            cont_pass,
            label="Položili kontinuirano",
            color="#2ecc71",
            edgecolor="black",
        )
        ax1.bar(
            x,
            other_pass,
            bottom=cont_pass,
            label="Položili na roku",
            color="#3498db",
            edgecolor="black",
        )
        bottoms = [c + o for c, o in zip(cont_pass, other_pass)]
        ax1.bar(
            x,
            failed,
            bottom=bottoms,
            label="Nisu položili",
            color="#e74c3c",
            edgecolor="black",
        )

        ax1.set_xlabel("Akademska godina")
        ax1.set_ylabel("Broj studenata")
        ax1.set_title(f"{course} - Ishod prvog upisa (samo brucoši)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(years)
        ax1.legend(fontsize=9)
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Right: percentage view
        ax2 = axes[1]
        totals = [fe[course][y]["total_first_enrollment"] for y in years]
        cont_pct = [c / t * 100 if t > 0 else 0 for c, t in zip(cont_pass, totals)]
        other_pct = [o / t * 100 if t > 0 else 0 for o, t in zip(other_pass, totals)]
        failed_pct = [f / t * 100 if t > 0 else 0 for f, t in zip(failed, totals)]

        ax2.bar(x, cont_pct, label="Kontinuirana", color="#2ecc71", edgecolor="black")
        ax2.bar(
            x,
            other_pct,
            bottom=cont_pct,
            label="Na roku",
            color="#3498db",
            edgecolor="black",
        )
        bottoms_pct = [c + o for c, o in zip(cont_pct, other_pct)]
        ax2.bar(
            x,
            failed_pct,
            bottom=bottoms_pct,
            label="Pali",
            color="#e74c3c",
            edgecolor="black",
        )

        # Add continual % annotation
        for i, pct in enumerate(cont_pct):
            ax2.text(
                i,
                pct / 2,
                f"{pct:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )

        ax2.set_xlabel("Akademska godina")
        ax2.set_ylabel("Postotak (%)")
        ax2.set_title(f"{course} - Postotak ishoda prvog upisa")
        ax2.set_xticks(x)
        ax2.set_xticklabels(years)
        ax2.legend(fontsize=9)
        ax2.set_ylim(0, 100)

        plt.tight_layout()
        save_figure(fig, f"first_enrollment_detail_{course}.png", output_dir)


def plot_enrollment_count_distribution(stats, output_dir):
    """Stacked bar chart showing how many students are on 1st, 2nd, 3rd, ... enrollment per year."""
    ed = stats["enrollment_distribution"]

    for course in ["MA1", "MA2"]:
        years = sorted(ed[course].keys())

        # Gather all enrollment numbers present
        all_enroll_nums = set()
        for y in years:
            all_enroll_nums.update(ed[course][y]["total"].keys())
        max_enroll = max(all_enroll_nums) if all_enroll_nums else 1

        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

        # Left: absolute counts stacked bar
        ax1 = axes[0]
        x = np.arange(len(years))
        enroll_colors = [
            "#2ecc71",
            "#e74c3c",
            "#e67e22",
            "#9b59b6",
            "#1abc9c",
            "#f39c12",
            "#34495e",
        ]
        bottoms = np.zeros(len(years))

        for en in range(1, max_enroll + 1):
            counts = [ed[course][y].get(en, 0) for y in years]
            label = f"{en}. upis" if en <= 5 else f"{en}. upis"
            color = enroll_colors[min(en - 1, len(enroll_colors) - 1)]
            ax1.bar(
                x, counts, bottom=bottoms, label=label, color=color, edgecolor="black"
            )
            bottoms += np.array(counts)

        ax1.set_xlabel("Akademska godina")
        ax1.set_ylabel("Broj studenata")
        ax1.set_title(f"{course} - Distribucija upisa (apsolutno)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(years)
        ax1.legend(fontsize=8, loc="upper left")
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Right: percentage stacked bar
        ax2 = axes[1]
        totals = np.array([sum(ed[course][y]["total"].values()) for y in years], dtype=float)
        bottoms_pct = np.zeros(len(years))

        for en in range(1, max_enroll + 1):
            counts = np.array([ed[course][y]["total"].get(en, 0) for y in years], dtype=float)
            pcts = np.where(totals > 0, counts / totals * 100, 0)
            label = f"{en}. upis"
            color = enroll_colors[min(en - 1, len(enroll_colors) - 1)]
            bars = ax2.bar(
                x, pcts, bottom=bottoms_pct, label=label, color=color, edgecolor="black"
            )

            # Annotate segments > 5%
            for i, (pct, bot) in enumerate(zip(pcts, bottoms_pct)):
                if pct > 5:
                    ax2.text(
                        i,
                        bot + pct / 2,
                        f"{pct:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color="white",
                    )
            bottoms_pct += pcts

        ax2.set_xlabel("Akademska godina")
        ax2.set_ylabel("Postotak (%)")
        ax2.set_title(f"{course} - Distribucija upisa (postotci)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(years)
        ax2.legend(fontsize=8, loc="upper left")
        ax2.set_ylim(0, 100)

        plt.tight_layout()
        save_figure(fig, f"enrollment_distribution_{course}.png", output_dir)


# ---------------------------------------------------------------------------
#  Table visualizations (rendered as images from CSV/stats data)
# ---------------------------------------------------------------------------


def _render_table_figure(
    df, title, filename, output_dir, col_widths=None, highlight_cols=None, figscale=1.0
):
    """Render a pandas DataFrame as a styled matplotlib table figure."""
    n_rows, n_cols = df.shape
    # Dynamic sizing
    col_w = 1.3 * figscale
    row_h = 0.42
    fig_w = max(n_cols * col_w, 8)
    fig_h = max((n_rows + 2) * row_h, 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    cell_text = df.values.tolist()
    col_labels = df.columns.tolist()

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Style header
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#34495e")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_height(0.06)

    # Style rows (alternating)
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = table[i, j]
            cell.set_height(0.05)
            if i % 2 == 0:
                cell.set_facecolor("#ecf0f1")
            else:
                cell.set_facecolor("white")
            # Highlight specific columns with colour
            if highlight_cols and col_labels[j] in highlight_cols:
                cell.set_facecolor("#d5f5e3" if i % 2 == 1 else "#abebc6")

    if col_widths:
        for j, w in enumerate(col_widths):
            for i in range(n_rows + 1):
                table[i, j].set_width(w)
    else:
        table.auto_set_column_width(list(range(n_cols)))

    plt.tight_layout()
    save_figure(fig, filename, output_dir)


def plot_table_summary_statistics(stats, output_dir):
    """Render the summary statistics as a visual table, one per course."""
    for course in ["MA1", "MA2"]:
        years = sorted(stats["single_course"][course].keys())
        rows = []
        for y in years:
            s = stats["single_course"][course][y]
            rows.append(
                {
                    "Godina": y,
                    "Ukupno": s["total_students"],
                    "Položili": s["passed_students"],
                    "Pali": s["failed_students"],
                    "Prolaz. (%)": f"{s['pass_rate']*100:.1f}",
                    "Prosjek bod.": s["avg_points_passed"],
                    "Std bod.": s["std_points_passed"],
                    "Prosjek ocj.": s["avg_grade"],
                    "Std ocj.": s["std_grade"],
                    "Medijan bod.": s["median_points"],
                    "Prosjek pok.": s["avg_attempts_to_pass"],
                    "Odbili ocj.": s["students_rejected_grade"],
                    "Pali (izašli)": s["failed_students_with_attempts"],
                    "Nikad izašli": s["failed_never_tried"],
                }
            )
        df = pd.DataFrame(rows)
        _render_table_figure(
            df,
            f"{course} — Sažetak statistika po godinama",
            f"table_summary_{course}.png",
            output_dir,
            highlight_cols=["Prolaz. (%)", "Prosjek ocj."],
        )


def plot_table_correlation(stats, output_dir):
    """Render correlation analysis as a visual table."""
    years = sorted(stats["correlation"].keys())
    rows = []
    for y in years:
        c = stats["correlation"][y]
        rows.append(
            {
                "Godina": y,
                "Pearson (bod.)": c["pearson_points"] if c["pearson_points"] else "—",
                "Pearson (ocj.)": c["pearson_grades"] if c["pearson_grades"] else "—",
                "Spearman (ocj.)": (
                    c["spearman_grades"] if c["spearman_grades"] else "—"
                ),
                "Oba pol.": c["students_both_passed"],
                "Samo MA1": c["students_ma1_only"],
                "Samo MA2": c["students_ma2_only"],
                "Nijedan": c["students_neither"],
                "MA2 prije MA1": c["ma2_before_ma1"],
                "Regr. nagib": c["regression_slope"] if c["regression_slope"] else "—",
                "R²": c["r_squared"] if c["r_squared"] else "—",
            }
        )
    df = pd.DataFrame(rows)
    _render_table_figure(
        df,
        "Korelacijska analiza MA1 vs MA2 po godinama",
        "table_correlation.png",
        output_dir,
        highlight_cols=["Pearson (bod.)", "R²"],
    )


def plot_table_first_enrollment(stats, output_dir):
    """Render first-enrollment statistics as a visual table, one per course."""
    fe = stats["first_enrollment"]
    for course in ["MA1", "MA2"]:
        years = sorted(fe[course].keys())
        rows = []
        for y in years:
            s = fe[course][y]
            rows.append(
                {
                    "Godina": y,
                    "Brucoši": s["total_first_enrollment"],
                    "Polož. kont.": s["passed_continual_first_enrollment"],
                    "Kont. (%)": f"{s['first_enrollment_continual_rate']*100:.1f}",
                    "Položili ukup.": s["passed_first_enrollment"],
                    "Prolaz (%)": f"{s['first_enrollment_pass_rate']*100:.1f}",
                    "Ponavljači": s["total_repeaters"],
                    "Pon. položili": s["passed_repeaters"],
                    "Pon. prolaz (%)": f"{s['repeater_pass_rate']*100:.1f}",
                    "Svi ukupno": s["total_all"],
                    "Svi polož.": s["passed_all"],
                    "Svi prolaz (%)": f"{s['all_pass_rate']*100:.1f}",
                }
            )
        df = pd.DataFrame(rows)
        _render_table_figure(
            df,
            f"{course} — Statistika prvog upisa (brucoši vs. ponavljači)",
            f"table_first_enrollment_{course}.png",
            output_dir,
            highlight_cols=["Kont. (%)", "Prolaz (%)", "Pon. prolaz (%)"],
        )


def plot_table_enrollment_distribution(stats, output_dir):
    """Render enrollment distribution as a pivot table image, one per course."""
    ed = stats["enrollment_distribution"]
    for course in ["MA1", "MA2"]:
        years = sorted(ed[course].keys())
        # Gather all possible enrollment numbers
        all_enroll = set()
        for y in years:
            all_enroll.update(ed[course][y]["total"].keys())
        max_e = max(all_enroll) if all_enroll else 1

        rows = []
        for y in years:
            row = {"Godina": y}
            total = sum(ed[course][y]["total"].values())
            for e in range(1, max_e + 1):
                cnt = ed[course][y]["total"].get(e, 0)
                pct = cnt / total * 100 if total > 0 else 0
                row[f"{e}. upis"] = f"{cnt} ({pct:.0f}%)" if cnt > 0 else "—"
            row["Ukupno"] = total
            rows.append(row)
        df = pd.DataFrame(rows)
        _render_table_figure(
            df,
            f"{course} — Distribucija broja upisa po godinama",
            f"table_enrollment_dist_{course}.png",
            output_dir,
            highlight_cols=["1. upis"],
        )


def plot_table_enrollment_with_pass_rates(stats, output_dir):
    """Render enrollment distribution with pass rates: two rows per year.

    Row 1 (white/green): enrollment counts per number (same as before)
    Row 2 (tinted):      passed counts / pass rate for each enrollment number
    """
    ed = stats["enrollment_distribution"]

    for course in ["MA1", "MA2"]:
        years = sorted(ed[course].keys())

        all_enroll = set()
        for y in years:
            all_enroll.update(ed[course][y]["total"].keys())
        max_e = max(all_enroll) if all_enroll else 1

        # Build column names
        col_labels = ["Godina"] + [f"{e}. upis" for e in range(1, max_e + 1)] + ["Ukupno"]
        n_cols = len(col_labels)

        # Build cell data: two rows per year
        cell_text = []
        row_types = []  # 'count' or 'pass' — used for styling
        for y in years:
            total_all = sum(ed[course][y]["total"].values())
            passed_all = sum(ed[course][y]["passed"].values())

            # Row 1: enrollment counts
            count_row = [str(y)]
            for e in range(1, max_e + 1):
                cnt = ed[course][y]["total"].get(e, 0)
                pct = cnt / total_all * 100 if total_all > 0 else 0
                count_row.append(f"{cnt} ({pct:.0f}%)" if cnt > 0 else "—")
            count_row.append(str(total_all))
            cell_text.append(count_row)
            row_types.append("count")

            # Row 2: passed counts / pass rates
            pass_row = ["↳ položili"]
            for e in range(1, max_e + 1):
                cnt = ed[course][y]["total"].get(e, 0)
                passed = ed[course][y]["passed"].get(e, 0)
                if cnt > 0:
                    rate = passed / cnt * 100
                    pass_row.append(f"{passed} ({rate:.0f}%)")
                else:
                    pass_row.append("—")
            pass_rate_all = passed_all / total_all * 100 if total_all > 0 else 0
            pass_row.append(f"{passed_all} ({pass_rate_all:.0f}%)")
            cell_text.append(pass_row)
            row_types.append("pass")

        n_rows = len(cell_text)

        # Dynamic figure sizing
        col_w = 1.3
        row_h = 0.38
        fig_w = max(n_cols * col_w, 10)
        fig_h = max((n_rows + 2) * row_h, 4)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")
        ax.set_title(
            f"{course} — Distribucija upisa i prolaznost po rednom broju upisa",
            fontsize=14, fontweight="bold", pad=20,
        )

        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)

        # Style header
        for j in range(n_cols):
            cell = table[0, j]
            cell.set_facecolor("#34495e")
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(0.045)

        # Style data rows
        for i in range(n_rows):
            rt = row_types[i]
            table_row = i + 1  # +1 because row 0 is header
            year_idx = i // 2  # which year block

            for j in range(n_cols):
                cell = table[table_row, j]
                cell.set_height(0.04)

                if rt == "count":
                    # Year count row — alternate by year_idx
                    if year_idx % 2 == 0:
                        cell.set_facecolor("white")
                    else:
                        cell.set_facecolor("#ecf0f1")
                    # Highlight 1. upis column
                    if col_labels[j] == "1. upis":
                        cell.set_facecolor("#d5f5e3" if year_idx % 2 == 0 else "#abebc6")
                    cell.set_text_props(fontweight="bold")
                else:
                    # Pass-rate row — slightly tinted
                    if year_idx % 2 == 0:
                        cell.set_facecolor("#fef9e7")
                    else:
                        cell.set_facecolor("#fdebd0")
                    cell.set_text_props(fontstyle="italic", color="#2c3e50")
                    # Highlight 1. upis column pass rate too
                    if col_labels[j] == "1. upis":
                        cell.set_facecolor("#d4efdf")

        table.auto_set_column_width(list(range(n_cols)))
        plt.tight_layout()
        save_figure(fig, f"table_enrollment_pass_{course}.png", output_dir)


def generate_all_visualizations(processed, merged, all_stats, output_dir):
    figures_dir = os.path.join(output_dir, "figures")
    ensure_dir(figures_dir)

    print("\nGenerating visualizations...")

    plot_summary_dashboard(all_stats, figures_dir)
    plot_pass_rate_by_year(all_stats, figures_dir)
    plot_enrollment_trend(all_stats, figures_dir)
    plot_grade_distribution_combined(processed, figures_dir)
    plot_covid_comparison(all_stats, figures_dir)
    plot_correlation_trend(all_stats, figures_dir)
    plot_ma2_before_ma1(all_stats, figures_dir)
    plot_most_common_pass_exam(all_stats, figures_dir)
    plot_rejection_analysis(processed, all_stats, figures_dir)
    plot_failed_analysis(processed, all_stats, figures_dir)

    plot_attempts_distribution(all_stats, figures_dir)
    plot_pass_rate_by_exam_period(all_stats, figures_dir)
    plot_points_by_exam_period(processed, figures_dir)

    plot_grade_heatmap_combined(all_stats, figures_dir)
    plot_scatter_points_combined(merged, all_stats, figures_dir)
    plot_ma1_predicts_ma2(all_stats, figures_dir)

    # New: first-enrollment and enrollment distribution plots
    plot_first_enrollment_pass_rate(all_stats, figures_dir)
    plot_first_enrollment_detail(all_stats, figures_dir)
    plot_enrollment_count_distribution(all_stats, figures_dir)

    # Table visualizations (rendered images of all CSV/report data)
    plot_table_summary_statistics(all_stats, figures_dir)
    plot_table_correlation(all_stats, figures_dir)
    plot_table_first_enrollment(all_stats, figures_dir)
    plot_table_enrollment_distribution(all_stats, figures_dir)
    plot_table_enrollment_with_pass_rates(all_stats, figures_dir)

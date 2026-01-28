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


def generate_all_visualizations(processed, merged, all_stats, output_dir):
    figures_dir = os.path.join(output_dir, "figures")
    ensure_dir(figures_dir)

    print("\nGenerating visualizations...")

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

import os
import pandas as pd
from src.ingestion import load_all_csvs
from src.processing import process_all_data, create_merged_data
from src.analysis import compute_all_statistics
from src.visualization import generate_all_visualizations

DATA_DIR = "data/MATAN"
OUTPUT_DIR = "output"


def save_summary_csv(stats, output_dir):
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    rows = []
    for course in ["MA1", "MA2"]:
        for year, s in stats["single_course"][course].items():
            row = {
                "year": year,
                "course": course,
                "total_students": s["total_students"],
                "passed_students": s["passed_students"],
                "failed_students": s["failed_students"],
                "pass_rate": s["pass_rate"],
                "pass_threshold": s["pass_threshold"],
                "avg_points_passed": s["avg_points_passed"],
                "std_points_passed": s["std_points_passed"],
                "avg_grade": s["avg_grade"],
                "std_grade": s["std_grade"],
                "median_points": s["median_points"],
                "avg_attempts": s["avg_attempts_to_pass"],
                "rejected_grade": s["students_rejected_grade"],
                "failed_with_attempts": s["failed_students_with_attempts"],
                "failed_never_tried": s["failed_never_tried"],
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(reports_dir, "summary_statistics.csv"), index=False)
    print(f"  - summary_statistics.csv... saved")

    corr_rows = []
    for year, c in stats["correlation"].items():
        corr_rows.append(
            {
                "year": year,
                "pearson_points": c["pearson_points"],
                "pearson_grades": c["pearson_grades"],
                "spearman_grades": c["spearman_grades"],
                "both_passed": c["students_both_passed"],
                "ma1_only": c["students_ma1_only"],
                "ma2_only": c["students_ma2_only"],
                "neither": c["students_neither"],
                "ma2_before_ma1": c["ma2_before_ma1"],
                "regression_slope": c["regression_slope"],
                "regression_intercept": c["regression_intercept"],
                "r_squared": c["r_squared"],
            }
        )

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(os.path.join(reports_dir, "correlation_analysis.csv"), index=False)
    print(f"  - correlation_analysis.csv... saved")


def print_summary(stats):
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    total_ma1 = sum(s["total_students"] for s in stats["single_course"]["MA1"].values())
    total_ma2 = sum(s["total_students"] for s in stats["single_course"]["MA2"].values())

    avg_pass_ma1 = sum(
        s["pass_rate"] for s in stats["single_course"]["MA1"].values()
    ) / len(stats["single_course"]["MA1"])
    avg_pass_ma2 = sum(
        s["pass_rate"] for s in stats["single_course"]["MA2"].values()
    ) / len(stats["single_course"]["MA2"])

    print(f"Total MA1 student records: {total_ma1}")
    print(f"Total MA2 student records: {total_ma2}")
    print(f"Average pass rate MA1: {avg_pass_ma1*100:.1f}%")
    print(f"Average pass rate MA2: {avg_pass_ma2*100:.1f}%")

    correlations = [
        c["pearson_points"]
        for c in stats["correlation"].values()
        if c["pearson_points"]
    ]
    if correlations:
        avg_corr = sum(correlations) / len(correlations)
        print(f"Average correlation (points): {avg_corr:.3f}")

    total_ma2_before = sum(c["ma2_before_ma1"] for c in stats["correlation"].values())
    print(f"Students who passed MA2 before MA1: {total_ma2_before}")

    eh = stats["easiest_hardest"]
    if eh["easiest"]:
        print(
            f"\nEasiest exam: {eh['easiest']['course']} {eh['easiest']['year']} ({eh['easiest']['pass_rate']*100:.1f}%)"
        )
    if eh["hardest"]:
        print(
            f"Hardest exam: {eh['hardest']['course']} {eh['hardest']['year']} ({eh['hardest']['pass_rate']*100:.1f}%)"
        )


def main():
    print("=" * 50)
    print("MATAN Analysis Tool")
    print("=" * 50)

    print("\nLoading data...")
    data = load_all_csvs(DATA_DIR)

    print("\nProcessing data...")
    processed = process_all_data(data)
    merged = create_merged_data(processed)
    print(f"  - Processed {len(processed['MA1'])} years of MA1 data")
    print(f"  - Processed {len(processed['MA2'])} years of MA2 data")
    print(f"  - Created {len(merged)} merged datasets")

    print("\nRunning analyses...")
    stats = compute_all_statistics(processed, merged)
    print("  - Single course statistics... done")
    print("  - Correlation analysis... done")
    print("  - COVID impact analysis... done")

    generate_all_visualizations(processed, merged, stats, OUTPUT_DIR)

    print("\nSaving reports...")
    save_summary_csv(stats, OUTPUT_DIR)

    print_summary(stats)

    print(f"\nOutput saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

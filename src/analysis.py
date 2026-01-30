import pandas as pd
import numpy as np
from scipy import stats


def detect_pass_threshold(df, year=None):

    try:
        year_int = int(year) if year is not None else None
    except Exception:
        year_int = None

    if year_int is not None:
        return 45 if year_int <= 2022 else 50

    # If year is unknown, default to 50
    return 50


def single_course_stats(df, course, year):
    total = len(df)
    passed = df["passed"].sum()
    failed = total - passed
    pass_rate = passed / total if total > 0 else 0

    passed_df = df[df["passed"]]
    failed_df = df[~df["passed"]]

    pass_threshold = detect_pass_threshold(df, year)

    result = {
        "year": year,
        "course": course,
        "total_students": total,
        "passed_students": int(passed),
        "failed_students": int(failed),
        "pass_rate": round(pass_rate, 4),
        "pass_threshold": pass_threshold,
        "avg_points_passed": (
            round(float(passed_df["final_points"].mean()), 2)
            if len(passed_df) > 0
            else 0
        ),
        "std_points_passed": (
            round(float(passed_df["final_points"].std()), 2)
            if len(passed_df) > 0
            else 0
        ),
        "avg_grade": (
            round(float(passed_df["final_grade"].mean()), 2)
            if len(passed_df) > 0
            else 0
        ),
        "std_grade": (
            round(float(passed_df["final_grade"].std()), 2) if len(passed_df) > 0 else 0
        ),
        "median_points": (
            round(float(passed_df["final_points"].median()), 2)
            if len(passed_df) > 0
            else 0
        ),
        "min_points": (
            round(float(passed_df["final_points"].min()), 2)
            if len(passed_df) > 0
            else 0
        ),
        "max_points": (
            round(float(passed_df["final_points"].max()), 2)
            if len(passed_df) > 0
            else 0
        ),
    }

    grade_dist = passed_df["final_grade"].value_counts().to_dict()
    result["grade_distribution"] = {int(k): int(v) for k, v in grade_dist.items()}

    pass_by_exam = df[df["passed"]]["passed_on_exam"].value_counts().to_dict()
    result["pass_by_exam"] = pass_by_exam

    passed_attempts = df[df["passed"]]["num_attempts"]
    result["avg_attempts_to_pass"] = (
        round(float(passed_attempts.mean()), 2) if len(passed_attempts) > 0 else 0
    )

    result["students_rejected_grade"] = int(df["rejected_grade"].sum())
    result["grade_improved_after_reject"] = int(
        (df["rejected_grade"] & (df["grade_change"] > 0)).sum()
    )
    result["grade_worsened_after_reject"] = int(
        (df["rejected_grade"] & (df["grade_change"] < 0)).sum()
    )

    rejected_and_failed = int((df["rejected_grade"] & ~df["passed"]).sum())
    result["rejected_and_failed"] = rejected_and_failed

    failed_with_attempts = failed_df[failed_df["num_attempts"] > 0]
    result["failed_students_with_attempts"] = int(len(failed_with_attempts))
    result["failed_never_tried"] = int(len(failed_df[failed_df["num_attempts"] == 0]))

    return result


def pass_rate_by_exam(df, course):
    from src.processing import get_exam_columns

    exams = get_exam_columns(df)

    result = {}
    cumulative_passed_ids = set()

    for name, points_col, prolaz_col, time_col in exams:
        if points_col not in df.columns:
            continue

        attempts = int((df[points_col] > 0).sum())
        passed_this_exam = (
            set(df[df[prolaz_col]]["id"].tolist())
            if prolaz_col in df.columns
            else set()
        )
        passed_count = len(passed_this_exam)

        new_passed = passed_this_exam - cumulative_passed_ids
        cumulative_passed_ids.update(passed_this_exam)

        rate = passed_count / attempts if attempts > 0 else 0

        result[name] = {
            "attempts": attempts,
            "passed": passed_count,
            "rate": round(rate, 4),
            "new_passed": len(new_passed),
            "cumulative_passed": len(cumulative_passed_ids),
            "cumulative_rate": (
                round(len(cumulative_passed_ids) / len(df), 4) if len(df) > 0 else 0
            ),
        }

    return result


def attempts_distribution(df):
    passed_df = df[df["passed"]]
    dist = passed_df["num_attempts"].value_counts().sort_index().to_dict()

    result = {}
    for k, v in dist.items():
        if k == 0:
            continue
        if k >= 5:
            result["5+"] = result.get("5+", 0) + int(v)
        else:
            result[int(k)] = int(v)

    return result


def failed_attempts_distribution(df):
    failed_df = df[~df["passed"]]
    dist = failed_df["num_attempts"].value_counts().sort_index().to_dict()

    result = {}
    for k, v in dist.items():
        if k >= 5:
            result["5+"] = result.get("5+", 0) + int(v)
        else:
            result[int(k)] = int(v)

    return result


def correlation_analysis(merged_df):
    both_passed = merged_df[merged_df["both_passed"]].copy()
    both_passed = both_passed.dropna(
        subset=["ma1_points", "ma2_points", "ma1_grade", "ma2_grade"]
    )

    if len(both_passed) < 2:
        return {
            "pearson_points": None,
            "pearson_grades": None,
            "spearman_grades": None,
            "students_both_passed": 0,
            "students_ma1_only": 0,
            "students_ma2_only": 0,
            "students_neither": 0,
            "ma2_before_ma1": 0,
            "regression_slope": None,
            "regression_intercept": None,
            "r_squared": None,
        }

    pearson_points_result = stats.pearsonr(
        both_passed["ma1_points"], both_passed["ma2_points"]
    )
    pearson_points = float(pearson_points_result.statistic)

    pearson_grades_result = stats.pearsonr(
        both_passed["ma1_grade"], both_passed["ma2_grade"]
    )
    pearson_grades = float(pearson_grades_result.statistic)

    spearman_grades_result = stats.spearmanr(
        both_passed["ma1_grade"], both_passed["ma2_grade"]
    )
    spearman_grades = float(spearman_grades_result.statistic)

    linreg_result = stats.linregress(
        both_passed["ma1_points"], both_passed["ma2_points"]
    )
    slope = linreg_result.slope
    intercept = linreg_result.intercept
    r_value = linreg_result.rvalue

    return {
        "pearson_points": round(pearson_points, 4),
        "pearson_grades": round(pearson_grades, 4),
        "spearman_grades": round(spearman_grades, 4),
        "students_both_passed": int(merged_df["both_passed"].sum()),
        "students_ma1_only": int(
            (merged_df["ma1_passed"] & ~merged_df["ma2_passed"]).sum()
        ),
        "students_ma2_only": int(
            (~merged_df["ma1_passed"] & merged_df["ma2_passed"]).sum()
        ),
        "students_neither": int(
            (~merged_df["ma1_passed"] & ~merged_df["ma2_passed"]).sum()
        ),
        "ma2_before_ma1": int(merged_df["ma2_before_ma1"].sum()),
        "regression_slope": round(float(slope), 4),
        "regression_intercept": round(float(intercept), 4),
        "r_squared": round(float(r_value**2), 4),
    }


def ma1_predicts_ma2(merged_df):
    ma1_passed = merged_df[merged_df["ma1_passed"]].copy()
    ma1_passed = ma1_passed.dropna(subset=["ma1_grade"])

    if len(ma1_passed) < 10:
        return None

    result = {}
    for grade in [2, 3, 4, 5]:
        subset = ma1_passed[ma1_passed["ma1_grade"] == grade]
        if len(subset) == 0:
            continue

        ma2_passed_count = int(subset["ma2_passed"].sum())
        total = len(subset)

        ma2_grades = subset[subset["ma2_passed"]]["ma2_grade"]
        avg_ma2 = round(float(ma2_grades.mean()), 2) if len(ma2_grades) > 0 else None

        result[grade] = {
            "total": int(total),
            "ma2_passed": ma2_passed_count,
            "ma2_pass_rate": round(ma2_passed_count / total, 4) if total > 0 else 0,
            "avg_ma2_grade": avg_ma2,
        }

    return result


def grade_matrix(merged_df):
    both = merged_df[merged_df["both_passed"]].copy()
    both["ma1_grade_int"] = both["ma1_grade"].astype(int)
    both["ma2_grade_int"] = both["ma2_grade"].astype(int)

    matrix = pd.crosstab(
        both["ma1_grade_int"], both["ma2_grade_int"], rownames=["MA1"], colnames=["MA2"]
    )

    for grade in [2, 3, 4, 5]:
        if grade not in matrix.index:
            matrix.loc[grade] = 0
        if grade not in matrix.columns:
            matrix[grade] = 0

    matrix = matrix.reindex(index=[2, 3, 4, 5], columns=[2, 3, 4, 5], fill_value=0)

    return matrix


def year_over_year_comparison(processed, merged):
    rows = []

    years = sorted(set(processed["MA1"].keys()) | set(processed["MA2"].keys()))

    for year in years:
        row = {"year": year}

        if year in processed["MA1"]:
            ma1_df = processed["MA1"][year]
            row["ma1_total"] = len(ma1_df)
            row["ma1_pass_rate"] = round(ma1_df["passed"].mean(), 4)
            passed_ma1 = ma1_df[ma1_df["passed"]]
            row["ma1_avg_grade"] = (
                round(float(passed_ma1["final_grade"].mean()), 2)
                if len(passed_ma1) > 0
                else None
            )

        if year in processed["MA2"]:
            ma2_df = processed["MA2"][year]
            row["ma2_total"] = len(ma2_df)
            row["ma2_pass_rate"] = round(ma2_df["passed"].mean(), 4)
            passed_ma2 = ma2_df[ma2_df["passed"]]
            row["ma2_avg_grade"] = (
                round(float(passed_ma2["final_grade"].mean()), 2)
                if len(passed_ma2) > 0
                else None
            )

        if year in merged:
            corr = correlation_analysis(merged[year])
            row["correlation_points"] = corr["pearson_points"]
            row["ma2_before_ma1"] = corr["ma2_before_ma1"]

        rows.append(row)

    return pd.DataFrame(rows)


def covid_impact_analysis(processed):
    pre_covid_years = [2018]
    covid_years = [2019, 2020]
    post_covid_years = [2021, 2022, 2023, 2024]

    def avg_pass_rate(years, course_data):
        rates = []
        for y in years:
            if y in course_data:
                rates.append(course_data[y]["passed"].mean())
        return float(np.mean(rates)) if rates else None

    result = {}
    for course in ["MA1", "MA2"]:
        pre = avg_pass_rate(pre_covid_years, processed[course])
        covid = avg_pass_rate(covid_years, processed[course])
        post = avg_pass_rate(post_covid_years, processed[course])

        result[course] = {
            "pre_covid_pass_rate": round(pre, 4) if pre else None,
            "covid_pass_rate": round(covid, 4) if covid else None,
            "post_covid_pass_rate": round(post, 4) if post else None,
        }

        if pre and covid:
            diff = covid - pre
            result[course]["covid_difference"] = round(diff, 4)

    return result


def easiest_hardest_exams(processed):
    all_stats = []

    for course in ["MA1", "MA2"]:
        for year, df in processed[course].items():
            pass_rate = float(df["passed"].mean())
            all_stats.append({"year": year, "course": course, "pass_rate": pass_rate})

    if not all_stats:
        return {"easiest": None, "hardest": None}

    easiest = max(all_stats, key=lambda x: x["pass_rate"])
    hardest = min(all_stats, key=lambda x: x["pass_rate"])

    return {
        "easiest": easiest,
        "hardest": hardest,
        "all": sorted(all_stats, key=lambda x: x["pass_rate"], reverse=True),
    }


def cross_year_rejections(processed):
    """
    Find students who passed an exam (had 'DA') in year X but didn't finalize
    (no ISVU Ocjena) and re-enrolled in year X+1.
    These are students who rejected their grade and had to retake the course.
    """
    from src.processing import get_exam_columns

    result = {"MA1": {}, "MA2": {}}

    for course in ["MA1", "MA2"]:
        years = sorted(processed[course].keys())

        for i in range(len(years) - 1):
            year_current = years[i]
            year_next = years[i + 1]

            df_current = processed[course][year_current]
            df_next = processed[course][year_next]

            # Students who appear in both years
            ids_current = set(df_current["id"].tolist())
            ids_next = set(df_next["id"].tolist())
            common_ids = ids_current & ids_next

            if not common_ids:
                continue

            # Find students who had 'DA' (passed) on any exam in current year
            # but didn't finalize (passed = False, meaning no ISVU Ocjena)
            exams = get_exam_columns(df_current)

            rejected_students = []
            for student_id in common_ids:
                student_row = df_current[df_current["id"] == student_id].iloc[0]

                # Check if student had 'DA' on any exam
                had_da = False
                for name, points_col, prolaz_col, time_col in exams:
                    if prolaz_col in student_row.index and student_row[prolaz_col]:
                        had_da = True
                        break

                # If they had DA but didn't finalize (passed = False)
                if had_da and not student_row["passed"]:
                    rejected_students.append(student_id)

            key = f"{year_current}->{year_next}"
            result[course][key] = {
                "count": len(rejected_students),
                "students": rejected_students,
            }

    return result


def statistical_significance_tests(processed):
    """
    Perform statistical significance tests:
    1. T-test: Is MA2 significantly harder than MA1?
    2. Linear regression trend analysis for each course
    """
    years = sorted(processed["MA1"].keys())
    
    ma1_rates = [processed["MA1"][y]["passed"].mean() for y in years]
    ma2_rates = [processed["MA2"][y]["passed"].mean() for y in years]
    
    # T-test
    t_stat, p_value = stats.ttest_ind(ma1_rates, ma2_rates)
    
    # Trend analysis
    year_nums = list(range(len(years)))
    ma1_trend = stats.linregress(year_nums, ma1_rates)
    ma2_trend = stats.linregress(year_nums, ma2_rates)
    
    return {
        "ttest": {
            "ma1_mean_pass_rate": round(float(np.mean(ma1_rates)), 4),
            "ma2_mean_pass_rate": round(float(np.mean(ma2_rates)), 4),
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value), 6),
            "significant": p_value < 0.05,
        },
        "ma1_trend": {
            "slope_per_year": round(float(ma1_trend.slope), 4),
            "r_squared": round(float(ma1_trend.rvalue ** 2), 4),
            "p_value": round(float(ma1_trend.pvalue), 4),
            "significant": ma1_trend.pvalue < 0.05,
        },
        "ma2_trend": {
            "slope_per_year": round(float(ma2_trend.slope), 4),
            "r_squared": round(float(ma2_trend.rvalue ** 2), 4),
            "p_value": round(float(ma2_trend.pvalue), 4),
            "significant": ma2_trend.pvalue < 0.05,
        },
    }


def grade_transition_analysis(grade_matrices):
    """
    Analyze how grades change from MA1 to MA2.
    """
    improved = 0
    same = 0
    dropped = 0
    
    for year, matrix in grade_matrices.items():
        for ma1 in [2, 3, 4, 5]:
            for ma2 in [2, 3, 4, 5]:
                count = matrix.loc[ma1, ma2] if ma1 in matrix.index and ma2 in matrix.columns else 0
                if ma2 > ma1:
                    improved += count
                elif ma2 == ma1:
                    same += count
                else:
                    dropped += count
    
    total = improved + same + dropped
    return {
        "improved": int(improved),
        "same": int(same),
        "dropped": int(dropped),
        "improved_pct": round(improved / total * 100, 1) if total > 0 else 0,
        "same_pct": round(same / total * 100, 1) if total > 0 else 0,
        "dropped_pct": round(dropped / total * 100, 1) if total > 0 else 0,
        "total": int(total),
    }


def dropout_analysis(processed):
    """
    Analyze students who enrolled but never attempted an exam.
    """
    result = {}
    for course in ["MA1", "MA2"]:
        total_students = 0
        never_tried = 0
        for year, df in processed[course].items():
            total_students += len(df)
            never_tried += (df["num_attempts"] == 0).sum()
        
        result[course] = {
            "total_enrolled": int(total_students),
            "never_attempted": int(never_tried),
            "dropout_rate": round(never_tried / total_students * 100, 2) if total_students > 0 else 0,
        }
    return result


def perfect_scores_analysis(processed):
    """
    Count students who achieved perfect scores (100 points).
    """
    result = {}
    for course in ["MA1", "MA2"]:
        perfect = 0
        for year, df in processed[course].items():
            perfect += (df["final_points"] == 100).sum()
        result[course] = int(perfect)
    return result


def compute_all_statistics(processed, merged):
    all_stats = {
        "single_course": {"MA1": {}, "MA2": {}},
        "pass_by_exam": {"MA1": {}, "MA2": {}},
        "attempts_dist": {"MA1": {}, "MA2": {}},
        "failed_attempts_dist": {"MA1": {}, "MA2": {}},
        "correlation": {},
        "grade_matrix": {},
        "ma1_predicts_ma2": {},
        "year_comparison": None,
        "covid_impact": None,
        "easiest_hardest": None,
        "cross_year_rejections": None,
        "statistical_tests": None,
        "grade_transition": None,
        "dropout": None,
        "perfect_scores": None,
    }

    for course in ["MA1", "MA2"]:
        for year, df in processed[course].items():
            all_stats["single_course"][course][year] = single_course_stats(
                df, course, year
            )
            all_stats["pass_by_exam"][course][year] = pass_rate_by_exam(df, course)
            all_stats["attempts_dist"][course][year] = attempts_distribution(df)
            all_stats["failed_attempts_dist"][course][year] = (
                failed_attempts_distribution(df)
            )

    for year, df in merged.items():
        all_stats["correlation"][year] = correlation_analysis(df)
        all_stats["grade_matrix"][year] = grade_matrix(df)
        all_stats["ma1_predicts_ma2"][year] = ma1_predicts_ma2(df)

    all_stats["year_comparison"] = year_over_year_comparison(processed, merged)
    all_stats["covid_impact"] = covid_impact_analysis(processed)
    all_stats["easiest_hardest"] = easiest_hardest_exams(processed)
    all_stats["cross_year_rejections"] = cross_year_rejections(processed)
    all_stats["statistical_tests"] = statistical_significance_tests(processed)
    all_stats["grade_transition"] = grade_transition_analysis(all_stats["grade_matrix"])
    all_stats["dropout"] = dropout_analysis(processed)
    all_stats["perfect_scores"] = perfect_scores_analysis(processed)

    return all_stats

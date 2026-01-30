import pandas as pd
import numpy as np


def get_exam_columns(df):
    if hasattr(df, "attrs") and "exams" in df.attrs:
        return df.attrs["exams"]

    exams = []
    bodovi_cols = [
        c for c in df.columns if "bodovi" in c.lower() and c != "ISVU Bodovi"
    ]

    for bodovi_col in bodovi_cols:
        base = bodovi_col.replace(" - bodovi", "")
        prolaz_col = f"{base} - prolaz"
        vrijeme_col = f"{base} - vrijeme"

        if prolaz_col in df.columns and vrijeme_col in df.columns:
            exams.append((base, bodovi_col, prolaz_col, vrijeme_col))

    return exams


def clean_dataframe(df):
    df = df.copy()
    if hasattr(df, "attrs"):
        df.attrs = df.attrs.copy()

    for col in df.columns:
        if "prolaz" in col:
            df[col] = df[col].astype(str).str.strip().str.upper() == "DA"

    for col in df.columns:
        if "bodovi" in col.lower() and col != "ISVU Bodovi":
            # Handle European decimal format (comma as decimal separator)
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Handle ISVU columns with European decimal format
    df["ISVU Bodovi"] = df["ISVU Bodovi"].astype(str).str.replace(",", ".", regex=False)
    df["ISVU Bodovi"] = pd.to_numeric(df["ISVU Bodovi"], errors="coerce")
    df["ISVU Ocjena"] = pd.to_numeric(df["ISVU Ocjena"], errors="coerce")

    return df


def calculate_attempts(row, exams):
    attempts = 0
    for name, points_col, prolaz_col, time_col in exams:
        if points_col in row.index and row[points_col] > 0:
            attempts += 1
    return attempts


def determine_pass_exam(row, exams):
    for name, points_col, prolaz_col, time_col in exams:
        if prolaz_col in row.index and row[prolaz_col]:
            return name
    return None


def determine_pass_date(row, exams):
    for name, points_col, prolaz_col, time_col in exams:
        if prolaz_col in row.index and row[prolaz_col]:
            return row[time_col] if time_col in row.index else None
    return None


def add_computed_columns(df, course):
    df = df.copy()
    if hasattr(df, "attrs"):
        df.attrs = df.attrs.copy()

    exams = get_exam_columns(df)

    df["passed"] = df["ISVU Ocjena"].notna()
    df["final_grade"] = df["ISVU Ocjena"]
    df["final_points"] = df["ISVU Bodovi"]

    df["num_attempts"] = df.apply(lambda row: calculate_attempts(row, exams), axis=1)
    df["passed_on_exam"] = df.apply(lambda row: determine_pass_exam(row, exams), axis=1)
    df["pass_date"] = df.apply(lambda row: determine_pass_date(row, exams), axis=1)

    if exams:
        kont_prolaz = exams[0][2]
        df["passed_on_continual"] = (
            df[kont_prolaz] if kont_prolaz in df.columns else False
        )
    else:
        df["passed_on_continual"] = False

    return df


def detect_grade_rejection(df, course, year=None):
    df = df.copy()
    if hasattr(df, "attrs"):
        df.attrs = df.attrs.copy()

    exams = get_exam_columns(df)

    df["rejected_grade"] = False
    df["grade_change"] = 0

    for idx, row in df.iterrows():
        if not row["passed"]:
            continue

        pass_found = False
        first_pass_points = None

        for name, points_col, prolaz_col, time_col in exams:
            if prolaz_col not in row.index:
                continue
            if row[prolaz_col] and not pass_found:
                pass_found = True
                first_pass_points = row[points_col] if points_col in row.index else None
            elif pass_found and points_col in row.index and row[points_col] > 0:
                df.at[idx, "rejected_grade"] = True
                old_grade = points_to_grade(first_pass_points, year)
                new_grade = row["final_grade"]
                if old_grade and new_grade:
                    df.at[idx, "grade_change"] = int(new_grade - old_grade)
                break

    return df


def points_to_grade(points, year=None):
    """Convert `points` to a grade using year-specific boundaries.

    If `year` is provided and is <= 2022, use the older boundaries:
      2 - 45, 3 - 55, 4 - 70, 5 - 85
    Otherwise use the newer boundaries:
      2 - 50, 3 - 58, 4 - 72, 5 - 86
    """
    if pd.isna(points):
        return None

    try:
        year_int = int(year) if year is not None else None
    except Exception:
        year_int = None

    # Older boundaries for years up to and including 2022
    if year_int is not None and year_int <= 2022:
        if points < 45:
            return None
        elif points < 55:
            return 2
        elif points < 70:
            return 3
        elif points < 85:
            return 4
        else:
            return 5

    # Newer boundaries for years after 2022 (or unknown year)
    if points < 50:
        return None
    elif points < 58:
        return 2
    elif points < 72:
        return 3
    elif points < 86:
        return 4
    else:
        return 5


def merge_ma1_ma2(ma1_df, ma2_df):
    ma1_subset = ma1_df[
        ["id", "passed", "final_grade", "final_points", "pass_date", "num_attempts"]
    ].copy()
    ma1_subset.columns = [
        "id",
        "ma1_passed",
        "ma1_grade",
        "ma1_points",
        "ma1_pass_date",
        "ma1_attempts",
    ]

    ma2_subset = ma2_df[
        ["id", "passed", "final_grade", "final_points", "pass_date", "num_attempts"]
    ].copy()
    ma2_subset.columns = [
        "id",
        "ma2_passed",
        "ma2_grade",
        "ma2_points",
        "ma2_pass_date",
        "ma2_attempts",
    ]

    merged = pd.merge(ma1_subset, ma2_subset, on="id", how="outer")

    merged["ma1_passed"] = merged["ma1_passed"].fillna(False)
    merged["ma2_passed"] = merged["ma2_passed"].fillna(False)

    merged["both_passed"] = merged["ma1_passed"] & merged["ma2_passed"]

    merged["ma2_before_ma1"] = False
    mask = (
        merged["both_passed"]
        & merged["ma1_pass_date"].notna()
        & merged["ma2_pass_date"].notna()
    )
    merged.loc[mask, "ma2_before_ma1"] = (
        merged.loc[mask, "ma2_pass_date"] < merged.loc[mask, "ma1_pass_date"]
    )

    return merged


def process_all_data(data):
    processed = {"MA1": {}, "MA2": {}}

    for course in ["MA1", "MA2"]:
        for year, df in data[course].items():
            df_clean = clean_dataframe(df)
            df_computed = add_computed_columns(df_clean, course)
            df_final = detect_grade_rejection(df_computed, course, year)
            processed[course][year] = df_final

    return processed


def create_merged_data(processed):
    merged = {}

    ma1_years = set(processed["MA1"].keys())
    ma2_years = set(processed["MA2"].keys())
    common_years = ma1_years & ma2_years

    for year in common_years:
        merged[year] = merge_ma1_ma2(processed["MA1"][year], processed["MA2"][year])

    return merged

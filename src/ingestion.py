import pandas as pd
import os
from glob import glob
import re

REQUIRED_COLUMNS = ["id", "ISVU Bodovi", "ISVU Ocjena", "ISVU Rok"]


def extract_year_and_course(filename):
    basename = os.path.basename(filename)
    match = re.match(r"(MA[12])_(\d{4})_clean\.csv", basename)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def parse_csv_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline()

    if first_line.count(";") > first_line.count(","):
        sep = ";"
    else:
        sep = ","

    df = pd.read_csv(filepath, sep=sep, encoding="utf-8", na_values=["", " ", '""'])
    return df


def parse_dates(df):
    date_columns = [col for col in df.columns if "vrijeme" in col or col == "ISVU Rok"]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def get_exam_columns(df):
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


def validate_dataframe(df):
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        return False

    exams = get_exam_columns(df)
    return len(exams) >= 2


def load_all_csvs(data_dir):
    data = {"MA1": {}, "MA2": {}}

    csv_files = glob(os.path.join(data_dir, "*.csv"))

    for filepath in csv_files:
        course, year = extract_year_and_course(filepath)
        if course is None:
            continue

        df = parse_csv_file(filepath)

        if not validate_dataframe(df):
            print(f"Warning: {filepath} has missing required columns")
            continue

        df = parse_dates(df)
        df.attrs["exams"] = get_exam_columns(df)
        data[course][year] = df
        print(
            f"  - Loaded {course}_{year}: {len(df)} students ({len(df.attrs['exams'])} exam periods)"
        )

    return data

import pandas as pd
import numpy as np
import re
from datetime import datetime

class EDA:

    @staticmethod
    def data_overview(df: pd.DataFrame, writer=print):
        buffer = []
        writer("### Data Overview:")
        info_buf = []
        df.info(buf=info_buf)
        writer("\n".join(info_buf))
        writer("\nFirst 5 Rows:")
        writer(df.head())
        writer("\nStatistical Summary:")
        writer(df.describe(include='all'))

    @staticmethod
    def remove_duplicates(df: pd.DataFrame, writer=print):
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        writer(f"Removed {before - after} duplicate rows (if any).")
        return df

    @staticmethod
    def data_type_conversion(df: pd.DataFrame, writer=print):
        log_entries = []
        date_regex = r'^\s*(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})\s*$'
        numeric_regex = r'^[\+\-]?(?:\d+|\d{1,3}(?:,\d{3})+)?(?:\.\d+)?(?:[eE][\+\-]?\d+)?$'

        for col in df.columns:
            old_type = df[col].dtype
            series = df[col].astype(str).str.strip()
            non_null = series[series.notna() & (series != 'nan')]

            if non_null.empty:
                log_entries.append([col, old_type, old_type, 0, "empty/unchanged"])
                continue

            date_mask = non_null.str.match(date_regex, na=False)
            date_ratio = date_mask.sum() / len(non_null)

            if date_ratio > 0.5 or any(k in col.lower() for k in ("date", "time", "ship", "order", "dob")):
                parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=True, infer_datetime_format=True)
                success = parsed.notna().sum() / len(df[col])
                if success > 0.5:
                    df[col] = parsed
                    log_entries.append([col, old_type, "datetime64[ns]", round(success * 100, 2), "date pattern matched"])
                    continue

            candidate = non_null[~non_null.str.match(date_regex, na=False)]
            cand_clean = candidate.str.replace(r'\s+', '', regex=True).str.replace(',', '', regex=False)
            numeric_mask = cand_clean.str.match(numeric_regex, na=False)
            numeric_ratio = numeric_mask.sum() / len(non_null)

            if numeric_ratio > 0.8:
                cleaned = non_null.str.replace(r'[^\d\.\-\+eE,]', '', regex=True).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(cleaned, errors='coerce')

                if np.all(df[col].dropna() == np.floor(df[col].dropna())):
                    df[col] = df[col].astype("Int64")
                    new_type = "int"
                else:
                    new_type = "float"

                log_entries.append([col, old_type, new_type, round(numeric_ratio * 100, 2), "numeric pattern matched"])
                continue

            log_entries.append([col, old_type, old_type, 0, "unchanged"])

        log_df = pd.DataFrame(log_entries, columns=["Column", "Old Type", "New Type", "Confidence (%)", "Reason"])
        writer("### Data Type Conversion Log:")
        writer(log_df)
        return df

    @staticmethod
    def fill_missing_values(df: pd.DataFrame, writer=print):
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    skewness = df[col].skew()
                    if abs(skewness) < 0.5:
                        fill_value = df[col].mean()
                        writer(f"Filled missing values in numeric column '{col}' with mean: {fill_value}.")
                    else:
                        fill_value = df[col].median()
                        writer(f"Filled missing values in numeric column '{col}' with median: {fill_value}.")
                    df[col].fillna(fill_value, inplace=True)

                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    mode_series = df[col].mode()
                    if not mode_series.empty:
                        fill_value = mode_series[0]
                        df[col].fillna(fill_value, inplace=True)
                        writer(f"Filled missing values in datetime column '{col}' with mode: {fill_value}.")
                    # keep as datetime dtype — don’t convert to string

                elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                    mode_series = df[col].mode()
                    fill_value = mode_series[0] if not mode_series.empty else "N/A"
                    df[col].fillna(fill_value, inplace=True)
                    writer(f"Filled missing values in categorical column '{col}' with mode: {fill_value}.")

        return df


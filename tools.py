import json
import pandas as pd
import numpy as np

MAX_GROUPS = 40          # cap result size so the context window survives


def _load(path: str) -> pd.DataFrame:
    """Load CSV or Excel. Parse anything that looks like a date."""
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    for c in df.columns:
        if df[c].dtype == object and ("date" in c.lower() or "time" in c.lower()):
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _check(df, col, name="column"):
    if col not in df.columns:
        raise ValueError(f"No {name} '{col}'. Available: {list(df.columns)}")


# ---------------------------------------------------------------- inspection

def describe_columns(path: str) -> str:
    """Column names, dtypes, row count."""
    df = _load(path)
    return json.dumps({
        "rows": int(df.shape[0]),
        "columns": {c: str(df[c].dtype) for c in df.columns},
    })


def profile_data(path: str) -> str:
    """Data quality: missing values, duplicates, cardinality, numeric summary."""
    df = _load(path)
    numeric = df.select_dtypes(include=np.number)

    return json.dumps({
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_per_column": {c: int(df[c].isna().sum()) for c in df.columns},
        "unique_per_column": {c: int(df[c].nunique()) for c in df.columns},
        "numeric_summary": json.loads(numeric.describe().round(2).to_json()) if not numeric.empty else {},
    })


def value_counts(path: str, column: str, top_n: int = 10) -> str:
    """Frequency of each distinct value in a column."""
    df = _load(path)
    _check(df, column)
    counts = df[column].value_counts().head(min(top_n, MAX_GROUPS))
    return json.dumps({
        "column": column,
        "distinct_values": int(df[column].nunique()),
        "counts": {str(k): int(v) for k, v in counts.items()},
    })


# ---------------------------------------------------------------- analysis

def group_aggregate(path: str, group_by: str, value_column: str, how: str) -> str:
    """Group by one column, aggregate another."""
    df = _load(path)
    _check(df, group_by)
    _check(df, value_column)
    if how not in {"sum", "mean", "count", "max", "min", "median", "std"}:
        raise ValueError(f"Unsupported aggregation '{how}'.")

    result = getattr(df.groupby(group_by)[value_column], how)()
    result = result.sort_values(ascending=False).round(2)
    truncated = len(result) > MAX_GROUPS

    return json.dumps({
        "group_by": group_by, "value_column": value_column, "how": how,
        "groups_total": int(len(result)),
        "truncated": truncated,
        "result": {str(k): float(v) for k, v in result.head(MAX_GROUPS).items()},
    })


def filter_aggregate(path: str, filter_column: str, operator: str, filter_value: str,
                     value_column: str = "", how: str = "count") -> str:
    """Filter rows by one condition, then aggregate. Returns the row count too."""
    df = _load(path)
    _check(df, filter_column, "filter column")

    ops = {"==": "eq", "!=": "ne", ">": "gt", ">=": "ge", "<": "lt", "<=": "le"}
    if operator not in ops:
        raise ValueError(f"Unsupported operator '{operator}'. Use one of {list(ops)}.")

    col = df[filter_column]
    if pd.api.types.is_numeric_dtype(col):
        try:
            value = float(filter_value)
        except ValueError:
            raise ValueError(f"Column '{filter_column}' is numeric but '{filter_value}' is not a number.")
    else:
        value = filter_value

    mask = getattr(col, ops[operator])(value)
    sub = df[mask]

    out = {"filter": f"{filter_column} {operator} {filter_value}",
           "matching_rows": int(len(sub)),
           "total_rows": int(len(df))}

    if value_column and how != "count":
        _check(df, value_column, "value column")
        if sub.empty:
            out["result"] = None
            out["note"] = "No rows matched, nothing to aggregate."
        else:
            out.update({"value_column": value_column, "how": how,
                        "result": round(float(getattr(sub[value_column], how)()), 2)})
    return json.dumps(out)


def time_series(path: str, date_column: str, value_column: str,
                freq: str = "M", how: str = "sum") -> str:
    """Aggregate a numeric column over time periods."""
    df = _load(path)
    _check(df, date_column, "date column")
    _check(df, value_column, "value column")

    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        raise ValueError(f"Column '{date_column}' is not a date. Its type is {df[date_column].dtype}.")
    if freq not in {"D", "W", "M", "Q", "Y"}:
        raise ValueError(f"Unsupported frequency '{freq}'. Use D, W, M, Q or Y.")

    s = df.set_index(date_column)[value_column].resample(freq)
    result = getattr(s, how)().round(2).dropna()

    return json.dumps({
        "date_column": date_column, "value_column": value_column,
        "freq": freq, "how": how, "periods": int(len(result)),
        "result": {str(k.date()): float(v) for k, v in result.head(MAX_GROUPS).items()},
    })


def correlation(path: str) -> str:
    """Pearson correlation between all numeric columns."""
    df = _load(path).select_dtypes(include=np.number)
    if df.shape[1] < 2:
        raise ValueError(f"Need at least two numeric columns, found {df.shape[1]}.")
    return json.dumps({"method": "pearson",
                       "matrix": json.loads(df.corr().round(3).to_json())})


def detect_anomalies(path: str, column: str, method: str = "iqr", top_n: int = 10) -> str:
    """Flag statistical outliers. An outlier is unusual, not necessarily fraudulent."""
    df = _load(path)
    _check(df, column)
    s = df[column]
    if not pd.api.types.is_numeric_dtype(s):
        raise ValueError(f"Column '{column}' is not numeric (type {s.dtype}).")

    if method == "iqr":
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < low) | (s > high)
        bounds = {"lower": round(float(low), 2), "upper": round(float(high), 2)}
    elif method == "zscore":
        z = (s - s.mean()) / s.std()
        mask = z.abs() > 3
        bounds = {"threshold_sd": 3, "mean": round(float(s.mean()), 2),
                  "std": round(float(s.std()), 2)}
    else:
        raise ValueError(f"Unsupported method '{method}'. Use 'iqr' or 'zscore'.")

    out = df[mask].nlargest(min(top_n, 20), column)
    return json.dumps({
        "column": column, "method": method, "bounds": bounds,
        "outlier_count": int(mask.sum()),
        "percent_of_rows": round(100 * float(mask.mean()), 2),
        "caveat": "Statistical outliers only. Not evidence of fraud or error.",
        "examples": json.loads(out.head(top_n).to_json(orient="records", date_format="iso")),
    })


REGISTRY = {
    "describe_columns": describe_columns,
    "profile_data": profile_data,
    "value_counts": value_counts,
    "group_aggregate": group_aggregate,
    "filter_aggregate": filter_aggregate,
    "time_series": time_series,
    "correlation": correlation,
    "detect_anomalies": detect_anomalies,
}


def _fn(name, desc, props, required):
    return {"type": "function",
            "function": {"name": name, "description": desc,
                         "parameters": {"type": "object", "properties": props,
                                        "required": required}}}

_PATH = {"type": "string", "description": "Path to the data file."}
_HOW = {"type": "string", "enum": ["sum", "mean", "count", "max", "min", "median", "std"],
        "description": "Aggregation operation."}

SCHEMAS = [
    _fn("describe_columns",
        "List every column name and its data type, plus the row count. Call this FIRST when you do not know the exact column names.",
        {"path": _PATH}, ["path"]),

    _fn("profile_data",
        "Data quality report: missing values per column, duplicate rows, distinct value counts, and min/max/mean/std for numeric columns. Use for 'describe the data', 'is the data clean', 'what is in this file'.",
        {"path": _PATH}, ["path"]),

    _fn("value_counts",
        "How many rows fall into each distinct value of one column. Use for 'how many per category', 'what values exist', 'which is most common'. Does not aggregate a second column.",
        {"path": _PATH,
         "column": {"type": "string", "description": "Exact column name to count."},
         "top_n": {"type": "integer", "description": "How many values to return. Default 10."}},
        ["path", "column"]),

    _fn("group_aggregate",
        "Group rows by one column and aggregate a numeric column. Use for totals, averages, maxima per category.",
        {"path": _PATH,
         "group_by": {"type": "string", "description": "Exact name of the column to group by."},
         "value_column": {"type": "string", "description": "Exact name of the numeric column to aggregate."},
         "how": _HOW},
        ["path", "group_by", "value_column", "how"]),

    _fn("filter_aggregate",
        "Keep only rows matching one condition, then count them or aggregate a numeric column. Use for 'transactions above 500', 'how many in France', 'average amount for Debit rows'.",
        {"path": _PATH,
         "filter_column": {"type": "string", "description": "Exact column name to filter on."},
         "operator": {"type": "string", "enum": ["==", "!=", ">", ">=", "<", "<="],
                      "description": "Comparison operator."},
         "filter_value": {"type": "string", "description": "Value to compare against, as a string."},
         "value_column": {"type": "string", "description": "Numeric column to aggregate. Omit to only count rows."},
         "how": _HOW},
        ["path", "filter_column", "operator", "filter_value"]),

    _fn("time_series",
        "Aggregate a numeric column over time periods. Use for monthly trends, growth over time, 'which month was highest'.",
        {"path": _PATH,
         "date_column": {"type": "string", "description": "Exact name of the date column."},
         "value_column": {"type": "string", "description": "Exact name of the numeric column."},
         "freq": {"type": "string", "enum": ["D", "W", "M", "Q", "Y"],
                  "description": "Period: daily, weekly, monthly, quarterly, yearly."},
         "how": _HOW},
        ["path", "date_column", "value_column", "freq", "how"]),

    _fn("correlation",
        "Pearson correlation between every pair of numeric columns. Use for 'are X and Y related', 'what drives Z'.",
        {"path": _PATH}, ["path"]),

    _fn("detect_anomalies",
        "Find statistical outliers in a numeric column using IQR or z-score. Returns the count and example rows. Outliers are unusual values, not proof of fraud.",
        {"path": _PATH,
         "column": {"type": "string", "description": "Exact numeric column to check."},
         "method": {"type": "string", "enum": ["iqr", "zscore"],
                    "description": "iqr is robust to skew and suits financial amounts. zscore assumes roughly normal data."},
         "top_n": {"type": "integer", "description": "How many example rows. Default 10."}},
        ["path", "column", "method"]),
]
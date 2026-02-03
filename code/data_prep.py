import io
import os
from typing import Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def scrape_data_from_webpage(page_url: str) -> pd.DataFrame:
    print(f"Scraping data from webpage (BeautifulSoup): {page_url}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(page_url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "lxml")
    tables = soup.find_all("table")

    df = pd.read_html(io.StringIO(str(tables[0])))[0]
    #if header row was <td> not <th>, pandas uses first row as data → columns are 0,1,2,...
    if "age_at_death" not in df.columns and len(df) > 0:
        first_row = df.iloc[0].astype(str).str.strip().str.lower()
        if "age_at_death" in first_row.values or "id" in first_row.values:
            df.columns = [str(c).strip() for c in df.iloc[0]]
            df = df.iloc[1:].reset_index(drop=True)
            numeric_names = ["avg_work_hours_per_day", "avg_rest_hours_per_day", "avg_sleep_hours_per_day", "avg_exercise_hours_per_day", "age_at_death"]
            for col in numeric_names:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
    if len(df.columns) > 0 and (df.columns[0] in (0, "0", "Unnamed: 0") or str(df.columns[0]).startswith("Unnamed")):
        df = df.drop(columns=df.columns[0], errors="ignore")
    print(f"Scraped table: {len(df)} rows, {len(df.columns)} columns")
    return df


def clean_dataset(df: pd.DataFrame, drop_outliers: bool = True) -> pd.DataFrame:
    df = df.copy()
    
    #drop id column
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    
    #drop rows with missing target
    df = df.dropna(subset=["age_at_death"])
    
    #drop outliers using IQR method
    if drop_outliers:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == "age_at_death":
                continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df


def normalize_dataset(df: pd.DataFrame, target_col: str = "age_at_death") -> Tuple[pd.DataFrame, StandardScaler]:
    
    df = df.copy()
    
    #separate target and features
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df
    
    #normalize only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    #recombine if target exists
    if y is not None:
        df = pd.concat([X, y], axis=1)
    else:
        df = X
    
    return df, scaler


def prepare_data_for_pdf(
    data_url: str,
    output_dir: str = "data",
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    #create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    #load and clean data
    print("1. Loading and cleaning data...")
    df_raw = scrape_data_from_webpage(data_url)
    df_clean = clean_dataset(df_raw, drop_outliers=True)
    print(f"   Cleaned dataset: {len(df_clean)} rows")

    #encode (derived features + one-hot) and split
    print("2. Encoding and splitting (saving encoded datasets)...")
    train_encoded, test_encoded, feature_names = _encode_and_split(
        df_clean, test_size=test_size, random_state=random_state
    )
    encoded_train_path = os.path.join(output_dir, "training_data_encoded.csv")
    encoded_test_path = os.path.join(output_dir, "test_data_encoded.csv")
    feature_names_path = os.path.join(output_dir, "feature_names.csv")
    train_encoded.to_csv(encoded_train_path, index=False)
    test_encoded.to_csv(encoded_test_path, index=False)
    pd.Series(feature_names.tolist()).to_csv(feature_names_path, index=False, header=False)
    print(f"   Saved {encoded_train_path} ({len(train_encoded)} rows)")
    print(f"   Saved {encoded_test_path} ({len(test_encoded)} rows)")
    print(f"   Saved {feature_names_path} ({len(feature_names)} features)")

    #normalize encoded data and save
    print("3. Normalizing data...")
    scaler = StandardScaler()
    X_train = train_encoded.drop(columns=["age_at_death"])
    X_test = test_encoded.drop(columns=["age_at_death"])
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    train_df = pd.DataFrame(X_train_norm, columns=X_train.columns)
    train_df["age_at_death"] = train_encoded["age_at_death"].values
    test_df = pd.DataFrame(X_test_norm, columns=X_test.columns)
    test_df["age_at_death"] = test_encoded["age_at_death"].values
    df_normalized = pd.concat([train_df, test_df], ignore_index=True)
    joint_data_path = os.path.join(output_dir, "joint_data_collection.csv")
    df_normalized.to_csv(joint_data_path, index=False)
    training_data_path = os.path.join(output_dir, "training_data.csv")
    test_data_path = os.path.join(output_dir, "test_data.csv")
    train_df.to_csv(training_data_path, index=False)
    test_df.to_csv(test_data_path, index=False)
    print(f"4. Saved joint_data_collection.csv, training_data.csv, test_data.csv")

    print("5. Creating activation_data.csv...")
    activation_df = test_encoded.iloc[[0]]
    activation_data_path = os.path.join(output_dir, "activation_data.csv")
    activation_df.to_csv(activation_data_path, index=False)
    print(f"   Saved activation_data.csv (1 row)")

    return {
        "joint_data_path": joint_data_path,
        "training_data_path": training_data_path,
        "test_data_path": test_data_path,
        "activation_data_path": activation_data_path,
        "encoded_train_path": encoded_train_path,
        "encoded_test_path": encoded_test_path,
        "feature_names_path": feature_names_path,
        "scaler": scaler,
        "train_df": train_df,
        "test_df": test_df,
        "activation_df": activation_df,
    }


def _encode_and_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """derived features, one-hot encode, split"""
    y = df["age_at_death"]
    X = df.drop(columns=["age_at_death"])

    hour_cols = ["avg_work_hours_per_day", "avg_rest_hours_per_day", "avg_sleep_hours_per_day", "avg_exercise_hours_per_day"]
    if all(c in X.columns for c in hour_cols):
        X = X.copy()
        w = X["avg_work_hours_per_day"]
        r = X["avg_rest_hours_per_day"]
        s = X["avg_sleep_hours_per_day"]
        e = X["avg_exercise_hours_per_day"]
        X["total_hours_per_day"] = w + r + s + e
        X["active_hours"] = w + e
        X["rest_sleep_ratio"] = r / (s + 1e-6)
        X["exercise_share"] = e / (w + e + 1e-6)

    X_encoded = pd.get_dummies(X, drop_first=True)
    feature_names = X_encoded.columns
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state
    )
    train_encoded = X_train.copy()
    train_encoded["age_at_death"] = y_train.values
    test_encoded = X_test.copy()
    test_encoded["age_at_death"] = y_test.values
    return train_encoded, test_encoded, feature_names


def load_encoded_datasets(
    encoded_train_path: str = "data/training_data_encoded.csv",
    encoded_test_path: str = "data/test_data_encoded.csv",
    feature_names_path: str = "data/feature_names.csv",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """load encoded train/test datasets and feature names"""
    train_df = pd.read_csv(encoded_train_path)
    test_df = pd.read_csv(encoded_test_path)
    fn_df = pd.read_csv(feature_names_path, header=None)
    feature_names = pd.Index(fn_df.iloc[:, 0].astype(str).tolist())
    X_train = train_df.drop(columns=["age_at_death"]).values
    X_test = test_df.drop(columns=["age_at_death"]).values
    y_train = train_df["age_at_death"].values
    y_test = test_df["age_at_death"].values
    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import DATASET_URL
    results = prepare_data_for_pdf(DATASET_URL)
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nCreated files:")
    print(f"  - {results['encoded_train_path']}")
    print(f"  - {results['encoded_test_path']}")
    print(f"  - {results['feature_names_path']}")
    print(f"  - {results['joint_data_path']}")
    print(f"  - {results['training_data_path']}")
    print(f"  - {results['test_data_path']}")
    print(f"  - {results['activation_data_path']}")


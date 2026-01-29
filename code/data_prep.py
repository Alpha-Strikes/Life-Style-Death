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
    
    #normalize data
    print("2. Normalizing data...")
    df_normalized, scaler = normalize_dataset(df_clean)
    
    #save joint_data_collection.csv
    joint_data_path = os.path.join(output_dir, "joint_data_collection.csv")
    df_normalized.to_csv(joint_data_path, index=False)
    print(f"3. Saved joint_data_collection.csv: {joint_data_path}")
    
    #split into train/test (80/20)
    print("4. Splitting into train/test sets...")
    train_df, test_df = train_test_split(
        df_normalized,
        test_size=test_size,
        random_state=random_state
    )
    
    #save training_data.csv (80%)
    training_data_path = os.path.join(output_dir, "training_data.csv")
    train_df.to_csv(training_data_path, index=False)
    print(f"   Saved training_data.csv: {training_data_path} ({len(train_df)} rows, {len(train_df)/len(df_normalized)*100:.1f}%)")
    
    #save test_data.csv (20%)
    test_data_path = os.path.join(output_dir, "test_data.csv")
    test_df.to_csv(test_data_path, index=False)
    print(f"   Saved test_data.csv: {test_data_path} ({len(test_df)} rows, {len(test_df)/len(df_normalized)*100:.1f}%)")
    
    #create activation_data.csv (1 entry from test)
    print("5. Creating activation_data.csv...")
    activation_df = test_df.iloc[[0]]  # Take first entry from test set
    activation_data_path = os.path.join(output_dir, "activation_data.csv")
    activation_df.to_csv(activation_data_path, index=False)
    print(f"   Saved activation_data.csv: {activation_data_path} (1 row)")
    
    return {
        "joint_data_path": joint_data_path,
        "training_data_path": training_data_path,
        "test_data_path": test_data_path,
        "activation_data_path": activation_data_path,
        "scaler": scaler,
        "train_df": train_df,
        "test_df": test_df,
        "activation_df": activation_df,
    }


def train_test_split_encoded(df:pd.DataFrame, test_size:float= 0.2, random_state:int =42)->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    df=clean_dataset(df)

    y=df["age_at_death"].values
    X=df.drop(columns=["age_at_death"])

    #derived lifestyle features from the data we have
    hour_cols = ["avg_work_hours_per_day", "avg_rest_hours_per_day", "avg_sleep_hours_per_day", "avg_exercise_hours_per_day"]
    if all(c in X.columns for c in hour_cols):
        X = X.copy()
        w, r, s, e = X["avg_work_hours_per_day"], X["avg_rest_hours_per_day"], X["avg_sleep_hours_per_day"], X["avg_exercise_hours_per_day"]
        X["total_hours_per_day"] = w + r + s + e
        X["active_hours"] = w + e  #work + exercise
        X["rest_sleep_ratio"] = r / (s + 1e-6)  #to avoid division by zero
        X["exercise_share"] = e / (w + e + 1e-6)  #exercise share of active time

    # one-hot encode gender, occupation_type (and any other object columns)
    X_encoded= pd.get_dummies(X, drop_first=True)
    feature_names=X_encoded.columns
    X_train, X_test, y_train, y_test = train_test_split(X_encoded.values, y, test_size=test_size, random_state=random_state)

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
    print(f"  - {results['joint_data_path']}")
    print(f"  - {results['training_data_path']}")
    print(f"  - {results['test_data_path']}")
    print(f"  - {results['activation_data_path']}")


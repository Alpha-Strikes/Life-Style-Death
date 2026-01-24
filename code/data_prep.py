import os
import io
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_raw_data(data_url: str) -> pd.DataFrame:
    print(f"Downloading dataset from: {data_url}")
    response = requests.get(data_url)
    response.raise_for_status()#error if download failed
    df = pd.read_csv(io.StringIO(response.text))#convert response to a dataframe
    print(f"Dataset loaded successfully ({len(df)} rows)")
    
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
    df_raw = load_raw_data(data_url)
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


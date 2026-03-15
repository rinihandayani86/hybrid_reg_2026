import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# =====================================================
# LOAD + SPLIT PIPELINE
# =====================================================

def load_and_split_dataset(dataset_name,
                           train_size=0.7,
                           val_size=0.1,
                           test_size=0.2,
                           random_state=42):

    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "Train + Val + Test must sum to 1.0"

    features = ['MQ3', 'TGS822', 'TGS2602', 'MQ5', 'MQ138', 'TGS2620']

    # =========================
    # LOAD DATASET
    # =========================
    if dataset_name == "Crop_1":
        file_path = "C:/Users/RHY/Downloads/Publikasi - PeerJ/Dataset_teh 20230407 REVISI v4.xlsx"
        df = pd.read_excel(file_path, sheet_name='Sheet1')

    elif dataset_name == "Crop_2":
        file_path = "C:/Users/RHY/Downloads/Publikasi - PeerJ/Dataset_TehHijauGambung_20250516_MODIFIED.csv"
        df = pd.read_csv(file_path)

    else:
        raise ValueError("Unknown dataset name")

    X = df[features].values
    y_reg = df['Aroma'].values

    # =========================
    # Stratification based on TASTER CLASS
    # =========================
    if 'Class' in df.columns:
        y_split = df['Class'].values
    else:
        raise ValueError("Class column required for stratified split")

    # =========================
    # FIRST SPLIT (Train+Val vs Test)
    # =========================
    X_temp, X_test, y_reg_temp, y_reg_test, y_split_temp, y_split_test = train_test_split(
        X, y_reg, y_split,
        test_size=test_size,
        stratify=y_split,
        random_state=random_state
    )

    # =========================
    # SECOND SPLIT (Train vs Val)
    # =========================
    val_ratio_adjusted = val_size / (train_size + val_size)

    X_train, X_val, y_reg_train, y_reg_val, y_split_train, y_split_val = train_test_split(
        X_temp, y_reg_temp, y_split_temp,
        test_size=val_ratio_adjusted,
        stratify=y_split_temp,
        random_state=random_state
    )

    return {
        "train": {
            "X": X_train,
            "y_reg": y_reg_train
        },
        "val": {
            "X": X_val,
            "y_reg": y_reg_val
        },
        "test": {
            "X": X_test,
            "y_reg": y_reg_test
        }
    }
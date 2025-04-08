import os
from typing import List

import numpy as np
import pandas as pd


def find_lostart_csv(id: int):
    """
    Find the csv file containing the Lost Art ID and return the corresponding dataframe.
    """
    for csv in os.listdir("data/lostart"):
        df = pd.read_csv(f"data/lostart/{csv}", sep=";")

        if df.loc[df["Lost Art ID"] == id].shape[0] > 0:
            return csv, df.loc[df["Lost Art ID"] == id]

    raise ValueError(f"Lost Art ID {id} not found in any csv file.")
    
    

def find_lostart_csvs(ids: List[int]):
    """
    Find the csv files containing the Lost Art IDs and return one dataframe combining all
    the corresponding dataframes, in the same order as the ids.
    If one id is not found, a warning is printed and the corresponding id is skipped.
    """

    dfs = np.array([None] * len(ids))

    for csv in os.listdir("data/lostart"):
        df = pd.read_csv(f"data/lostart/{csv}", sep=";")

        for idx, id in enumerate(ids):
            if df.loc[df["Lost Art ID"] == id].shape[0] > 0:
                dfs[idx] = df.loc[df["Lost Art ID"] == id]
    
    for idx, id in enumerate(ids):
        if dfs[idx] is None:
            print(f"Lost Art ID {id} not found in any csv file. Skipping.")
    return pd.concat(dfs)


def find_pop(id: str):
    """
    Find the pop line containing the Lost Art ID and return the corresponding dataframe.
    If the pop line is not found, return None.
    """
    df = pd.read_excel("../data/mnr_20250303_17h40m54s.ods")

    try :
        df.loc[df["REF"] == id].shape[0] > 0
        return df.loc[df["REF"] == id]
    except:
        return None

def remove_leakage(df: pd.DataFrame):
    """
    Remove leakage columns from the dataframe.
    """
    return df.drop(columns=["Inventarnummer/Signatur", "Provenienz", "Literatur / Quelle"])

def keep_necessary_columns_la(df: pd.DataFrame):
    # to_keep = ["Lost Art ID", "Hersteller/Künstler/Autor:in", "Titel", "Datierung", "Objektart", "Beschreibung"]
    to_keep = ["Lost Art ID", "Hersteller/Künstler/Autor:in", "Titel", "Beschreibung",]
    return df[to_keep]

def keep_necessary_columns_mnr(df: pd.DataFrame):
    # to_keep = ['AATT', 'ATIT', 'AUTR', 'BIBL', 'CONTIENT_IMAGE', 'DMAJ', 'DOMN',
    #    'HIST', 'HIST3', 'LOCA', 'MARQ', 'MILL', 'REPR', 'SCLE', 'TITR']
    to_keep = ["REF", "AUTR", "TITR", "REPR"]
    df["AUTR"] = df["AUTR"].fillna("")
    df["TITR"] = df["TITR"].fillna("")
    df["REPR"] = df["REPR"].fillna("")
    df["DESC"] = df["DESC"].fillna("")

    # We concatenate REPR and DESC into one column
    df["REPR"] = df["REPR"].astype(str) + " " + df["DESC"].astype(str)
    return df[to_keep]


def get_concatenated_txt(series: pd.Series, keep_columns: bool = False):
    """
    Concatenate the non-NaN values of a series into a string.
    """
    result = ""
    
    for idx, value in series.items():
        if pd.notna(value):
            if keep_columns:
                result += f"{idx}: {value} "
            else:
                result += f"{value} "
            result += "\n"
        else:
            if keep_columns:
                result += f"{idx}: "
            else:
                result += " "
            result += "\n"

    return result

def remove_leakage_mnr(df: pd.DataFrame):
    """
    Remove leakage columns from the dataframe.
    """
    leakage_cols = ["HIST4", "LOCA", "NOTE"]
    df = df.drop(columns=leakage_cols)

    unecessary_cols = ["POP_IMPORT", "VIDEO"]
    df = df.drop(columns=unecessary_cols)
    return df

def add_column_with_concatenated_txt(df: pd.DataFrame):
    """
    Add a new column to the dataframe containing the concatenated text of the other columns.
    """
    df["CONCATENATED"] = df.apply(lambda row: get_concatenated_txt(row), axis=1)
    return df

import os
from typing import List

import pandas as pd
import requests
from tqdm import tqdm


def download_lostart_csv(download_dir: str = "../data/lostart", verbose: bool = False):
    """
    Download the lostart.csv files from the given URL and save them to the specified folder.
    Progress is displayed using tqdm.

    Parameters
    ----------
    download_dir : str
        Directory where the CSV files will be saved.
    verbose : bool
        Whether to print detailed download messages.
    """
    os.makedirs(download_dir, exist_ok=True)

    start = 0
    base_url = "https://www.lostart.de/de/search-export/csv?start={start}&filter%5Btype%5D%5B0%5D=Objektdaten"

    pbar = tqdm(desc="Downloading CSV files", unit="file")

    while True:
        csv_url = base_url.format(start=start)
        csv_path = os.path.join(download_dir, f"lostart_start={start}.csv")

        try:
            response = requests.get(csv_url)
            response.raise_for_status()

            if len(response.text.strip()) < 200:
                if verbose:
                    print("No more data available.")
                break

            if os.path.exists(csv_path):
                if verbose:
                    print(f"File already exists: {csv_path}")
                start += 500
                pbar.update(1)
                continue

            with open(csv_path, 'wb') as f:
                f.write(response.content)

            if verbose:
                print(f"CSV file downloaded and saved as {csv_path}")

        except requests.RequestException as e:
            if verbose:
                print(f"Failed to download CSV file: {e}")
            break

        start += 500
        pbar.update(1)

    pbar.close()


def find_lostart_csvs(ids: List[int]):
    """
    Find the csv files containing the Lost Art IDs and return one dataframe combining all
    the corresponding dataframes, in the same order as the ids.
    If one id is not found, a warning is printed and the corresponding id is skipped.
    """

    dfs = [None] * len(ids)
    csvs = []

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(base_dir, "data", "lostart")

    for csv in os.listdir(data_dir):
        df = pd.read_csv(os.path.join(data_dir, csv), sep=";", on_bad_lines='skip')

        for idx, id in enumerate(ids):
            if df.loc[df["Lost Art ID"] == id].shape[0] > 0:
                dfs[idx] = df.loc[df["Lost Art ID"] == id]
                csvs.append(csv)
    
    for idx, id in enumerate(ids):
        if dfs[idx] is None:
            print(f"Lost Art ID {id} not found in any csv file. Skipping.")
    
    if sum(x is None for x in dfs) == len(dfs):
        return None

    return pd.concat(dfs), csvs


def find_lostart_csv(id: int):
    """
    Find the csv file containing the Lost Art ID and return the corresponding dataframe.
    """
    return find_lostart_csvs([id])


def find_pop(id: str):
    """
    Find the pop line containing the MNR ID and return the corresponding dataframe.
    If the pop line is not found, return None.
    """
    df = pd.read_excel("data/mnr_20250303.ods")

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
    to_keep = ["Lost Art ID", "Hersteller/Künstler/Autor:in", "Titel", "Beschreibung"]
    df["Lost Art ID"] = df["Lost Art ID"].fillna("")
    df["Hersteller/Künstler/Autor:in"] = df["Hersteller/Künstler/Autor:in"].fillna("")
    df["Titel"] = df["Titel"].fillna("")
    df["Beschreibung"] = df["Beschreibung"].fillna("")
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


if __name__ == "__main__":
    found = [589707, 589708, 614072, 526702, 567247, 429210, 310418, 600027, 323038]
    
    df = find_lostart_csvs(found)
    print(df)

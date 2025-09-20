import pandas as pd
import os


def load_data(arquivo: str) -> pd.DataFrame:
    current_path = os.path.dirname(__file__)
    file_path = os.path.join(current_path, '../..', 'data/raw/', arquivo)
    mysep = ";"
    mydec = ","
    myenc = "utf-8-sig"
    return pd.read_csv(file_path,
                       na_values=['na'],
                       sep=mysep,
                       decimal=mydec,
                       encoding=myenc)

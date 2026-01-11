import pandas as pd
from typing import List

def merge_dataframes(dfs: List[pd.DataFrame]):
    return pd.concat(dfs, ignore_index=True)
import pandas as pd
from data.ensure import german_data_as_df, polish_data_as_df, belgium_data_as_df
from data.merge import merge_dataframes
from data.balance import balance_dataframe


if __name__=="__main__":
    traing, _, _ = german_data_as_df()
    trainp, _, _ = polish_data_as_df()
    trainb, _, _ = belgium_data_as_df()
    dfs = [traing, trainp, trainb]
    merged = merge_dataframes(dfs)
    print(len(merged))
    balanced = balance_dataframe(merged)
    print(len(balanced))

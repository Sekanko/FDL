import pandas as pd

def balance_dataframe(df, column='ClassId'):
    max_size = df[column].value_counts().max()
    lst = [df]
    for class_index, group in df.groupby(column):
        if len(group) < max_size:
            count_diff = max_size - len(group)
            upsampled_group = group.sample(count_diff, replace=True, random_state=50)
            lst.append(upsampled_group)

    frame_new = pd.concat(lst)
    return frame_new.sample(frac=1, random_state=50).reset_index(drop=True)


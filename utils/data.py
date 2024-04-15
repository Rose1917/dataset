from pandas import DataFrame
def random_split(df: DataFrame, N: int):
    df = df[df['is_negative'] != "True"]
    random_selected = df.sample(n=N)
    remaining = df.drop(random_selected.index)
    return random_selected, remaining

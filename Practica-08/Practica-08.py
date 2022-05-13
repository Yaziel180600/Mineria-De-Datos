from cProfile import label
from unicodedata import category
import pandas as pd
import re
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import numpy as np
from functools import reduce
from scipy.stats import mode

def drop_column(df, column_names) -> pd.DataFrame:
    df = df.drop(column_names, axis = 1)
    return df

def transform_cylinders(str_cylinders:str) -> int:
    Cil = re.findall(r'[0-9]+',str(str_cylinders))
    if not Cil : 
        Cil=0 
    else: 
        Cil=Cil[0]
    return int(Cil)

def get_df() -> pd.DataFrame:
    df = pd.read_csv(R".\vehicles.csv")
    df = drop_column(df, ["url","region_url","image_url","title_status"])
    df["posting_date"] = pd.to_datetime(df["posting_date"], format = '%Y-%m-%dT%H:%M:%S-%f')
    df['cylinders'] = df['cylinders'].apply(transform_cylinders)
    df["year"] = df["year"].fillna(0).astype(int)
    
    return df

def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def scatter_group_by(
    file_path: str, df: pd.DataFrame, x_column: str, y_column: str, label_column: str
):
    fig, ax = plt.subplots()
    labels = pd.unique(df[label_column])
    cmap = get_cmap(len(labels) + 1)
    for i, label in enumerate(labels):
        filter_df = df.query(f'{label_column} == "{label}"')
        ax.scatter(filter_df[x_column], filter_df[y_column], label=label, color=cmap(i))
    ax.legend(prop={'size': 5})
    ax.ticklabel_format(style='plain')
    plt.savefig(file_path)
    plt.close()


def euclidean_distance(p_1: np.array, p_2: np.array) -> float:
    return np.sqrt(np.sum((p_2 - p_1) ** 2))


def k_nearest_neightbors(
    points: List[np.array], labels: np.array, input_data: List[np.array], k: int
):
    input_distances = [
        [euclidean_distance(input_point, point) for point in points]
        for input_point in input_data
    ]
    points_k_nearest = [
        np.argsort(input_point_dist)[:k] for input_point_dist in input_distances
    ]
    return [
        mode([labels[index] for index in point_nearest])
        for point_nearest in points_k_nearest
    ]


df = get_df()
df = df[["price", "odometer","manufacturer"]]
points = list(df[["price", "odometer"]].to_records(index=False))
labels = df["manufacturer"].tolist()

df_filter = df["price"] < 500000
df = df[df_filter]

df_filter = df["odometer"] < 500000
df = df[df_filter]

ford = df["manufacturer"] == "ford"
jeep = df["manufacturer"] == "jeep"
honda = df["manufacturer"] == "honda"
manufacturers = ford | jeep | honda

df=df[manufacturers]
scatter_group_by("Practica-08/plots/knn.png", df, "price", "odometer", "manufacturer")

list_t = [
    (np.array(tuples[0:1]), tuples[2])
    for tuples in df.itertuples(index=False, name=None)
]
points = [point for point, _ in list_t]
labels = [label for _, label in list_t]

kn = k_nearest_neightbors(
    points,
    labels,
    [np.array([100, 150]), np.array([3, 7]), np.array([10000, 30000]), np.array([8000, 4000])],
    5,
)
print(kn)
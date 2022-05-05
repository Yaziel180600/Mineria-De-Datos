import pandas as pd
import re
import matplotlib.pyplot as plt

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

df = get_df()

group_filter_year = df.groupby(df["year"])

print("Suma de precio de carros por año: ")
df_plot = group_filter_year.sum()["price"]
df_plot.drop(0, inplace = True, axis = 0)
df_plot = df_plot[df_plot > 0]
print(df_plot)


print("La cantidad de autos por marca ")
df_plot = df["manufacturer"].value_counts()
print(df_plot)

import statsmodels.api as sm
from statsmodels.formula.api import ols
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

grp_manu_and_year = df.groupby(["manufacturer", "year"])
total_od_cy = grp_manu_and_year[["odometer", "cylinders"]].sum()
total_od_cy.reset_index(inplace=True)
total_od_cy.drop("year", inplace=True, axis=1)

manu_x_views = total_od_cy[["manufacturer", "odometer"]]

model = ols("odometer ~ manufacturer", data=manu_x_views).fit()
df_anova = sm.stats.anova_lm(model, typ=2)

if df_anova["PR(>F)"][0] < 0.005:
    print("Si hay diferencias")
    print(df_anova)
else:
    print("No hay diferencias")
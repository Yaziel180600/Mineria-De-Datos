import statsmodels.api as sm
import pandas as pd
import re
import matplotlib.pyplot as plt
import numbers

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

def transform_variable(df: pd.DataFrame, x: str) -> pd.Series:
    if isinstance(df[x][0], numbers.Number):
        return df[x]  # type: pd.Series
    else:
        return pd.Series([i for i in range(0, len(df[x]))])


def linear_regression(df: pd.DataFrame, x, y) -> None:
    fixed_x = transform_variable(df, x)
    model = sm.OLS(df[y], sm.add_constant(fixed_x)).fit()
    print(model.summary())
    html = model.summary().tables[1].as_html()
    coef = pd.read_html(html, header=0, index_col=0)[0]["coef"]
    df.plot(x=x, y=y, kind="scatter")
    plt.plot(df[x], [pd.DataFrame.mean(df[y]) for _ in fixed_x.items()], color="green")
    plt.plot(
        df[x],
        [coef.values[1] * x + coef.values[0] for _, x in fixed_x.items()],
        color="red",
    )
    plt.xticks(rotation=90)
    plt.savefig(f"Practica-06/plots/lr_{y}_{x}.png")
    plt.close()

df = get_df()

df_peaks = df["price"] >= 37369287 #mediana de precio
grp_year = df[df_peaks].groupby(df["year"])
grp_year = grp_year.mean()

grp_year.drop("year", inplace=True, axis=1)
grp_year.reset_index(inplace=True)
grp_year.drop("year", inplace=True, axis=1)

linear_regression(grp_year, "cylinders", "price")
linear_regression(grp_year, "odometer", "price")
linear_regression(grp_year, "cylinders", "odometer")
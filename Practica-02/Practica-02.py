
import pandas as pd,re

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

df = pd.read_csv(R".\vehicles.csv")

df = drop_column(df, ["url","region_url","image_url","title_status"])

print(df.dtypes)

df["posting_date"] = pd.to_datetime(df["posting_date"], format = '%Y-%m-%dT%H:%M:%S-%f')
df['cylinders'] = df['cylinders'].apply(transform_cylinders)

print()
print(df.dtypes)


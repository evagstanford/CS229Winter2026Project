import pandas as pd 
import json
from sklearn.preprocessing import StandardScaler
print("RUNNING:", __file__)


input_csv = "/Users/evageierstanger/CS229FinalProject/Preprocess/san_mateo_cut_cols.csv"
output_path = "/Users/evageierstanger/CS229FinalProject/Preprocess"
map_path = "/Users/evageierstanger/CS229FinalProject/Preprocess/category_maps.json"
scaler = StandardScaler()

bldg_ids_to_Drop = {
    # outliers
    536576, 141113, 150396, 366539, 522048, 124296, 335080, 408686, 202027, 332073, 443912
}


target_cols = {
    'out.electricity.net.energy_savings..kwh',
    'out.utility_bills.total_bill_savings..usd',
    'out.energy_burden_savings..percentage',
    'out.emissions_reduction.total.aer_mid_case_avg..co2e_kg'}

data = pd.read_csv(input_csv, low_memory=False)

# remove outlier bldg ids
data = data[~data['bldg_id'].isin(list(bldg_ids_to_Drop))]

cleaned = data.copy()
allmaps = {} # for maps value -> category when needed

# CONVERT ALL DATA TO FLOAT NUMERIC
for col in cleaned.columns:
    if col.lower() == "bldg_id" or col.lower() == "in.representative_income":
        continue
    elif col in target_cols: 
        continue
    stripped = cleaned[col].astype(str).str.strip().str.lower()
    stripped = stripped.replace(["none", "nan", "na", "n/a", "unknown", "<null>", ""], pd.NA)
    try_numeric = pd.to_numeric(stripped, errors="coerce")
    # simply numeric column 
    if try_numeric.notna().sum() == stripped.notna().sum():
        cleaned[col] = try_numeric
        continue
    # ranges
    if stripped.str.contains(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$", na=False).any():
        range = stripped.str.extract(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$")
        mean = range.astype(float).mean(axis=1)
        cleaned[col] = mean
        continue
    # percents
    if stripped.str.contains("%", na=False).all():
        cleaned[col] = pd.to_numeric(stripped.str.replace("%", "", regex=False), errors="coerce")/100.0
        continue
    # Yes/No -> 1/0
    if stripped.isin(["yes", "no", "y", "n"]).any():
        map = {"yes": 1, "y": 1, "no": 0, "n":0}
        cleaned[col] = stripped.replace(map).fillna(0).astype(float)
        continue
    # Contains both Number and text -> number
    if stripped.str.contains(r"[0-9]", na=False).any() and stripped.str.contains(r"[A-Za-z]", na=False).any():
        cleaned[col] = stripped.str.extract(r"(\d+(?:\.\d+)?)", expand=False).astype(float)
        continue
    # category maps for remaining non-numeric columns
    # if non-numeric at this point
    if stripped.dtype != "float64":
        stripped = stripped.astype(str)
        categorical_vals = sorted(stripped.dropna().unique())
        new_map = {}
        for index, cat in enumerate(categorical_vals):
            new_map[cat] = index
        allmaps[col] = new_map 
        cleaned[col] = stripped.replace(new_map).astype(float)

    # make sure it's numeric now, float64
    cleaned[col] = cleaned[col].astype("float64", errors="ignore") 
    cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

fraction_nans = cleaned.isna().mean()
for col in cleaned.columns:
    if fraction_nans[col] > 0.5:
        cleaned = cleaned.drop(columns=[col])


# NORMALIZE NUMERIC COLUMNS with standard scalar
"""
for column in cleaned.columns:
    if column.lower() == "bldg_id" or column.lower() == "in.representative_income":
        continue
    elif column in target_cols:
        continue
    column_values = cleaned[column].dropna().unique()
    if set(column_values) <= {0,1}:
        continue
    cleaned[column] = scaler.fit_transform(cleaned[[column]]).ravel()
"""
    
cleaned.to_csv(output_path + "sanmateo_preprocessed.csv", index=False)
energy_data = cleaned.dropna(subset=['out.electricity.net.energy_savings..kwh']).drop(columns=['out.energy_burden_savings..percentage',
    'out.utility_bills.total_bill_savings..usd',
    'out.emissions_reduction.total.aer_mid_case_avg..co2e_kg'])
bill_data = cleaned.dropna(subset=['out.utility_bills.total_bill_savings..usd']).drop(columns = ['out.electricity.net.energy_savings..kwh',
    'out.energy_burden_savings..percentage',
    'out.emissions_reduction.total.aer_mid_case_avg..co2e_kg'])
emiss_data = cleaned.dropna(subset=['out.emissions_reduction.total.aer_mid_case_avg..co2e_kg']).drop(columns=['out.electricity.net.energy_savings..kwh',
    'out.utility_bills.total_bill_savings..usd',
    'out.energy_burden_savings..percentage'])
energy_data.to_csv("/Users/evageierstanger/CS229FinalProject/Preprocess/sanmateo_energy_data.csv", index=False)
bill_data.to_csv("/Users/evageierstanger/CS229FinalProject/Preprocess/sanmateo_bill_data.csv", index=False)
emiss_data.to_csv("/Users/evageierstanger/CS229FinalProject/Preprocess/sanmateo_emis_data.csv", index=False)

with open(map_path, "w") as f:
    json.dump(allmaps, f, indent=4)

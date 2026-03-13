import pandas as pd

# 4 models (bill, emissions, burden, energy) -> top 200 csvs 
emission_top200 = pd.read_csv("HGBoost/200_test_emis.csv")
bill_top200 = pd.read_csv("HGBoost/200_test_bill.csv")
burden_top200 = pd.read_csv("HGBoost/200_test_burden.csv")
energy_top200 = pd.read_csv("HGBoost/200_test_elec.csv")

# select the weights
weights = [0.25, 0.25, 0.25, 0.25]
# Order is Emissions, Bill, Burden, Energy

# merge preds by building ID
merged = emission_top200.merge(bill_top200, on="bldg_id", how="outer")
merged = merged.merge(burden_top200, on="bldg_id", how="outer")
merged = merged.merge(energy_top200, on="bldg_id", how="outer")

def z_score(val):
    num = val - val.mean()
    denominator = val.std()
    if denominator == 0:
        z_score = 0
    else:
        z_score = num / denominator
    return z_score

#normalize z score predictions, csvs are currently  bldg_id, score
for col in merged.columns:
    if col != "bldg_id":
        merged[col] = z_score(merged[col])
    
#add total score column
merged['total_score'] = (weights[0]*merged["emis_red_pred"] 
+ weights[1]*merged["bill_red_pred"]
+ weights[2]*merged["burden_red_pred"]
+ weights[3]*merged["elecsav_pred"])

# sort preds descending (highest score first), reset index to show rank
merged =merged.sort_values(by=['total_score'], ascending=False).reset_index(drop=True).head(200)

# add in actual data (actual emissions, actual bill, burden, energy sav.)
# actual data
bill_data = pd.read_csv("Preprocess/sanmateo_bill_data.csv")
emission_data = pd.read_csv("Preprocess/sanmateo_emis_data.csv")
burden_data = pd.read_csv("Preprocess/sanmateo_burden_data.csv")
energy_data = pd.read_csv("Preprocess/sanmateo_energy_data.csv")

actual_data = bill_data.merge(emission_data[['bldg_id', 'out.emissions_reduction.total.aer_mid_case_avg..co2e_kg']], on="bldg_id")\
.merge(burden_data[['bldg_id', 'out.energy_burden_savings..percentage']], on="bldg_id")\
        .merge(energy_data[['bldg_id', 'out.electricity.net.energy_savings..kwh']], on="bldg_id")

actual_data = actual_data.rename(columns={
    "out.emissions_reduction.total.aer_mid_case_avg..co2e_kg": "actual_emiss_red",
    'out.utility_bills.total_bill_savings..usd': 'actual_bill_sav',
    'out.energy_burden_savings..percentage': 'actual_burd_red',
    "out.electricity.net.energy_savings..kwh": "actual_elec_save"})

merged = merged.merge(actual_data, on="bldg_id", how="left")

merged["total_score"] = merged["total_score"].astype(float)

merged = (
    merged
    .sort_values(by="total_score", ascending=False)
    .reset_index(drop=True)
    .head(200)
)

merged = merged[[
    'bldg_id',
    'emis_red_pred',
    'actual_emiss_red',
    'bill_red_pred',
    'actual_bill_sav',
    'burden_red_pred',
    'actual_burd_red',
    'elecsav_pred',
    'actual_elec_save', 
    'total_score'
]]

#Save to output csv 
merged.to_csv("HGBoost/0.25_0.25_0.25_0.25_top200_HGBoost.csv", index=False)
print("Successfully ranked the top 200 from each model")

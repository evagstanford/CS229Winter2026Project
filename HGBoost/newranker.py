import pandas as pd

# 3 models (bill, emissions, energy) -> top 200 csvs 
emission_top200 = pd.read_csv("HGBoost/200_test_emis.csv")
bill_top200 = pd.read_csv("HGBoost/200_test_bill.csv")
energy_top200 = pd.read_csv("HGBoost/200_test_elec.csv")
orig_data = pd.read_csv("/Users/evageierstanger/CS229FinalProject/Preprocess/san_mateo_cut_cols.csv")[['bldg_id', 'in.representative_income']]

# select the weights
weights = [0.1, 0.2, 0.3, 0.4]
weightstr = "0.1,0.2,0.3,0.4"
# Order is Emissions, Bill, Energy, Burden

# merge preds by building ID
merged = emission_top200.merge(bill_top200, on="bldg_id", how="inner")
merged = merged.merge(energy_top200, on="bldg_id", how="inner")
merged = merged.merge(orig_data, on="bldg_id", how="left") # adds in rep. income
merged.rename(columns={'in.representative_income': 'norm_income'}, inplace=True)

def min_max(val):
    if val.max() == val.min():
        return pd.Series(0, index=val.index)
    else:
        return (val - val.min())/(val.max() - val.min())

#normalize score predictions, csvs are currently  bldg_id, score
for col in merged.columns:
    if col != "bldg_id":
        merged[col] = min_max(merged[col])


    
#add total score column
merged['total_score'] = (weights[0]*merged["emis_red_pred"] 
+ weights[1]*merged["bill_red_pred"]
+ weights[2]*merged["elecsav_pred"]
- weights[3]*merged["norm_income"]) # higher income is punished



# sort preds descending (highest score first), reset index to show rank
merged =merged.sort_values(by=['total_score'], ascending=False).reset_index(drop=True).head(200)

# add in actual data (actual emissions, actual bill, burden, energy sav.)
# actual data
bill_data = pd.read_csv("Preprocess/sanmateo_bill_data.csv")
emission_data = pd.read_csv("Preprocess/sanmateo_emis_data.csv")
energy_data = pd.read_csv("Preprocess/sanmateo_energy_data.csv")

actual_data = bill_data.merge(emission_data[['bldg_id', 'out.emissions_reduction.total.aer_mid_case_avg..co2e_kg']], on="bldg_id")\
        .merge(energy_data[['bldg_id', 'out.electricity.net.energy_savings..kwh']], on="bldg_id")

actual_data = actual_data.rename(columns={
    "out.emissions_reduction.total.aer_mid_case_avg..co2e_kg": "actual_emiss_red",
    'out.utility_bills.total_bill_savings..usd': 'actual_bill_sav',
    "out.electricity.net.energy_savings..kwh": "actual_elec_save",
    'in.representative_income': 'actual_income_rep'})

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
    'elecsav_pred',
    'actual_elec_save',
    'norm_income',
    'actual_income_rep', 
    'total_score'
]]

#Save to output csv 
merged.to_csv("HGBoost/" + weightstr + "_newrankingwithincome.csv", index=False)


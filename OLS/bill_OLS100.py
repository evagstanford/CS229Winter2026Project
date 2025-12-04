# This file produces the top 100 buildings for predicted bill savings using OLS
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from scipy.stats import randint
from scipy.stats import loguniform
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv("Preprocess/sanmateo_bill_data.csv")
inputs = data.drop(columns=["bldg_id", 'out.utility_bills.total_bill_savings..usd'])

# split training and testing data, 20% Test
input_train, input_test, output_train, output_test, bldgid_train, bldgid_test = train_test_split(
    inputs,
    data['out.utility_bills.total_bill_savings..usd'],
    data['bldg_id'],
    test_size=0.2,
    random_state=42
)

# write out pipeline
pipeline = Pipeline(steps=[
    ('imputation', SimpleImputer(strategy='mean')),
    ('chosen_model', LinearRegression()
    )])

pipeline.fit(input_train, output_train)

preds = pipeline.predict(input_test)
r2 = r2_score(output_test, preds)
RMSE = np.sqrt(mean_squared_error(output_test, preds))
print("R^2 score:", r2, " RMSE ", RMSE)

# get top 100
top100 = pd.DataFrame({
    "bldg_id": bldgid_test.values,
    "bill_red_pred": preds
}).sort_values("bill_red_pred", ascending=False).head(100)
top100.to_csv("OLS/100_test_bill.csv", index=False)
print("See OLS/100_test_bill.csv for top 100")

# compare top from the testing set to the actual top for this category, in testing set
top_ids = top100["bldg_id"].to_list()
actual_top = pd.DataFrame({
    "bldg_id": bldgid_test,
    "actual": output_test
}).sort_values(by=["actual"], ascending=False).head(100)
actual_top100 = actual_top["bldg_id"].to_list()
shared = set(actual_top100) & set(top_ids)
print("this model got these IDs correct: ", shared)
number_right = len(shared)
print("There are this many top 100 shared: ", number_right)
print("Precision for 100: ", number_right/100)

with open("OLS/OLS_bill_results100.txt", "w") as f:
    f.write(f"OLS for Bill Savings\n")
    f.write(f"R^2: {r2} RMSE: {RMSE} \n")
    f.write(f"Top ids: {top_ids} \n")
    f.write(f"Got {number_right} out of 100, prec. {number_right/100} \n")
    
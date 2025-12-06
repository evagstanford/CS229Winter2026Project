# This file produces the top 200 buildings for predicted electricity energy savings using a Histogram Gradient Boosting Regressor
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from scipy.stats import randint
from scipy.stats import loguniform
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, r2_score, mean_squared_error, mean_absolute_error

data = pd.read_csv("Preprocess/sanmateo_energy_data.csv")
inputs = data.drop(columns=["bldg_id", 'out.electricity.net.energy_savings..kwh'])

# split training and testing data, 20% Test
input_train_all, input_test, output_train_all, output_test, bldgid_train_all, bldgid_test = train_test_split(
    inputs,
    data['out.electricity.net.energy_savings..kwh'],
    data['bldg_id'],
    test_size=0.2,
    random_state=42
)

input_train, input_valid, output_train, output_valid, bldgid_train, bldgid_valid = train_test_split(
    input_train_all,
    output_train_all,
    bldgid_train_all,
    test_size=0.25,
    random_state=42
)

# write out pipeline
HGB_pipeline = Pipeline(steps=[
    ('imputation', SimpleImputer(strategy='median')),
    ('chosen_model', HistGradientBoostingRegressor(
        random_state=42
    ))])

# parameters for HistGradient
dist_of_params = {
    'chosen_model__max_iter': randint(200, 1000),
    'chosen_model__learning_rate': loguniform(0.02, 0.25),
    'chosen_model__max_depth': [3, 4, 5, 7, 10, None],
    'chosen_model__max_leaf_nodes': [31, 63, 127, None],
    'chosen_model__min_samples_leaf': randint(10, 100),
    'chosen_model__l2_regularization': loguniform(0.001, 8.0),
    'chosen_model__max_bins': [150, 255],
    'chosen_model__early_stopping': [True],
    'chosen_model__validation_fraction': [0.1, 0.2],
    'chosen_model__n_iter_no_change': [10, 12, 15],
    'chosen_model__tol': loguniform(1e-4, 1e-2)
}

def rank_score(model, input, output, num_rank):
    predictions = model.predict(input) # numpy predictions data row by row
    # compare the indices to those of the actual top 200
    bldg_ids = data.loc[input.index, "bldg_id"]
    actual_and_pred = pd.DataFrame({
        "bldg_id": bldg_ids,
        "actual": output.values,
        "predicted": predictions
    })
    top_pred = actual_and_pred.sort_values("predicted", ascending=False).head(num_rank)
    top_pred_ids = top_pred["bldg_id"].to_list()
    top_actual = actual_and_pred.sort_values(by=["actual"], ascending=False).head(num_rank)
    top_actual_ids = top_actual["bldg_id"].to_list()
    
    shared = set(top_pred_ids) & set(top_actual_ids)
    num_right = len(shared)
    precision = num_right/num_rank
    return precision


# CROSS VALIDATION (RANDOM SEARCH) : each model has separate
# do pipeline for each param combo: imputation for the training folds, 
# train model on training folds, then repeat for val. fold
# average validation scores and then find which param combo is best
cv_search_random = RandomizedSearchCV(
    estimator=HGB_pipeline,
    param_distributions =dist_of_params,
    n_jobs=-1,
    n_iter=40, # iterations
    verbose=2,
    scoring=make_scorer(rank_score, greater_is_better=True, num_rank=200),
    cv=5, # five fold cross val.
    random_state=42,
    pre_dispatch='2*n_jobs',
    error_score= np.nan,
    return_train_score=True,
    refit=True,
)

cv_search_random.fit(input_train, output_train)

# train
use_pipeline = cv_search_random.best_estimator_
print("optimal params for elec_hist:", cv_search_random.best_params_)


preds = use_pipeline.predict(input_test)
preds_training = use_pipeline.predict(input_train_all)
r2_test = r2_score(output_test, preds)
RMSE_test = np.sqrt(mean_squared_error(output_test, preds))
mae_test = mean_absolute_error(output_test, preds)
mape_test = np.mean(np.abs((output_test - preds) / output_test)) * 100
mae_train = mean_absolute_error(output_train_all, preds_training)
mape_train = np.mean(np.abs((output_train_all - preds_training) / output_train_all)) * 100
print("R^2 score test:", r2_test, " RMSE test", RMSE_test)
r2_train = r2_score(output_train_all, preds_training)
RMSE_train = np.sqrt(mean_squared_error(output_train_all, preds_training))
print("R^2 score train:", r2_train, " RMSE train", RMSE_train)

# get top 200
top200 = pd.DataFrame({
    "bldg_id": bldgid_test.values,
    "elecsav_pred": preds
}).sort_values("elecsav_pred", ascending=False).head(200)
top200.to_csv("HGBoost/200_test_elec.csv", index=False)
print("See HGBoost/200_test_elec.csv for top 200")

# compare top from the testing set to the actual top for this category, in testing set
top_ids = top200["bldg_id"].to_list()
actual_top = pd.DataFrame({
    "bldg_id": bldgid_test,
    "actual": output_test
}).sort_values(by=["actual"], ascending=False).head(200)
actual_top200 = actual_top["bldg_id"].to_list()
shared = set(actual_top200) & set(top_ids)
print("this model got these IDs correct: ", shared)
number_right = len(shared)
print("There are this many top 200 shared: ", number_right)
print("Precision for 200: ", number_right/200)

with open("HGBoost/HGBoost_elec_results.txt", "w") as f:
    f.write(f"Hist. Grad. Boosting Reg. for Electricity Red.\n")
    f.write(f"Opt. params: {cv_search_random.best_params_} \n")
    f.write(f"R^2 test: {r2_test} RMSE test: {RMSE_test}\n")
    f.write(f"MAPE train: {mape_train} MAPE test: {mape_test}\n")
    f.write(f"mae train: {mae_train} mae test: {mae_test}\n")
    f.write(f"R2 for training data: {r2_train} RMSE for train: {RMSE_train} \n")
    f.write(f"Top ids: {top_ids} \n")
    f.write(f"Got {number_right} out of 100, prec. {number_right/100} \n")
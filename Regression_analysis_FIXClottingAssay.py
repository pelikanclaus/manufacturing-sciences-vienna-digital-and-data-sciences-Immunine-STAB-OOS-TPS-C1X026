# Databricks notebook source
# MAGIC %pip install openpyxl xgboost lazypredict shap scikit-optimize

# COMMAND ----------


import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Additional libraries from Asan
import xgboost as xgb
import sklearn
from sklearn.model_selection import train_test_split
import sklearn.metrics as sk_metrics
from skopt.searchcv import BayesSearchCV
import plotly.express as px
from lazypredict.Supervised import LazyRegressor
import shap

# COMMAND ----------

def find_outliers(df, CI_lower = '1',
                             CI_upper = '99',
                             threshold_iqr = 1.5, 
                             threshold_nullity = 50,
                             threshold_corr = 0.5,
                             col_y = ['Diss_Time_[min]']):
    """Go through the dataframe and identify which columns contain extreme values"""
    
    records = []

    for col in df.columns:
        Q1 = df[col].dropna().quantile(0.25)
        Q3 = df[col].dropna().quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = df[col].dropna().quantile(0.05)
        upper_bound = df[col].dropna().quantile(0.95)
        
        df_outlier_IQR = df[(df[col] < Q1 - threshold_iqr * IQR) | (df[col] > Q3 + threshold_iqr * IQR)][col]
        n_extremes_IQR = len(df_outlier_IQR) if IQR != 0 else 0
        
        df_outlier_CI = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        n_extremes_CI = len(df_outlier_CI)

        records.append({
            'col' : col,
            'n_extremes_IQR' : n_extremes_IQR,
            'n_extremes_CI' : n_extremes_CI,
            'min' : df[col].min(),
            'max' : df[col].max(),
            'IQR' : IQR,
            f'IQR * {threshold_iqr}' : IQR * threshold_iqr,
            'Q1' : Q1,
            'Q3' : Q3,
            f'{CI_lower}%' : lower_bound,
            f'{CI_upper}%' : upper_bound,
        })

    return pd.DataFrame(records)

# COMMAND ----------

# Set python rows to be shown 
pd.set_option('display.max_rows', 10000000)
pd.options.display.float_format = '{:.2f}'.format
np.set_printoptions(suppress=True)

# COMMAND ----------

# Read TPS data
tps_data = pd.read_excel("/Workspace/Users/claus.pelikan@takeda.com/Immunine STAB OOS TPS (C1X026)/Data/Action 23- MH - MVDA - final Raw Data set - 2024-Jul-15_1056_updated EH_2024-Jul-17.xlsx")

# COMMAND ----------

# Save data sets for datarobot

# Rmove columns (missing, correlated either with Y or with other columns, e..,g S16-FC.ProducedAmt (EA) and FillingSize, information columns)
remove_cols = ['S12-IEChromat.BatchNo_1','S12-IEChromat.BatchNo_2','S13 Produktionsdatum (Formulierung/ Sterilfiltration)']

# Deselect Y2
tps_data_Y1 = tps_data[[col for col in tps_data.columns if col not in remove_cols+['S16-FC.Sampl.SpecificActivity (IU/mg Protein)']]]

# Remove FillingSize 200
tps_data_Y1no200 = tps_data_Y1[tps_data_Y1['S16-FC.Sampl.FillingSize (mL)']>200]
tps_data_Y1no200.to_csv("/Workspace/Users/claus.pelikan@takeda.com/Immunine STAB OOS TPS (C1X026)/For_DataRobot/Immunine_TPS_07_2024_FIXClottingAssay_no200.csv", sep=";", index=False)

# Leave all Filling sizes
tps_data_Y1allFS = tps_data_Y1
tps_data_Y1allFS.to_csv("/Workspace/Users/claus.pelikan@takeda.com/Immunine STAB OOS TPS (C1X026)/For_DataRobot/Immunine_TPS_07_2024_FIXClottingAssay_allFS.csv", sep=";", index=False)

# Deselect Y1
tps_data_Y2 = tps_data[[col for col in tps_data.columns if col not in remove_cols+['S16-FC.Sampl.FIXClottingAssay (IU/ml)']]]

# Remove FillingSize 200
tps_data_Y2no200 = tps_data_Y2[tps_data_Y2['S16-FC.Sampl.FillingSize (mL)']>200]
tps_data_Y2no200.to_csv("/Workspace/Users/claus.pelikan@takeda.com/Immunine STAB OOS TPS (C1X026)/For_DataRobot/Immunine_TPS_07_2024_SpecificActivity_no200.csv", sep=";", index=False)

# Leave all Filling sizes
tps_data_Y2allFS = tps_data_Y2
tps_data_Y2allFS.to_csv("/Workspace/Users/claus.pelikan@takeda.com/Immunine STAB OOS TPS (C1X026)/For_DataRobot/Immunine_TPS_07_2024_SpecificActivity_allFS.csv", sep=";", index=False)


# COMMAND ----------

# Spezfic filter for specific Lot Sizes
tps_data = tps_data[tps_data['S16-FC.Sampl.FillingSize (mL)']>200]
#tps_data = tps_data[tps_data['S16-FC.Sampl.FillingSize (mL)']==600]
#tps_data = tps_data[tps_data['S16-FC.Sampl.FillingSize (mL)']==1200]

# COMMAND ----------


col_y = 'S16-FC.Sampl.FIXClottingAssay (IU/ml)' # primary Y
col_y2 = 'S16-FC.Sampl.SpecificActivity (IU/mg Protein)' # alternative Y that can be tried as well. 

# Remove second y column
tps_data = tps_data.drop(col_y2, axis=1)

# Remove rows with NaN in Y
tps_data.dropna(subset=[col_y], inplace = True)

# COMMAND ----------

# Set Lot ID as index
#tps_data.set_index('Lot Nr.', inplace=True)

# COMMAND ----------

# Check data types
#pd.DataFrame(tps_data.dtypes)

# COMMAND ----------

# Plot missing percentage 
percent_missing = tps_data.isnull().sum() * 100 / len(tps_data)
missing_value_df = pd.DataFrame({'column_name': tps_data.columns,
                                 'percent_missing': percent_missing})

missing_value_df.sort_values('percent_missing', ascending=False).plot.bar(figsize=(25, 5))

# COMMAND ----------

# Remove parameters with more than 55% missing (slightly higher acceptance due to Durations that were extracted from Visplore)
threshold_nullity = 50
no_missing = missing_value_df[missing_value_df['percent_missing'] <= threshold_nullity].index
tps_data = tps_data.loc[:, no_missing]
print("The following parameters were removed with nullilty greater than " + str(threshold_nullity) + "%: '" + "','".join(missing_value_df[missing_value_df['percent_missing'] >= threshold_nullity].index) + "'")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Define data types

# COMMAND ----------

# Numeric columns (exclude objects and date)
num_cols = [col for col in tps_data.select_dtypes(exclude=['object']).columns if col not in [col_y,'S13 Produktionsdatum (Formulierung/ Sterilfiltration)']]
tps_data[num_cols].display()

# COMMAND ----------

# Columnns that should be ignored for PCA, linreg but might be informative for other reasons
info_cols = ['Lot Nr.', 'S12-IEChromat.BatchNo_1','S12-IEChromat.BatchNo_2','S13 Produktionsdatum (Formulierung/ Sterilfiltration)']
tps_data[info_cols].display()

# COMMAND ----------

# Categorical variables
cat_cols = [col for col in tps_data.select_dtypes(exclude=['int64','float64']).columns if col not in info_cols]
tps_data[cat_cols].display()

# COMMAND ----------

# Encode categorical variables
cat_cols_encoded = []
for cat_col in cat_cols:
    tps_data[cat_col + "_code"] = tps_data[cat_col].astype('category').cat.codes
    cat_cols_encoded.append(cat_col + "_code")

# COMMAND ----------

tps_data[cat_cols + cat_cols_encoded].drop_duplicates().display()

# COMMAND ----------

# Define which data is used during TPS
tps_data = tps_data[['Lot Nr.'] + num_cols +  [col_y]] # + cat_cols_encoded

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Evaluate correlation of variables

# COMMAND ----------

# Plot correlation matrix
corrmat = tps_data.corr()
fig_corr_y = px.imshow(corrmat, height = 2000, width = 2000) #(corrmat[col_y].dropna(), height = 2000) #
fig_corr_y.show()

# COMMAND ----------

# Remove highly correlated features (except for Y itself)
threshold_corr = 0.6
for parameter in corrmat.columns:
    if len([par for par in corrmat[[parameter]][corrmat[[parameter]].abs()>=threshold_corr].dropna().index if par not in [parameter]]) > 0:
        print(parameter + " is highly correlated to: " + "'" + "','".join([par for par in corrmat[[parameter]][corrmat[[parameter]].abs()>=threshold_corr].dropna().index if par not in [parameter]]) + "'")


# COMMAND ----------

col_y

# COMMAND ----------

# Remove highly correlated features for Y (except for Y itself)
threshold_corr = 0.6
print("Removed:" + "'" + "','".join([par for par in corrmat[col_y][corrmat[col_y].abs()>=threshold_corr].dropna().index if par != col_y]) + "'")
no_corr_parameters = [par for par in corrmat[col_y][corrmat[col_y].abs()<threshold_corr].dropna().index]
tps_data = tps_data.loc[:, ['Lot Nr.'] + no_corr_parameters + [col_y]] # info paramters are removed this way

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Evaluate outliers

# COMMAND ----------

# Find outliers
outliers = find_outliers(tps_data[no_corr_parameters + [col_y]]).sort_values('n_extremes_IQR', ascending=False)
outliers.display()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Overview plots

# COMMAND ----------

# Plot relationship
dbutils.widgets.combobox(
    "Parameter",
    str(tps_data.columns[2]),
    [str(x) for x in tps_data.columns.tolist()]
)

# COMMAND ----------

fig = px.histogram(tps_data,x=dbutils.widgets.get("Parameter"),marginal="box")
fig.update_layout(height=750, width=1000, modebar_orientation="v").show()

fig = px.scatter(tps_data.dropna(subset=[dbutils.widgets.get("Parameter")]),x=dbutils.widgets.get("Parameter"),y=col_y,trendline="ols", hover_data= ['Lot Nr.'])
fig.update_layout(height=750, width=1000, modebar_orientation="v").show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #Regression analysis

# COMMAND ----------

# Generate dataset without Y
x_cols = [col for col in list(tps_data.columns) if col not in [col_y ,'Lot Nr.']]
tps_data_noY = tps_data[x_cols]

# COMMAND ----------

# Train test split
test_size = 0.2
random_state = 42

X, y = tps_data_noY.values, tps_data[col_y]


X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size = test_size, random_state = random_state)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Lazy regressor

# COMMAND ----------

#reg = LazyRegressor(verbose=1, ignore_warnings=False, custom_metric=None)
#models, predictions = reg.fit(X_tr, X_te, y_tr, y_te)

# COMMAND ----------

#predictions

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### XGBoost regressor

# COMMAND ----------

xgb_init_params = { 
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'verbosity': 0,
    }

xgb_params = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'min_child_weight': (1, 10),
        'max_depth': (1, 10),
        'max_delta_step': (1, 20),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1, 1000, 'log-uniform'),
        'reg_alpha': (0.1, 1.0, 'log-uniform'),
        'gamma': (0.1, 0.5, 'log-uniform'),
        'n_estimators': [20, 50, 100, 200, 400],
    }

# COMMAND ----------

bayes_searchCV = BayesSearchCV(
    estimator = xgb.XGBRegressor(**xgb_init_params),
    search_spaces = xgb_params,
    cv = 3,
    n_iter = 20,
    scoring = 'r2',
    verbose = 1,
    n_jobs = -1,
    refit = True
)

bayes_searchCV.fit(X_tr, y_tr)

# COMMAND ----------

bayes_searchCV.best_score_

# COMMAND ----------

# Check if the bayes_searchCV function found a better score othwise use initial parameters

estimator = bayes_searchCV.best_estimator_
#estimator = xgb.XGBRegressor(**xgb_init_params)
estimator.fit(X_tr, y_tr)

# COMMAND ----------

print(f'Train R2 {sk_metrics.r2_score(y_tr.values, estimator.predict(X_tr))}')
print(f'Test R2 {sk_metrics.r2_score(y_te.values, estimator.predict(X_te))}')

# COMMAND ----------

df_preds_vs_observed_tr = pd.DataFrame({
    'predicted' : estimator.predict(X_tr),
    'observed' : y_tr.values.ravel()
})

fig_predicted_vs_observed_tr = px.scatter(df_preds_vs_observed_tr, x = 'predicted', y = 'observed', title='Predicted dissolution time vs observed (train set)')
fig_predicted_vs_observed_tr.update_xaxes(range=[df_preds_vs_observed_tr.observed.min()-1, df_preds_vs_observed_tr.observed.max()+1])
fig_predicted_vs_observed_tr.update_yaxes(range=[df_preds_vs_observed_tr.observed.min()-1, df_preds_vs_observed_tr.observed.max()+1])
fig_predicted_vs_observed_tr.show()

# COMMAND ----------

df_preds_vs_observed_te = pd.DataFrame({
    'predicted' : estimator.predict(X_te),
    'observed' : y_te.values.ravel()
})

fig_predicted_vs_observed_te = px.scatter(df_preds_vs_observed_te, x = 'predicted', y = 'observed', title='Predicted Yield vs observed (test set)')
fig_predicted_vs_observed_te.update_layout(height=750, width=1000, modebar_orientation="v")
fig_predicted_vs_observed_te.update_xaxes(range=[df_preds_vs_observed_te.observed.min()-1, df_preds_vs_observed_te.observed.max()+1])
fig_predicted_vs_observed_te.update_yaxes(range=[df_preds_vs_observed_te.observed.min()-1, df_preds_vs_observed_te.observed.max()+1])
fig_predicted_vs_observed_te.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Explain predictions using SHAP

# COMMAND ----------

explainer = shap.Explainer(estimator, X, feature_names = [col for col in tps_data_noY.columns if col !=col_y])
shap_values = explainer(X)

# COMMAND ----------

shap.plots.bar(shap_values.abs.mean(0))

# COMMAND ----------

shap.summary_plot(shap_values, X, plot_size = [15.0,8.0])

# COMMAND ----------

top_k = 15
order_mean = np.argsort(shap_values.abs.mean(0).values)
top_variables = np.array(x_cols)[order_mean][::-1][:top_k].tolist()
top_variables

# COMMAND ----------

for par in top_variables:
    shap.plots.scatter(shap_values[:, par])

# COMMAND ----------

df_pandas_with_y = tps_data_noY
df_pandas_with_y[col_y] = y

# COMMAND ----------

shap.plots.heatmap(shap_values) #, instance_order=np.argsort(df_pandas_with_y[col_y]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Look at the most important variables

# COMMAND ----------

px.parallel_coordinates(df_pandas_with_y, dimensions=top_variables + [col_y])

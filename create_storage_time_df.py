# Databricks notebook source
# MAGIC %pip install openpyxl

# COMMAND ----------

import pandas as pd

# COMMAND ----------

# Read TPS data
tps_data = pd.read_excel("/Workspace/Users/claus.pelikan@takeda.com/Immunine STAB OOS TPS (C1X026)/Data/Action 23- MH - MVDA - final Raw Data set - 2024-Jul-15_1056.xlsx")

# COMMAND ----------

# Read Storage time data
storage_time = pd.read_excel("/Workspace/Users/claus.pelikan@takeda.com/Immunine STAB OOS TPS (C1X026)/Data/Action 23_EH_17072024_MVDA Bulk Lagerzeit.xlsx", sheet_name="Bulklagerzeiten")

# Melt tps_data for FC to bulk converion
FC_bulk_df = pd.melt(tps_data[['Lot Nr.','S12-IEChromat.BatchNo_1','S12-IEChromat.BatchNo_2']],id_vars=['Lot Nr.'])
FC_bulk_df.columns = ['Lot Nr.','BatchNo','Bulk Lot Nr.']

# Read bulk storage times and link to FC_bulk_df
bulk1 = storage_time[['Parameter Set Name','Step16-FreezeDrying(FinalContainer).BatchNo➡1','Step16-FreezeDrying(FinalContainer).Bulk Storage Duration➡1\n(days)']]
bulk1.columns = ['Bulk Lot Nr.','Lot Nr.','Step16-FreezeDrying(FinalContainer).Bulk Storage Duration']
bulk2 = storage_time[['Parameter Set Name','Step16-FreezeDrying(FinalContainer).BatchNo➡2','Step16-FreezeDrying(FinalContainer).Bulk Storage Duration➡2\n(days)']]
bulk2.columns = ['Bulk Lot Nr.','Lot Nr.','Step16-FreezeDrying(FinalContainer).Bulk Storage Duration']
bulk3 = storage_time[['Parameter Set Name','Step16-FreezeDrying(FinalContainer).BatchNo➡3','Step16-FreezeDrying(FinalContainer).Bulk Storage Duration➡3\n(days)']]
bulk3.columns = ['Bulk Lot Nr.','Lot Nr.','Step16-FreezeDrying(FinalContainer).Bulk Storage Duration']
bulk4 = storage_time[['Parameter Set Name','Step16-FreezeDrying(FinalContainer).BatchNo➡4','Step16-FreezeDrying(FinalContainer).Bulk Storage Duration➡4\n(days)']]
bulk4.columns = ['Bulk Lot Nr.','Lot Nr.','Step16-FreezeDrying(FinalContainer).Bulk Storage Duration']

bulk_duration = pd.concat([bulk1, bulk2, bulk3, bulk4]).merge(FC_bulk_df, on = ['Lot Nr.','Bulk Lot Nr.'], how = "right")
#bulk_duration1 = bulk_duration.pivot(index='Lot Nr.',columns='BatchNo', values='Bulk Lot Nr.').reset_index()
bulk_duration2 = bulk_duration.pivot(index='Lot Nr.',columns='BatchNo', values='Step16-FreezeDrying(FinalContainer).Bulk Storage Duration').reset_index()
#bulk_duration1.merge(bulk_duration2, on = 'Lot Nr.', how="outer").display()
bulk_duration2.columns = ['Lot Nr.','Step16-FreezeDrying(FinalContainer).Bulk Storage Duration 1', 'Step16-FreezeDrying(FinalContainer).Bulk Storage Duration 2']
bulk_duration2.display()

# COMMAND ----------

bulk_duration.display()

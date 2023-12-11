# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:17:58 2023

@author: USER
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data_uri = 'C:\\Users\\USER\\Desktop\\ADS\\Assignment_2\\Dataset\\population.csv'
pivoted_data_uri = 'C:\\Users\\USER\\Desktop\\ADS\\Assignment_2\\Dataset\\PivotedDataset.csv'
cleaned_data_uri = 'C:\\Users\\USER\\Desktop\\ADS\\Assignment_2\\Dataset\\CleanedDataset.csv'


def Dataset(data_uri):
    dataset = pd.read_csv(data_uri)
    return dataset

""" Defining Feature codes and country names to analyse """

column_codes=[ 'Country Name','Country Code','Year','SP.POP.TOTL.MA.ZS', 'SP.POP.TOTL.FE.ZS', 'SP.POP.GROW', 'SP.POP.DPND.YG',
                'SP.POP.DPND.OL', 'SP.POP.DPND', 'SP.DYN.CDRT.IN','SP.DYN.CBRT.IN','SP.ADO.TFRT']

feature_map={
    "SP.POP.TOTL.MA.ZS" : "Population, male (% of total population) ",
    "SP.POP.TOTL.FE.ZS" : "Population, female (% of total population)",
    "SP.POP.GROW" : "Population growth (annual %)",
    "SP.POP.DPND.YG" : "Age dependency ratio, young (% of working-age population) ",
    "SP.POP.DPND.OL" : "Age dependency ratio, old (% of working-age population)",
    "SP.POP.DPND" : "Age dependency ratio (% of working-age population)",
    "SP.DYN.CDRT.IN": "Death rate, crude (per 1,000 people)",
    "SP.DYN.CBRT.IN":"Birth rate, crude (per 1,000 people)",
    "SP.ADO.TFRT":"Adolescent fertility rate (births per 1,000 women ages 15-19)"
}
countries={
    "AUS":"Australia",
    "BHR":"Bahrain",
    "CAN":"Canada",
    "DEU":"Germany",
    "ARG":"Argentina",
    "GRD":"Grenada",
    "JPN":"Japan",
    "LBR":"Liberia",
    "MEX":"Mexico",
    "PAN":"Panama",
    "SWE":"Sweden",
    "UGA":"Uganda",
    "ZWE":"Zimbabwe"
}

melted_df = Dataset(data_uri).melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], var_name='Year', value_name='Value')

# Pivot the table to have Indicator Names as columns
pivoted_df = melted_df.pivot_table(index=['Country Name', 'Country Code', 'Year'], columns='Indicator Code', values='Value').reset_index()

# Fill NaN values with a specific value (e.g., 0)
pivoted_df.fillna(0, inplace=True)

# Save the pivoted DataFrame to a CSV file
pivoted_df.to_csv('C:/Users/USER/Desktop/ADS/Assignment_2/Dataset/PivotedDataset.csv', index=False)


filtered_columns = [col for col in pivoted_df.columns if col in column_codes]
df_filtered = pivoted_df[filtered_columns]


#cleaning the transformed data set 
# Fill missing values with the mean of the column
df_filtered = df_filtered.fillna(df_filtered.mean(numeric_only=True))
df_filtered.to_csv('C:/Users/USER/Desktop/ADS/Assignment_2/Dataset/CleanedDataset.csv')

# Applying Statistical Methods on cleaned dataset
copy_df_cleaned = df_filtered.drop(['Year', 'Country Name'], axis='columns')
print(copy_df_cleaned.describe())

df_ARG = df_filtered[df_filtered["Country Name"] == "Argentina"]
df_JPN = df_filtered[df_filtered["Country Name"] == "Japan"]
df_SWE = df_filtered[df_filtered["Country Name"] == "Sweden"]

# Correlation Matrix and Heat map for Argentina
correaltion_matrix_ARG = df_ARG.corr(numeric_only=True)
correaltion_matrix_ARG = correaltion_matrix_ARG.rename(columns=feature_map)
correaltion_matrix_ARG = correaltion_matrix_ARG.rename(index=feature_map)
plt.figure(1,figsize=(10,10))
heatmap_data = sns.heatmap(correaltion_matrix_ARG, annot=True,fmt=".1g", vmax=1, vmin=0) 
plt.title('Correlation Matrix for Argentina')
plt.show()

# Correlation Matrix and Heat map for Liberia 
correaltion_matrix_JPN = df_JPN.corr(numeric_only=True)
correaltion_matrix_JPN = correaltion_matrix_JPN.rename(columns=feature_map)
correaltion_matrix_JPN = correaltion_matrix_JPN.rename(index=feature_map)
plt.figure(2,figsize=(10,10))
heatmap_data = sns.heatmap(correaltion_matrix_JPN , annot=True,fmt=".1g", vmax=1, vmin=0) 
plt.title('Correlation Matrix for Japan')
plt.show()

# Correlation Matrix and Heat map for Sweden
correaltion_matrix_SWE = df_SWE.corr(numeric_only=True)
correaltion_matrix_SWE = correaltion_matrix_SWE.rename(columns=feature_map)
correaltion_matrix_SWE = correaltion_matrix_SWE.rename(index=feature_map)
plt.figure(2,figsize=(10,10))
heatmap_data = sns.heatmap(correaltion_matrix_SWE , annot=True,fmt=".1g", vmax=1, vmin=0) 
plt.title('Correlation Matrix for Sweden')
plt.show()

# bar graphs 
# Filtering for India and China for the years 1960 and 2022
df_population_by_year = pd.read_csv(cleaned_data_uri)

filtered_population = df_population_by_year[
    ((df_population_by_year['Country Name'] == 'Bahrain') | (df_population_by_year['Country Name'] == 'Canada') | 
    (df_population_by_year['Country Name'] == 'Sweden') | (df_population_by_year['Country Name'] == 'Uganda')) &
    ((df_population_by_year['Year'] == 1980) | (df_population_by_year['Year'] == 2000) | (df_population_by_year['Year'] == 2022))]

filtered_population = filtered_population[["Country Name","Year","SP.POP.GROW"]]
pivoted_population_df = filtered_population.pivot(index='Country Name', columns='Year', values='SP.POP.GROW').reset_index()
pivoted_population_df.plot(kind='bar',x='Country Name',y=[1980, 2000, 2022])
plt.xticks(rotation=30, horizontalalignment="center")
plt.show()
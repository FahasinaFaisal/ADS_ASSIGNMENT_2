# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:07:45 2023

@author: USER
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define file paths
data_uri = 'C:\\Users\\USER\\Desktop\\ADS\\Assignment_2\\Dataset\\population.csv'
pivoted_data_uri = 'C:\\Users\\USER\\Desktop\\ADS\\Assignment_2\\Dataset\\PivotedDataset.csv'
cleaned_data_uri = 'C:\\Users\\USER\\Desktop\\ADS\\Assignment_2\\Dataset\\CleanedDataset.csv'

# Function to load dataset
def load_dataset(file_path):
    dataset = pd.read_csv(file_path)
    return dataset

# Define feature codes, feature map, and country names
feature_codes = ['Country Name', 'Country Code', 'Year', 'SP.POP.TOTL.MA.ZS', 'SP.POP.TOTL.FE.ZS', 'SP.POP.GROW',
                 'SP.POP.DPND.YG', 'SP.POP.DPND.OL', 'SP.POP.DPND', 'SP.DYN.CDRT.IN', 'SP.DYN.CBRT.IN', 'SP.ADO.TFRT']

feature_map = {
    "SP.POP.TOTL.MA.ZS": "Population, male (% of total population)",
    "SP.POP.TOTL.FE.ZS": "Population, female (% of total population)",
    "SP.POP.GROW": "Population growth (annual %)",
    "SP.POP.DPND.YG": "Age dependency ratio, young (% of working-age population)",
    "SP.POP.DPND.OL": "Age dependency ratio, old (% of working-age population)",
    "SP.POP.DPND": "Age dependency ratio (% of working-age population)",
    "SP.DYN.CDRT.IN": "Death rate, crude (per 1,000 people)",
    "SP.DYN.CBRT.IN": "Birth rate, crude (per 1,000 people)",
    "SP.ADO.TFRT": "Adolescent fertility rate (births per 1,000 women ages 15-19)"
}

countries = {
    "AUS": "Australia",
    "BHR": "Bahrain",
    "CAN": "Canada",
    "DEU": "Germany",
    "ARG": "Argentina",
    "GRD": "Grenada",
    "JPN": "Japan",
    "LBR": "Liberia",
    "MEX": "Mexico",
    "PAN": "Panama",
    "SWE": "Sweden",
    "UGA": "Uganda",
    "ZWE": "Zimbabwe"
}

# Load and melt the dataset
melted_df = load_dataset(data_uri).melt(id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                                        var_name='Year', value_name='Value')

# Pivot the table to have Indicator Codes as columns
pivoted_df = melted_df.pivot_table(index=['Country Name', 'Country Code', 'Year'], columns='Indicator Code',
                                   values='Value').reset_index()

# Fill NaN values with 0
pivoted_df.fillna(0, inplace=True)

# Save the pivoted DataFrame to a CSV file
pivoted_df.to_csv(pivoted_data_uri, index=False)

# Filter columns based on feature codes
filtered_columns = [col for col in pivoted_df.columns if col in feature_codes]
df_filtered = pivoted_df[filtered_columns]

# Clean the transformed dataset
# Fill missing values with the mean of the column
df_filtered = df_filtered.fillna(df_filtered.mean(numeric_only=True))
df_filtered.to_csv(cleaned_data_uri)

# Apply statistical methods on the cleaned dataset
copy_df_cleaned = df_filtered.drop(['Year', 'Country Name'], axis='columns')
print(copy_df_cleaned.describe())

# Create DataFrames for specific countries
df_argentina = df_filtered[df_filtered["Country Name"] == "Argentina"]
df_japan = df_filtered[df_filtered["Country Name"] == "Japan"]
df_sweden = df_filtered[df_filtered["Country Name"] == "Sweden"]

# Correlation Matrix and Heat map for Argentina
correlation_matrix_argentina = df_argentina.corr(numeric_only=True).rename(columns=feature_map).rename(index=feature_map)
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix_argentina, annot=True, fmt=".1g", vmax=1, vmin=0)
plt.title('Correlation Matrix for Argentina')
plt.show()

# Correlation Matrix and Heat map for Japan
correlation_matrix_japan = df_japan.corr(numeric_only=True).rename(columns=feature_map).rename(index=feature_map)
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix_japan, annot=True, fmt=".1g", vmax=1, vmin=0)
plt.title('Correlation Matrix for Japan')
plt.show()

# Correlation Matrix and Heat map for Sweden
correlation_matrix_sweden = df_sweden.corr(numeric_only=True).rename(columns=feature_map).rename(index=feature_map)
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix_sweden, annot=True, fmt=".1g", vmax=1, vmin=0)
plt.title('Correlation Matrix for Sweden')
plt.show()

# Bar graphs
# Filtering for specific countries and years
df_population_by_year = pd.read_csv(cleaned_data_uri)

filtered_population = df_population_by_year[
    ((df_population_by_year['Country Name'] == 'Bahrain') | (df_population_by_year['Country Name'] == 'Canada') |
     (df_population_by_year['Country Name'] == 'Sweden') | (df_population_by_year['Country Name'] == 'Uganda')) &
    ((df_population_by_year['Year'] == 1980) | (df_population_by_year['Year'] == 2000) | (df_population_by_year['Year'] == 2022))]

filtered_population = filtered_population[["Country Name", "Year", "SP.POP.GROW"]]
pivoted_population_df = filtered_population.pivot(index='Country Name', columns='Year', values='SP.POP.GROW').reset_index()
pivoted_population_df.plot(kind='bar', x='Country Name', y=[1980, 2000, 2022])
plt.xticks(rotation=30, horizontalalignment="center")
plt.show()

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
                'SP.POP.DPND.OL', 'SP.POP.DPND', 'SP.DYN.TO65.MA.ZS', 'SP.DYN.TO65.FE.ZS', 'SP.DYN.LE00.MA.IN', 'SP.DYN.LE00.IN',
                'SP.DYN.LE00.FE.IN', 'SP.DYN.IMRT.MA.IN', 'SP.DYN.IMRT.IN', 'SP.DYN.IMRT.FE.IN']

feature_map={
    "SP.POP.TOTL.MA.ZS" : "Population, male (% of total population) ",
    "SP.POP.TOTL.FE.ZS" : "Population, female (% of total population)",
    "SP.POP.GROW" : "Population growth (annual %)",
    "SP.POP.DPND.YG" : "Age dependency ratio, young (% of working-age population) ",
    "SP.POP.DPND.OL" : "Age dependency ratio, old (% of working-age population)",
    "SP.POP.DPND" : "Age dependency ratio (% of working-age population)",
    "SP.DYN.TO65.MA.ZS" : "Survival to age 65, male (% of cohort)",
    "SP.DYN.TO65.FE.ZS" : "Survival to age 65, female (% of cohort) ",
    "SP.DYN.LE00.MA.IN" : "Life expectancy at birth, male (years)",
    "SP.DYN.LE00.IN" : "Life expectancy at birth, total (years)",
    "SP.DYN.LE00.FE.IN" : "Life expectancy at birth, female (years)",
    "SP.DYN.IMRT.MA.IN" : "Mortality rate, infant, male (per 1,000 live births) ",
    "SP.DYN.IMRT.IN" : "Mortality rate, infant (per 1,000 live births)",
    "SP.DYN.IMRT.FE.IN" : "Mortality rate, infant, female (per 1,000 live births) "
}

countries={
    "AUS":"Australia",
    "BHR":"Bahrain",
    "CAN":"Canada",
    "DEU":"Germany",
    "FJI":"Fiji",
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

df_AUS = df_filtered[df_filtered["Country Name"] == "Australia"]
df_LBR = df_filtered[df_filtered["Country Name"] == "Liberia"]
df_SWE = df_filtered[df_filtered["Country Name"] == "Sweden"]

# Correlation Matrix and Heat map for Australia 
correaltion_matrix_AUS = df_AUS.corr(numeric_only=True)
correaltion_matrix_AUS = correaltion_matrix_AUS.rename(columns=feature_map)
correaltion_matrix_AUS = correaltion_matrix_AUS.rename(index=feature_map)
plt.figure(1,figsize=(10,10))
heatmap_data = sns.heatmap(correaltion_matrix_AUS , annot=True,fmt=".1g", vmax=1, vmin=0) 
plt.title('Correlation Matrix for Australia')
plt.show()

# Correlation Matrix and Heat map for Liberia 
correaltion_matrix_LBR = df_LBR.corr(numeric_only=True)
correaltion_matrix_LBR = correaltion_matrix_LBR.rename(columns=feature_map)
correaltion_matrix_LBR = correaltion_matrix_LBR.rename(index=feature_map)
plt.figure(2,figsize=(10,10))
heatmap_data = sns.heatmap(correaltion_matrix_AUS , annot=True,fmt=".1g", vmax=1, vmin=0) 
plt.title('Correlation Matrix for Liberia')
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
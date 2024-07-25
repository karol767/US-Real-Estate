import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from lxml import html
from typing import Optional
import lxml.etree
import requests
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from google.colab import drive
drive.mount('/content/drive')

real_estate_dataset_df = pd.read_csv('/content/housing.csv')

real_estate_dataset_df.head()

real_estate_dataset_df.columns

real_estate_dataset_df.info()

real_estate_dataset_df.describe()

columns_to_drop = ['id','url', 'region_url', 'image_url', 'region', 'description']
real_estate_dataset_df_cleaned = real_estate_dataset_df.drop(columns_to_drop,axis=1)

real_estate_dataset_df_cleaned.dropna(inplace=True)
real_estate_dataset_df_cleaned.drop_duplicates(inplace=True)

real_estate_dataset_df_cleaned = real_estate_dataset_df_cleaned[real_estate_dataset_df_cleaned['price'] != 0]

real_estate_dataset_df_cleaned['state'] = real_estate_dataset_df_cleaned['state'].str.upper()
real_estate_dataset_df_cleaned.rename(columns = {'state': 'State'}, inplace = True)

real_estate_dataset_df_cleaned['baths'] = real_estate_dataset_df_cleaned['baths'].astype(int)
real_estate_dataset_df_cleaned['beds'] = real_estate_dataset_df_cleaned['beds'].astype(int)

real_estate_dataset_df_cleaned.info()

from typing import Optional

import lxml.etree

def indent_lxml(element: lxml.etree.Element, level: int = 0, is_last_child: bool = True) -> None:
    space = "  "
    indent_str = "\n" + level * space

    element.text = strip_or_null(element.text)
    if element.text:
        element.text = f"{indent_str}{space}{element.text}"

    num_children = len(element)
    if num_children:
        element.text = f"{element.text or ''}{indent_str}{space}"

        for index, child in enumerate(element.iterchildren()):
            is_last = index == num_children - 1
            indent_lxml(child, level + 1, is_last)

    elif element.text:
        element.text += indent_str

    tail_level = max(0, level - 1) if is_last_child else level
    tail_indent = "\n" + tail_level * space
    tail = strip_or_null(element.tail)
    element.tail = f"{indent_str}{tail}{tail_indent}" if tail else tail_indent


def strip_or_null(text: Optional[str]) -> Optional[str]:
    if text is not None:
        return text.strip() or None

w = requests.get("https://en.wikipedia.org/wiki/List_of_U.S._states_and_territories_by_GDP")
dom_tree = html.fromstring(w.content)
print(dom_tree)


result = lxml.etree.tostring(dom_tree, encoding="unicode")
for node in dom_tree.xpath("//table/*"):
  result = lxml.etree.tostring(node, encoding="unicode")

updated_dom_tree = dom_tree.xpath('.//tbody')[0]

result = lxml.etree.tostring(updated_dom_tree, encoding="unicode")

x_path_states = ".//tr/td[1]/a/text()"
states_raw = updated_dom_tree.xpath(x_path_states)
states = [item.replace('\n', '').replace('\u202f*','').strip() for item in states_raw]

x_path_gdp = ".//tr/td[7]/text()"
gdp_raw = updated_dom_tree.xpath(x_path_gdp)
gdp = [int(item.replace('\n', '').replace(',','').replace('$', '').strip()) for item in gdp_raw]

gdp_df = pd.DataFrame({'States': states, 'GDP per capita': gdp})
gdp_df.sort_values(by=['GDP per capita'], inplace=True, ascending=False)
gdp_df.reset_index(inplace=True, drop=True)
print(gdp_df)

states_abb_df = pd.read_csv('https://raw.githubusercontent.com/XuanyouLiu/US-Real-Estate-Analysis/main/Datasets/us_states_full_and_abbrev.csv')
states_abb_df.rename(columns={'State Full Name': 'States'}, inplace=True)

state_gdp_df = pd.merge(gdp_df, states_abb_df, on='States')

state_gdp_df.drop(columns = 'States', inplace = True)
state_gdp_df.rename(columns = {'State Abbreviation': 'State'}, inplace = True)

state_gdp_df

data_file = 'https://raw.githubusercontent.com/XuanyouLiu/US-Real-Estate-Analysis/main/Datasets/reported_violent_crimes_usa_2022.xlsx'
crime_data_df = pd.read_excel(data_file)

states_list = states_abb_df['State Abbreviation'].tolist()
abbreviation_map = dict(zip(states_abb_df['States'], states_abb_df['State Abbreviation']))

crime_data_df['Abbreviated'] = crime_data_df['State'].map(abbreviation_map)
crime_data_df['State'] = crime_data_df['Abbreviated']

crime_data_df.drop(columns=["Abbreviated"], inplace=True)
crime_data_df
weather_df = pd.read_csv('https://raw.githubusercontent.com/XuanyouLiu/US-Real-Estate-Analysis/main/Datasets/average_monthly_temperature_by_state_1950-2022.csv')
weather_df.info()

weather_df = weather_df[weather_df['year'] >= 2012]
weather_cleaned_df = weather_df.drop(columns=['Unnamed: 0', 'monthly_mean_from_1901_to_2000', 'centroid_lon', 'centroid_lat'])
weather_cleaned_df

def weather_score(temperature):
    score = 0
    if 64.4 <= temperature <= 75.2:
        score = 10
    elif temperature < 64.4:
        score = 10 - (64.4 - temperature) / 3
    elif temperature > 75.2:
        score = 10 - (temperature - 75.2) / 3

    if score < 0:
        score = 0

    if temperature < 50 or temperature > 86:
        score = score - 5

    return score

weather_cleaned_df['weather_score'] = weather_cleaned_df['average_temp'].apply(weather_score)

weather_score_df = weather_cleaned_df.groupby(['state'])['weather_score'].mean().reset_index()
weather_score_df.sort_values(by=['weather_score'], ascending=False, inplace=True)
weather_score_df.reset_index(drop=True, inplace=True)
weather_score_df['weather_score'] = weather_score_df['weather_score'].apply(lambda x: round((x/10) * 100, 2))


states_abb_df = pd.read_csv('https://raw.githubusercontent.com/XuanyouLiu/US-Real-Estate-Analysis/main/Datasets/us_states_full_and_abbrev.csv')
states_abb_df.rename(columns={'State Full Name': 'state'}, inplace=True)
weather_score_df_merge = weather_score_df.merge(states_abb_df, on='state', how = 'right')
weather_score_df_merge[weather_score_df_merge['weather_score'].isnull()]

hawaii_df = pd.read_csv('https://raw.githubusercontent.com/XuanyouLiu/US-Real-Estate-Analysis/main/Datasets/Hawaii-monthly-2006-2020.csv')

hawaii_df = hawaii_df[['DATE', 'MLY-TAVG-NORMAL']]

hawaii_df.rename(columns={'DATE': 'month', 'MLY-TAVG-NORMAL': 'average_temp'}, inplace=True)

hawaii_df['score'] = hawaii_df['average_temp'].apply(weather_score)

hawaii_score = hawaii_df['score'].mean()/10 * 100

weather_score_df_merge.loc[10, 'weather_score'] = round(hawaii_score, 2)

alaska_df = pd.read_csv('https://raw.githubusercontent.com/XuanyouLiu/US-Real-Estate-Analysis/main/Datasets/Alaska-monthly-2006-2020.csv')

alaska_df = alaska_df[['DATE', 'MLY-TAVG-NORMAL']]

alaska_df.rename(columns={'DATE': 'month', 'MLY-TAVG-NORMAL': 'average_temp'}, inplace=True)

alaska_df['score'] = alaska_df['average_temp'].apply(weather_score)

alaska_score = alaska_df['score'].mean()/10 * 100

weather_score_df_merge.loc[1, 'weather_score'] = round(alaska_score, 2)

dc_df = pd.read_csv('https://raw.githubusercontent.com/XuanyouLiu/US-Real-Estate-Analysis/main/Datasets/DC-monthly-1871-2022.csv')

dc_df = dc_df[dc_df['YEAR'] >= 2012].reset_index(drop = True)

dc_df.drop(columns=['YEAR'], inplace=True)

dc_score = []

for col in dc_df.columns:
    dc_score.append(dc_df[col].apply(weather_score).mean())

dc_score = (sum(dc_score)/len(dc_score))/10 * 100

weather_score_df_merge.loc[50, 'weather_score'] = round(dc_score, 2)


weather_score_df_merge.sort_values(by=['weather_score'], ascending=False, inplace=True)
weather_score_df_merge.reset_index(drop=True, inplace=True)
weather_score_df_merge.drop(columns = ['state'], inplace = True)
weather_score_df_merge.rename(columns = {'State Abbreviation': 'State'}, inplace = True)
weather_score_df = weather_score_df_merge
weather_score_df



spending_df = pd.read_csv("https://raw.githubusercontent.com/XuanyouLiu/US-Real-Estate-Analysis/main/Datasets/spending.csv")
spending_df.head()


spending_cleaned_df = spending_df.copy()
spending_cleaned_df['Region and State'] = spending_df['Region and State'].str.replace('.', '').str.strip()
spending_cleaned_df.rename(columns={'Region and State': 'State'}, inplace=True)

cols = spending_cleaned_df.columns[1:]

for col in cols:
    spending_cleaned_df[col] = spending_cleaned_df[col].str.replace(',', '')
    spending_cleaned_df[col] = spending_cleaned_df[col].astype(int)

column_name_mapping = {
    'Public Welfare': 'Spendings on Public Welfare',
    'Public Hospitals': 'Spendings on Public Hospitals',
    'Highways': 'Spending on Highways',
    'Police': 'Spending on Police'
}

spending_cleaned_df.rename(columns=column_name_mapping, inplace=True)


states_df = pd.read_csv("https://raw.githubusercontent.com/prekshi99/CIS-595-Big-Data-Project/main/datasets/us_states_full_and_abbrev.csv")
states_df.rename(columns={'State Full Name': 'State'}, inplace=True)

spending_merged_df = pd.merge(spending_cleaned_df, states_df, on="State")
spending_merged_df.drop(columns = ['State'], inplace = True)
spending_merged_df.rename(columns = {'State Abbreviation': 'State'}, inplace = True)
spending_merged_df


merged_df = pd.merge(real_estate_dataset_df_cleaned, state_gdp_df, how='left', on='State')

merged_df = pd.merge(merged_df, crime_data_df, how='left', on='State')

merged_df = pd.merge(merged_df, weather_score_df, how='left', on = 'State')

merged_df = pd.merge(merged_df, spending_merged_df, how='left', on='State')

merged_df = merged_df.drop(columns = ['Total', 'Other'])

merged_df


merged_df = pd.read_csv('https://raw.githubusercontent.com/XuanyouLiu/US-Real-Estate-Analysis/main/Datasets/preprocessed_data.csv')



corrl = merged_df.corr(numeric_only = True)
plt.figure(figsize=(13, 13))
sns.heatmap(corrl,
            cbar=True,
            square=True,
            fmt='.1f',
            annot=True,
            annot_kws={'size': 12},
            cmap='coolwarm',
            linewidths=.5,
            linecolor='gray')

cbar = plt.gcf().axes[-1]
cbar.set_aspect(20)

plt.title('Heatmap of Correlation', fontsize=15)

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()

price_data = merged_df['price']

bins = [0, 50, 100, 500, 1000, 2000, 4000, 6000, 8000, 10000, 1000000, 3000000000]
bin_labels = ['<50', '50-100', '100-500', '500- 1000', '1000 - 2000', '2000 - 4000', '4000-6000', '6000-8000', '8000-10000', '10000 - 1M', '1M+']

bin_counts = np.histogram(price_data, bins=bins)[0]

plt.figure(figsize=(10, 5))
plt.bar(bin_labels, bin_counts)
plt.yscale('log')

plt.xlabel('Price Bins')
plt.ylabel('Number of Properties (log scale)')
plt.title('Real Estate Price Distribution')

for i, count in enumerate(bin_counts):
    plt.text(i, count, str(count), ha='center', va='bottom')

plt.xticks(rotation=45, ha='right')

plt.show()



sqfeet_data = merged_df['sqfeet']

bins = [0, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 9000000]
bin_labels = ['< 100', '100 - 500', '500 - 1000', '1000 - 1500', '1500-2000', '2000-2500', '2500-3000', '3000-3500', '3500-4000', '4000+']

bin_counts = np.histogram(sqfeet_data, bins=bins)[0]



plt.figure(figsize=(10, 5))
plt.bar(bin_labels, bin_counts)
plt.yscale('log')


plt.xlabel('Square feet Bins')
plt.ylabel('Number of properties (log scale)')
plt.title('Square feet Distribution')
plt.xticks(rotation=45)
for i, count in enumerate(bin_counts):
    plt.text(i, count, str(count), ha='center', va='bottom')
plt.show()



import numpy as np
import matplotlib.pyplot as plt

bath_data = merged_df['baths']

bed_data = merged_df['beds']

bed_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1500]
bed_bin_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10+']
bed_bin_counts = np.histogram(bed_data, bins=bed_bins)[0]

bath_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1500]
bath_bin_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10+']
bath_bin_counts = np.histogram(bath_data, bins=bath_bins)[0]

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.bar(bed_bin_labels, bed_bin_counts)
plt.xlabel('Beds Bins')
plt.ylabel('Number of Properties')
plt.title('Number of Beds Distribution')
for i, count in enumerate(bed_bin_counts):
    plt.text(i, count, str(count), ha='center', va='bottom')

plt.subplot(1, 2, 2)
plt.bar(bath_bin_labels, bath_bin_counts)
plt.xlabel('Baths Bins')
plt.ylabel('Number of Properties')
plt.title('Number of Baths Distribution')
for i, count in enumerate(bath_bin_counts):
    plt.text(i, count, str(count), ha='center', va='bottom')

plt.tight_layout()
plt.show()



merged_df_without_outliers = merged_df[(merged_df['price'] >= 100) & (merged_df['price'] <= 10000)]

merged_df_without_outliers = merged_df_without_outliers[merged_df_without_outliers['sqfeet'] >= 100]

merged_df_without_outliers = merged_df_without_outliers[(merged_df_without_outliers['beds'] <= 5) & (merged_df_without_outliers['baths'] <= 4)]


correlation_matrix = merged_df_without_outliers.corr(numeric_only = True)
plt.figure(figsize=(13, 13))
sns.heatmap(correlation_matrix,
            cbar=True,
            square=True,
            fmt='.1f',
            annot=True,
            annot_kws={'size': 12},
            cmap='PuOr',
            linewidths=.5,
            linecolor='gray')

cbar = plt.gcf().axes[-1]
cbar.set_aspect(20)

plt.title('Heatmap of Correlation', fontsize=15)

plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()




plt.figure(figsize=(15, 5))

ax = sns.boxplot(x="type", y="price", data=merged_df_without_outliers, palette='pastel')

ax.set_title("Price Distribution by Type")
ax.set_xlabel("Type")
ax.set_ylabel("Price ($)")
ax.set_ylim(0, 10000)

plt.show()



plt.figure(figsize=(15, 5))

ax = sns.boxplot(x="parking_options", y="price", data=merged_df_without_outliers, palette='pastel')

ax.set_title("Price Distribution by Parking Options")
ax.set_xlabel("Type")
ax.set_ylabel("Price ($)")
ax.set_ylim(0, 10000)

plt.show()



plt.figure(figsize=(15, 5))

ax = sns.boxplot(x="laundry_options", y="price", data=merged_df_without_outliers, palette='pastel')

ax.set_title("Price Distribution by Laundry Options")
ax.set_xlabel("Laundry Options")
ax.set_ylabel("Price ($)")
ax.set_ylim(0, 10000)

plt.show()



sqfeet = merged_df_without_outliers["sqfeet"]
price = merged_df_without_outliers["price"]

plt.figure(figsize=(15, 5))

plt.scatter(sqfeet, price, alpha=0.5)

plt.title("Square Feet vs. Price Scatter Plot", fontsize=16)
plt.xlabel("Square Feet", fontsize=14)
plt.ylabel("Price", fontsize=14)

plt.ylim(0, 10000)
plt.xlim(0, 10000)

plt.grid(True)
plt.show()
unique_state_list = merged_df_without_outliers['State'].unique().tolist()

merged_df_10_per = pd.DataFrame()

for state in unique_state_list:
    merged_df_state_length = int(len(merged_df_without_outliers[merged_df_without_outliers['State'] == state]) * 0.1)
    state_df = merged_df_without_outliers[merged_df_without_outliers['State'] == state][:merged_df_state_length]
    merged_df_10_per = merged_df_10_per.append(state_df, ignore_index=True)

import folium

f = folium.Figure(width=1000, height=1000)

m = folium.Map(width=1000, height=1000, location=[39.8283, -98.5795], zoom_start=4).add_to(f)

merged_df_10_per.apply(lambda row: folium.Circle(
   location=[row["lat"], row["long"]],
   color="crimson",
   fill=False,
).add_to(m), axis=1)

sns.set_style("whitegrid")

plt.figure(figsize=(20, 5))

ax = sns.boxplot(y="price", x="State", data=merged_df_without_outliers, palette='pastel')

ax.set_ylim(0, 10000)

ax.set_title('Comparison of House Rental Prices by State', fontsize=20, fontweight='bold')
ax.set_xlabel('State', fontsize=14, fontweight='bold')
ax.set_ylabel('Price (USD)', fontsize=14, fontweight='bold')

ax.tick_params(axis='y', labelsize=12)

ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

plt.subplots_adjust(bottom=0.15)

plt.tight_layout()
plt.show()



from folium.plugins import HeatMap


f = folium.Figure(width=1000, height=1000)

m = folium.Map(width=1000, height=1000, location=[39.8283, -98.5795], zoom_start=4).add_to(f)

data = merged_df_without_outliers[['lat', 'long', 'price']].values.tolist()

HeatMap(data, radius=14).add_to(m)

m




import folium
from folium.plugins import HeatMap
import plotly.graph_objects as go
from IPython.display import display


fig = go.Figure(data=go.Choropleth(
    z=spending_merged_df['Spending on Police'],
    locations=spending_merged_df['State'],
    locationmode='USA-states',
    colorscale='Blues',
    colorbar_title="Spending on Police",
    zmin=0,
    zmax=700,
    hovertemplate='State: %{location}<br>Spending on Police: %{z}<extra></extra>'
))

fig.update_layout(
    title_text='Spending on Police',
)


from IPython.core.display import HTML
display(HTML('<table><tr><td>' + m._repr_html_() + '</td><td>' + fig.to_html(full_html=False) + '</td></tr></table>'))



import plotly.express as px

spending_merged_df = spending_merged_df.sort_values('Spending on Police', ascending=False)

fig = px.bar(spending_merged_df, x='State', y='Spending on Police', text='Spending on Police',
             title='Spending on Police by State', labels={'Spending on Police': 'Spending on Police'})

fig.update_layout(xaxis=dict(tickangle=45, tickmode='array', tickvals=list(range(len(spending_merged_df))),
                             ticktext=spending_merged_df['State']))

fig.show()

import folium
from folium.plugins import HeatMap
import plotly.graph_objects as go
from IPython.display import display

m = folium.Map(location=[39.8283, -98.5795], zoom_start=3.4).add_to(f)
data = merged_df_without_outliers[['lat', 'long', 'price']].values.tolist()
HeatMap(data, radius=14).add_to(m)

fig = go.Figure(data=go.Choropleth(
    z=spending_merged_df['Population (Thousands)'],
    locations=spending_merged_df['State'],
    locationmode='USA-states',
    colorscale='Greens',
    colorbar_title="Population (Thousands)",
    hovertemplate='State: %{location}<br>Population (Thousands): %{z}<extra></extra>'
))


fig.update_layout(
    title_text='Population (Thousands)',
)


from IPython.core.display import HTML
display(HTML('<table><tr><td>' + m._repr_html_() + '</td><td>' + fig.to_html(full_html=False) + '</td></tr></table>'))

spending_merged_df = spending_merged_df.sort_values('Population (Thousands)', ascending=False)

fig = px.bar(spending_merged_df, x='State', y='Population (Thousands)', text='Population (Thousands)',
             title='Population (Thousands) by State', labels={'Population (Thousands)': 'Population (Thousands)'})

fig.update_layout(xaxis=dict(tickangle=45, tickmode='array', tickvals=list(range(len(spending_merged_df))),
                             ticktext=spending_merged_df['State']))

fig.show()



m = folium.Map(location=[39.8283, -98.5795], zoom_start=3.4).add_to(f)
data = merged_df_without_outliers[['lat', 'long', 'price']].values.tolist()
HeatMap(data, radius=14).add_to(m)

fig = go.Figure(data=go.Choropleth(
    z= state_gdp_df['GDP per capita'],
    locations=state_gdp_df['State'],
    locationmode='USA-states',
    colorscale='Reds',
    colorbar_title="GDP per capitas",
    hovertemplate='State: %{location}<br>GDP per capita: %{z}<extra></extra>',
    zmin=0,
    zmax=165000

))

fig.update_layout(
    title_text='USA Spending on Higher Education by State',
)

from IPython.core.display import HTML
display(HTML('<table><tr><td>' + m._repr_html_() + '</td><td>' + fig.to_html(full_html=False) + '</td></tr></table>'))

import plotly.express as px

fig = px.bar(state_gdp_df, x='State', y='GDP per capita', text='GDP per capita',
             title='GDP per capita by State', labels={'GDP per capita': 'GDP per capita'})

fig.update_layout(xaxis=dict(tickangle=45, tickmode='array', tickvals=list(range(len(state_gdp_df))),
                             ticktext=state_gdp_df['State']))

fig.show()



m = folium.Map(location=[39.8283, -98.5795], zoom_start=3.4).add_to(f)
data = merged_df_without_outliers[['lat', 'long', 'price']].values.tolist()
HeatMap(data, radius=14).add_to(m)

fig = go.Figure(data=go.Choropleth(
    z=spending_merged_df['Spendings on Public Welfare'],
    locations=spending_merged_df['State'],
    locationmode='USA-states',
    colorscale='Reds',
    colorbar_title="Public Welfare",
    hovertemplate='State: %{location}<br>Spending on Public Hospitals: %{z}<extra></extra>',
    colorbar=dict(
    )
))

fig.update_layout(
    title_text='USA Spending on Public Welfare by State',
    geo_scope='usa',
    height=500
)

from IPython.core.display import HTML
display(HTML('<table><tr><td>' + m._repr_html_() + '</td><td>' + fig.to_html(full_html=False) + '</td></tr></table>'))

spending_merged_df = spending_merged_df.sort_values('Spendings on Public Welfare', ascending=False)

fig = px.bar(spending_merged_df, x='State', y='Spendings on Public Welfare', text='Spendings on Public Welfare',
             title='Spending on Public Welfare by State', labels={'Spendings on Public Welfare': 'Spendings on Public Welfare'})

fig.update_layout(xaxis=dict(tickangle=45, tickmode='array', tickvals=list(range(len(spending_merged_df))),
                             ticktext=spending_merged_df['State']))

fig.show()




parking_option_ratings = {
}

laundry_option_ratings = {
}

merged_df['parking_options'] = merged_df['parking_options'].replace(parking_option_ratings)
merged_df['laundry_options'] = merged_df['laundry_options'].replace(laundry_option_ratings)

merged_df_without_outliers['parking_options'] = merged_df['parking_options'].replace(parking_option_ratings)
merged_df_without_outliers['laundry_options'] = merged_df['laundry_options'].replace(parking_option_ratings)

merged_df_without_outliers


merged_df['pets_allowed'] = merged_df['cats_allowed'] & merged_df['dogs_allowed']

columns_to_drop = ['cats_allowed', 'dogs_allowed']
merged_df = merged_df.drop(columns=columns_to_drop)

merged_df_without_outliers['pets_allowed'] = merged_df_without_outliers['cats_allowed'] & merged_df_without_outliers['dogs_allowed']

columns_to_drop = ['cats_allowed', 'dogs_allowed']
merged_df_without_outliers = merged_df_without_outliers.drop(columns=columns_to_drop)


merged_df_final = pd.read_csv('https://raw.githubusercontent.com/XuanyouLiu/US-Real-Estate-Analysis/main/Datasets/final_merged_df.csv')
merged_df_without_outliers_final = pd.read_csv('https://raw.githubusercontent.com/XuanyouLiu/US-Real-Estate-Analysis/main/Datasets/final_merged_df_without_outliers.csv')



merged_df_with_outliers_one_hot = merged_df_final.copy()
merged_df_with_outliers_one_hot = pd.get_dummies(merged_df_with_outliers_one_hot, columns=['type'], prefix='type')

merged_df_without_outliers_one_hot = merged_df_without_outliers_final.copy()
merged_df_without_outliers_one_hot = pd.get_dummies(merged_df_without_outliers_one_hot, columns=['type'], prefix='type')


merged_df_with_outliers_one_hot = merged_df_with_outliers_one_hot.dropna()

features_with_outliers_one_hot = merged_df_with_outliers_one_hot.drop(['price', 'State'], axis = 1)

target_with_outliers_one_hot = merged_df_with_outliers_one_hot['price']

seed = 42
X_train_with_outliers_one_hot, X_test_with_outliers_one_hot, y_train_with_outliers_one_hot, y_test_with_outliers_one_hot = train_test_split(features_with_outliers_one_hot, target_with_outliers_one_hot, train_size = 0.8, random_state = seed)

merged_df_without_outliers_one_hot = merged_df_without_outliers_one_hot.dropna()

features_without_outliers_one_hot = merged_df_without_outliers_one_hot.drop(['price', 'State'], axis = 1)

target_without_outliers_one_hot = merged_df_without_outliers_one_hot['price']

seed = 42
X_train_without_outliers_one_hot, X_test_without_outliers_one_hot, y_train_without_outliers_one_hot, y_test_without_outliers_one_hot = train_test_split(features_without_outliers_one_hot, target_without_outliers_one_hot, train_size = 0.8, random_state = seed)


X_train_without_outliers_one_hot

scaler = StandardScaler()

X_train_onehot_n_scaled = scaler.fit_transform(X_train_without_outliers_one_hot)

pca = PCA()

X_train_onehot_n_initial_fit = pca.fit(X_train_onehot_n_scaled)

explained_variance_ratios = pca.explained_variance_ratio_

cum_evr = np.cumsum(explained_variance_ratios)


plt.plot(cum_evr)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs Number of Components')
plt.axhline(y = 0.8, color = 'r', linestyle = '-')
plt.grid()
plt.show()

X_test_onehot_n_scaled = scaler.transform(X_test_without_outliers_one_hot)
pca_final = PCA(n_components = 18)
X_train_onehot_pca = pca_final.fit_transform(X_train_onehot_n_scaled)
X_test_onehot_pca = pca_final.transform(X_test_onehot_n_scaled)


<<<<<<< HEAD
=======


>>>>>>> 126c90fe18724879784982ecffe6c1fd5f1edd22
reg = linear_model.LinearRegression()
reg.fit(X_train_with_outliers_one_hot, y_train_with_outliers_one_hot)

y_pred_lin_reg_with_outliers_one_hot = reg.predict(X_test_with_outliers_one_hot)

lin_reg_score_test = reg.score(X_test_with_outliers_one_hot, y_test_with_outliers_one_hot)
lin_reg_score_train = reg.score(X_train_with_outliers_one_hot, y_train_with_outliers_one_hot)

mse = mean_squared_error(y_test_with_outliers_one_hot, y_pred_lin_reg_with_outliers_one_hot)

print('Testing Set R^2 value: ' + str(lin_reg_score_test))
print('Training Set R^2 value: ' + str(lin_reg_score_train))
print('Mean Squared Error: ' + str(mse))


reg = linear_model.LinearRegression()
reg.fit(X_train_without_outliers_one_hot, y_train_without_outliers_one_hot)

y_pred_lin_reg_without_outliers_one_hot = reg.predict(X_test_without_outliers_one_hot)

lin_reg_score_test = reg.score(X_test_without_outliers_one_hot, y_test_without_outliers_one_hot)
lin_reg_score_train = reg.score(X_train_without_outliers_one_hot, y_train_without_outliers_one_hot)

mse = mean_squared_error(y_test_without_outliers_one_hot, y_pred_lin_reg_without_outliers_one_hot)

print('Testing Set R^2 value: ' + str(lin_reg_score_test))
print('Training Set R^2 value: ' + str(lin_reg_score_train))
print('Mean Squared Error: ' + str(mse))


reg = linear_model.LinearRegression()
reg.fit(X_train_onehot_pca, y_train_without_outliers_one_hot)

y_pred_lin_reg_without_outliers_one_hot = reg.predict(X_test_onehot_pca)

lin_reg_score_test = reg.score(X_test_onehot_pca, y_test_without_outliers_one_hot)
lin_reg_score_train = reg.score(X_train_onehot_pca, y_train_without_outliers_one_hot)

mse = mean_squared_error(y_test_without_outliers_one_hot, y_pred_lin_reg_without_outliers_one_hot)

print('Testing Set R^2 value: ' + str(lin_reg_score_test))
print('Training Set R^2 value: ' + str(lin_reg_score_train))
print('Mean Squared Error: ' + str(mse))


<<<<<<< HEAD
=======


>>>>>>> 126c90fe18724879784982ecffe6c1fd5f1edd22
xgb1 = XGBRegressor()

parameters = {
    'objective':['reg:squarederror'],
    'learning_rate': [.01, 0.04, .06],
    'max_depth': [5, 6, 7],
    'n_estimators': [500]
}

xgb_grid = GridSearchCV(
    xgb1,
    parameters,
    cv = 2,
    n_jobs = 5,
    verbose=True
)

xgb_grid.fit(
    X_train_without_outliers_one_hot,
    y_train_without_outliers_one_hot
)

print(xgb_grid.best_score_)
print("\n".join(f"{key}: {val}" for key, val in xgb_grid.best_params_.items()))


best_xgb_model = xgb_grid.best_estimator_

y_pred_test = best_xgb_model.predict(X_test_without_outliers_one_hot)

xgb_score_train = xgb_grid.best_score_
xgb_score_test = r2_score(y_test_without_outliers_one_hot, y_pred_test)

mse = mean_squared_error(y_test_without_outliers_one_hot, y_pred_test)

print('Testing Set R^2 value: ' + str(xgb_score_test))
print('Training Set R^2 value: ' + str(xgb_score_train))
print('Mean Squared Error: ' + str(mse))


importances = best_xgb_model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': X_train_without_outliers_one_hot.columns,
    'Importance': importances
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

print(feature_importance_df.head(10))




rf_regressor = RandomForestRegressor()

parameters = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10]
}

rf_grid = GridSearchCV(rf_regressor, parameters, cv=2, n_jobs=5, verbose=True)

rf_grid.fit(X_train_without_outliers_one_hot, y_train_without_outliers_one_hot)

print(rf_grid.best_score_)
print("\n".join(f"{key}: {val}" for key, val in rf_grid.best_params_.items()))


best_rf_model = rf_grid.best_estimator_

y_pred_test = best_rf_model.predict(X_test_without_outliers_one_hot)

rf_score_train = rf_grid.best_score_
rf_score_test = r2_score(y_test_without_outliers_one_hot, y_pred_test)

mse = mean_squared_error(y_test_without_outliers_one_hot, y_pred_test)

print('Testing Set R^2 value: ' + str(rf_score_test))
print('Training Set R^2 value: ' + str(rf_score_train))
print('Mean Squared Error: ' + str(mse))


importances = best_rf_model.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': X_train_without_outliers_one_hot.columns, 'Importance': importances})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).reset_index(drop = True)

print(feature_importance_df.head(10))




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_without_outliers_one_hot)
X_test_scaled = scaler.transform(X_test_without_outliers_one_hot)

X_train_tensor = torch.tensor(X_train_scaled).float()
X_test_tensor = torch.tensor(X_test_scaled).float()
y_train_tensor = torch.tensor(y_train_without_outliers_one_hot.values).float()
y_test_tensor = torch.tensor(y_test_without_outliers_one_hot.values).float()

y_train_tensor = y_train_tensor.unsqueeze(1)
y_test_tensor = y_test_tensor.unsqueeze(1)




import torch
import torch.nn as nn

class CustomNet(nn.Module):
    def __init__(self, input_size, hidden1_size=64, hidden2_size=32, dropout_rate=0.2):
        super(CustomNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden1_size)
        self.layer2 = nn.Linear(hidden1_size, hidden2_size)
        self.layer3 = nn.Linear(hidden2_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

input_size = X_train_tensor.shape[1]
model = CustomNet(input_size=input_size, hidden1_size=128, hidden2_size=64, dropout_rate=0.3)


from sklearn.model_selection import ParameterGrid

param_grid = {
    'learning_rate': [0.0001, 0.001, 0.01, 0.1],
    'betas': [(0.9, 0.999), (0.95, 0.999), (0.9, 0.9999)],
    'hidden1_size': [64, 128],
    'hidden2_size': [32, 64],
    'dropout_rate': [0.2, 0.3]
}

pgrid = ParameterGrid(param_grid)

best_r_square = 0
best_params = {}
best_model = None

batch_size = 64
num_epochs = 10

batch_start = torch.arange(0, len(X_train_tensor), batch_size)

X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)

for params in pgrid:
  fnn_tuned = CustomNet(input_size, hidden1_size=params['hidden1_size'],
                      hidden2_size=params['hidden2_size'],
                      dropout_rate=params['dropout_rate']).to(device).to(device)
  loss_fn = nn.MSELoss()
  optimizer = optim.Adam(fnn_tuned.parameters(), lr=params['learning_rate'], betas=params['betas'])

  for epoch in range(num_epochs):

    total_train_loss = 0

    for start in batch_start:

        X_batch = X_train_tensor[start: start + batch_size]
        y_batch = y_train_tensor[start: start + batch_size]

        y_pred = fnn_tuned(X_batch)

        loss = loss_fn(y_pred, y_batch)

        total_train_loss += loss.item() * X_batch.size(0)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    avg_train_loss = total_train_loss / len(X_train_tensor)

    y_pred = fnn(X_test_tensor)

    mse = loss_fn(y_pred, y_test_tensor).item()

    y_test_np = y_test_tensor.cpu().numpy()
    y_pred_np = y_pred.cpu().detach().numpy()

    r_square = r2_score(y_test_np, y_pred_np)

  if r_square > best_r_square:
    best_r_square = r_square
    best_params = best_params = params
    best_model = fnn_tuned


print(f"Best Model Accuracy: {best_r_square}")
print(f"Best Parameters: {best_params}")
class FNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(X_train_tensor.shape[1], 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x
    
fnn = FNN().to(device)

loss_fn = nn.MSELoss()

optimizer = optim.Adam(fnn.parameters(), lr=0.001, betas = (0.9, 0.9999))

n_epochs = 30
batch_size = 64

batch_start = torch.arange(0, len(X_train_tensor), batch_size)

X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)

r_2 = []

for epoch in range(n_epochs):

    total_train_loss = 0

    for start in batch_start:

        X_batch = X_train_tensor[start: start + batch_size]
        y_batch = y_train_tensor[start: start + batch_size]

        y_pred = fnn(X_batch)

        loss = loss_fn(y_pred, y_batch)

        total_train_loss += loss.item() * X_batch.size(0)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    avg_train_loss = total_train_loss / len(X_train_tensor)

    y_pred = fnn(X_test_tensor)

    mse = loss_fn(y_pred, y_test_tensor).item()

    y_test_np = y_test_tensor.cpu().numpy()
    y_pred_np = y_pred.cpu().detach().numpy()

    r_square = r2_score(y_test_np, y_pred_np)

    r_2.append(r_square)

    print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss}, Test MSE: {mse}, R-squared: {r_square}')



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (10,5))

plt.plot(range(1, 31), r_2)

plt.xlim(1,30)
plt.xticks(range(1,31))
plt.xlabel('Epochs')
plt.ylabel('R Square')
plt.title('Epochs vs Testing R Square (FNN)')
plt.grid()
plt.tight_layout()

plt.show()
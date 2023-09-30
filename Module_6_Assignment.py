"""
Copyright (c) 2022
Written by : Mehak Gagneja
Description: CRIME INCIDENT REPORTS - 2020 Data Analysis
"""

# import the required library
import pandas as pd
import Module_6_Assignment_Service
import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

import readFileData

# Get the data from URL
df = pd.read_csv('https://data.boston.gov/dataset/6220d948-eae2-4e4b-8723-2dc8e67722a3/resource/be047094-85fe-4104-a480-4fa3d03f9623/download/script_113631134_20210423193017_combine.csv')

#Get the data from CSV file
#UNCOMMNET THE BELOW LOC IF THE URL DOES NOT WORK!!
#df = readFileData.readCSVFile('CRIME_INCIDENT_REPORTS_2020.csv', ',', None)


# Dataframe structure
print('\n Dataframe has', df.shape, 'rows and columns respectively')

print('\n Initial Column of the dataframe are-', df.columns.values)

#Data Cleaning and Preparation:

# 1. Remove unwanted columns
df.drop(['OFFENSE_CODE_GROUP', 'Lat', 'Long', 'UCR_PART', 'Location'], axis=1, inplace=True)
print('\n Final Column of the dataframe are-', df.columns.values)

# 2. Check for null values and process them
print('\n Column wise total number of null values: \n', df.isnull().sum())
# Remove rows that have null values for District column 
df = df.dropna(subset=['DISTRICT'], how='all')

# 3. Add new columns that are required for Analysis
df['DATE'] = pd.to_datetime(df['OCCURRED_ON_DATE']).dt.date
holidays = ['2020-02-20', '2020-09-10', '2020-11-26', '2020-12-31']
df['HOLIDAY'] = df["DATE"].astype(str).isin(holidays)

# 4. Replace column values for better visualization
df['OFFENSE_DESCRIPTION'] = df['OFFENSE_DESCRIPTION'].str.replace('WEAPON VIOLATION - CARRY/ POSSESSING/ SALE/ TRAFFICKING/ OTHER','WEAPON VIOLATION')

print(df.columns)
print(df.head())

# Visualizations

# 1. What are the top 10 offenses reported in Boston during the year 2020?
topOffenses = df.groupby('OFFENSE_DESCRIPTION').size().nlargest(10)
fig1 = plt.figure()
ax = fig1.add_axes([0,0,1,1])
plt.bar(
        topOffenses.keys(),
        topOffenses.values,
        width = 0.4
        )
plt.ylabel('Offense')
plt.ylabel('Frequency')
plt.title('Top 10 Offenses Reported in Boston')
plt.xticks(rotation=30, ha='right')
plt.show()

# 2. How many times were these top 10 offenses accompanied by shooting?
shootingOffeses = df[df.SHOOTING == 1]
topShootingOffenses = shootingOffeses.groupby('OFFENSE_DESCRIPTION').size().nlargest(10)
bar_plot = sns.barplot(
    x=topShootingOffenses.keys(),
    y=topShootingOffenses.values).set(
        title='Top 10 Offenses With Shooting Reported in Boston',
        xlabel='Offense', ylabel='Frequency')
plt.xticks(rotation=45, ha='right')
plt.show()

# 3. Crime Distribution in each hours of Days?
crimesByDaysOfWeekAndHours = df.groupby(['DAY_OF_WEEK', 'HOUR']).size().reset_index()
crimesByDaysOfWeekAndHours.columns = ['DAY_OF_WEEK', 'HOUR', 'COUNT']
crimesByDayAndHour = crimesByDaysOfWeekAndHours.pivot('DAY_OF_WEEK', 'HOUR', 'COUNT')
ax = sns.heatmap(crimesByDayAndHour, cmap = sns.cm.rocket_r)
plt.title("Heatmap Crimes By Day And Hours")
plt.show()

# 4. How many crimes are reported in each district category?

#to categorize districts into categories, create a new column for the dataset
df['DISTRICT_CATEGORY'] = df['DISTRICT'].str[:1]

crime_districtcategory = df.groupby('DISTRICT_CATEGORY', as_index = False)['INCIDENT_NUMBER'].count()

bar = sns.barplot(x = 'DISTRICT_CATEGORY',
           y = 'INCIDENT_NUMBER',
           data = crime_districtcategory)

plt.title('Crimes reported in each district category')
plt.xticks(rotation=30, ha='right')
plt.xlabel('Dsitrict Categories')
plt.ylabel('Frequency')

plt.show()         

# 5. What is the overall percentage of shootings involved in the crime incidents reported all around Boston?

shooting_involved = df.groupby('SHOOTING', as_index = False)['INCIDENT_NUMBER'].count()
shooting_involved['Percentage'] = (shooting_involved['INCIDENT_NUMBER'] / shooting_involved['INCIDENT_NUMBER'].sum()) * 100
shooting_involved['Percentage'] = round(shooting_involved['Percentage'],1)

shooting_involved['SHOOTING'] = shooting_involved['SHOOTING'].replace([0],'No Shooting')
shooting_involved['SHOOTING'] = shooting_involved['SHOOTING'].replace([1],'Shooting')

labels = shooting_involved['SHOOTING']+ "_" + shooting_involved['Percentage'].astype(str)+'%'
plt.pie(shooting_involved['Percentage'], shadow=False, startangle=60)
plt.legend(labels, loc="lower left")
plt.title('Percentage of shootings involved in the crime incidents')
plt.show()


# 6. What is the trend of the incidents reported over the time period of the whole year?
crimesByMonth = df.groupby(['MONTH', 'DISTRICT']).size().reset_index()
crimesByMonth.columns = ['MONTH', 'DISTRICT', 'COUNT']
g = sns.lineplot(data = crimesByMonth, x='MONTH', y = 'COUNT', hue='DISTRICT').set(
    title='Incident Reported Over the Time Period 2020',
    xlabel='Month', ylabel='Count')
# Put a legend to the right side
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.show()

# 7. What is the ratio of crimes reported on weekdays Vs weekends?
crimesByDaysOfWeek = df.groupby(['DAY_OF_WEEK']).size().reset_index()
crimesByDaysOfWeek.columns = ['DAY_OF_WEEK', 'COUNT']
crimesByDaysOfWeek['DAY_OF_WEEK'] = pd.Categorical(crimesByDaysOfWeek['DAY_OF_WEEK'],
                                   categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                                   ordered=True)
crimeTrend = sns.lineplot(data = crimesByDaysOfWeek, x='DAY_OF_WEEK', y = 'COUNT').set(
    title='Crimes Reported For Each Day of Week',
    xlabel='Day of Week', ylabel='Count')
plt.xticks(rotation=45, ha='right')
plt.show()

# 8. At what different parts of the day are these incidents reported the most?
hourlyCrimeTrent = sns.countplot(data=df, x='HOUR').set(
    title='Crimes Reported For Each Hour of the Day',
    xlabel='Hour', ylabel='Count')
plt.show()

# 9. What are the top street locations where crimes are reported the most?

topOffensesStreet = df.groupby('STREET').size().nlargest(10)
fig1 = plt.figure()
ax = fig1.add_axes([0,0,1,1])
plt.bar(
        topOffensesStreet.keys(),
        topOffensesStreet.values,
        width = 0.4
        )
plt.ylabel('Street')
plt.ylabel('Frequency')
plt.title('Top 10 locations where incidents were reported in Boston')
plt.xticks(rotation=30, ha='right')
plt.show() 

# 10. Whether crime incidents were reported on US Holidays or not? If yes, how many of them occurred?

incidents_on_holidays = df.groupby('HOLIDAY', as_index = False)['INCIDENT_NUMBER'].count()

bar = sns.barplot(x = 'HOLIDAY',
           y = 'INCIDENT_NUMBER',
           data = incidents_on_holidays)

plt.title('Crime incidents reported on US Holidays')
plt.xticks(rotation=30, ha='right')
plt.xlabel('Holiday ')
plt.ylabel('Number of Incidents')

plt.show()

# Prediction Model
#Convert the categorical data into numerical using OneHotEncoder
categorical_cols = ['DISTRICT', 'DAY_OF_WEEK']
numerical_cols = ['MONTH', 'HOLIDAY', 'HOUR']
df_feats = df[['DISTRICT', 'DAY_OF_WEEK','MONTH', 'HOLIDAY', 'HOUR','SHOOTING']]

transformer = make_column_transformer(
    (OneHotEncoder(), categorical_cols),
    remainder = 'passthrough'
)
transformed = transformer.fit_transform(df_feats).toarray()
transformed_df = pd.DataFrame(transformed, columns=transformer.get_feature_names())
transformed_df['SHOOTING'] = transformed_df.SHOOTING.astype('int')
# Call method for k_neighbors prediction model
Module_6_Assignment_Service.k_neighbors_model(transformed_df)
# Call method for Decision Tree Classifier prediction model
Module_6_Assignment_Service.decisiontree_model(transformed_df)
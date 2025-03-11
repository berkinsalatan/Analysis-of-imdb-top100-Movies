#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:02:03 2023

@author: berkin
"""

import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('movies.csv')


column = ["votes","run_time","year_of_release"]

for i in range(len(df["votes"])):
    df["votes"][i] = df["votes"][i].replace(",","")
    df["year_of_release"][i] = df["year_of_release"][i][1:-1]
    df["gross_total"][i] = df["gross_total"][i][1:-1]
    df["run_time"][i] = df["run_time"][i].replace("min","")

for k in column:
    df[k] = df[k].astype("int64")

df["gross_total"] = df["gross_total"].astype("float64")




df = df.fillna("$5.27M")
df.tail(50)




df.plot(x='index' , y='gross_total' , kind='scatter')
plt.xlabel('Movies(index numbers)')
plt.ylabel('Gross Total (Million $)')
plt.title("Do high-grossing movies have a high imdb score?")
plt.show()

df.plot(x='imdb_rating' , y='gross_total' , kind='scatter')
plt.xlabel('Movies imdb scores')
plt.ylabel('Gross Total (Million $)')
plt.title("Do high-grossing movies have a high imdb score?")
plt.show()

print(df['gross_total'].describe())
print(df['imdb_rating'].describe())




imdb_choosen_1 = df[(df['imdb_rating'] >= 8.3) & (df['imdb_rating'] <= 9.3)]
average = imdb_choosen_1['gross_total'].mean()
print("average : ", average)


imdb_choosen_2 = df[(df['imdb_rating'] >= 7.2) & (df['imdb_rating'] <= 8.2)]
average = imdb_choosen_2['gross_total'].mean()
print("average : ", average)





year_of_release1 = df[(df['year_of_release'] >=1931) & (df['year_of_release'] <=1939)]
average = year_of_release1['imdb_rating'].mean()
print("average : ", average)

year_of_release2 = df[(df['year_of_release'] >=1941) & (df['imdb_rating'] <=1959)]
average = year_of_release2['imdb_rating'].mean()
print("average : ", average)

year_of_release3 = df[(df['year_of_release'] >=1960) & (df['imdb_rating'] <=1979)]
average = year_of_release3['imdb_rating'].mean()
print("average : ", average)

year_of_release4 = df[(df['year_of_release'] >=1980) & (df['imdb_rating'] <=1999)]
average = year_of_release4['imdb_rating'].mean()
print("average : ", average)

year_of_release5 = df[(df['year_of_release'] >=2000) & (df['imdb_rating'] <=2015)]
average = year_of_release5['imdb_rating'].mean()
print("average : ", average)





#Import all libraries that we going to use
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Step 1: Load the dataset
data = pd.read_csv('Salary_dataset.csv')  

data.drop(['Unnamed: 0'],axis=1,inplace=True)

# Step 2: Split the data into features (X) and target variable (y)
X = data[['YearsExperience']]  
y = data['Salary']  


# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 4: Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# Step 5: Predict the target variable for the test set
y_pred = model.predict(X_test)


# Step 6: Evaluate the model using metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)



# Step 7: Print the evaluation results
print("Mean Squared Error:", mse)
print("R-squared:", r2)


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


data = pd.read_csv('Salary_dataset.csv')


X = data['YearsExperience'].values.reshape(-1, 1)
y = data['Salary'].values


model = LinearRegression()
model.fit(X, y)


y_pred = model.predict(X)


plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Linear Regression')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression - Salary vs Years of Experience')
plt.legend()
plt.show()









































import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# loading the data set by using pandas & and then checking for the rows and col with grtting info of data types and bytes 
df = pd.read_csv("data.csv")
print(df.head())
print(df.info())
df = df.dropna()
# Feature selection: selecting col to split in to two aprt to get the price of house
X = df[['sqft_living', 'bedrooms', 'city']]
y = df['price']

# Split data after spliting we have to specify the test size and also random  shuffal size to make predication best 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=22)

# column names here
num_features = ['sqft_living', 'bedrooms']
cate_features = ['city']

# Preprocessing: scale numerical + encode categorical / preparing data for the model
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cate', OneHotEncoder(), cate_features)
    ]
)

# Model pipeline to make the model predication better
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train model: trainning model to make the result 
model.fit(X_train, y_train)

# Predictions on the model 
y_pred = model.predict(X_test)

# Evaluation
#MAE = "On average, how wrong are we?"
#RMSE = "How wrong are we, especially when we make big mistakes?"


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)

# Visualization to represent the result
plt.figure(figsize=(7,4))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()


"""Results:MAE tells you, on average, how far off your house price guesses are from reality.
RMSE says the same but gives extra weight to those really bad misses.
Smaller numbers mean your predictions are much closer to what houses actually sell for"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from scipy.special import inv_boxcox
from sklearn.preprocessing import RobustScaler

# Get Year and Return Scaler
def get_cpi(year):
    cpi_data = pd.read_csv("cpi.csv")
    if (year > 2005) & (year < 2025):
        cpi_val = cpi_data.loc[(cpi_data["date_inflation"] == year), "CPI"].values[0]
        return cpi_val
    elif (year >= 2025):
        cpi_2025 = cpi_data.loc[(cpi_data["date_inflation"] == 2025), "CPI"].values[0]
        return cpi_2025
    else:
        return "Not Valid Year"

# 2- Get Scaler and Return DataFrame
def get_intervals(pred):
    #Confidence Level 99%
    t_crit = 2.5759671215567197
    stderr = 7050.850580914985 * 3
    # Calculate intervals
    lower = pred - t_crit * (stderr)
    upper = pred + t_crit * (stderr) 

    lower_upper = {
        'prediction': pred,
        'lower': lower,
        'upper': upper
    }
    lower_upper = pd.DataFrame(lower_upper, index=[0])
    return lower_upper

data_inflated = pd.read_csv("data_inflated.csv")
data_inflated.drop(columns="Unnamed: 0", inplace=True)
data_inflated = pd.DataFrame(data_inflated)

features = data_inflated.drop(columns=["prices", "core_inflation", "CPI", "Adjusted_Price"])

features["Price_per_m"] = features["Price_per_m"] / 10000
features["Price_per_m"] = np.log(features["Price_per_m"])
#features["Price_per_m"], lambda_fitted1 = stats.boxcox(features["Price_per_m"])


target = data_inflated["Adjusted_Price"]
target = target / 10000000
target, lambda_fitted = stats.boxcox(target)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=42)

y_train = pd.DataFrame(y_train)

# Scaling
scaler1 = RobustScaler()
scaler1.fit(X_train[["Price_per_m"]])

scaler = RobustScaler()
scaler.fit(y_train)


# 1- Get Scaler and Return Scaler
def transform(pred, scaler, lambda_fitted, year, num):
    pred_scaled = scaler.inverse_transform([[pred]])
    pred_scaled_inv = inv_boxcox(pred_scaled, lambda_fitted)
    pred_scaled_inv = pred_scaled_inv * num
    pred_scaled_inv = pred_scaled_inv.reshape(-1,1)

    base = get_cpi(2024)
    pred_transformed = pred_scaled_inv * (get_cpi(year) / base)

    return pred_transformed[0][0]



def get_pred(model, df):
    pred = model.predict(df)[0]
    return pred



def get_df(locations, Price_per_m, rooms, bathrooms, finishes, views, floors, years, payments, property_age):
    df = pd.DataFrame([[locations, Price_per_m, rooms, bathrooms,
                                 finishes, views, floors, years, payments, property_age]],
                               columns=['locations', 'Price_per_m', 'rooms', 'bathrooms', 'finishes', 'views', 'floors',
                                        'years', 'payments', 'property_age'])

    df["locations"] = df["locations"].map(locations_cat)[0]
    df["finishes"] = df["finishes"].map(finishes_cat)[0]
    df["views"] = df["views"].map(views_cat)[0]
    df["payments"] = df["payments"].map(payments_cat)[0]

    return df

locations_cat = {
    "other" : 1,
    "Greater Cairo" : 2,
    "Alexandria" : 3,
    "North Coast" : 4
}

finishes_cat = {
    'Without Finish': 1,
    'Semi Finished': 2,
    'Lux': 3,
    'Super Lux': 4,
    'Extra Super Lux': 5
}

views_cat = {
    'Corner': 1,
    'Garden': 2,
    'Main Street': 3,
    'Other': 4,
    'Pool': 5,
    'Sea View': 6,
    'Side Street': 7
}

payments_cat = {
    'Cash': 1,
    'Cash or Installments': 2,
    'Installments': 3
}




get_cpi(2010)



















'''
df = get_df("Alexandria",0.797966	,3,3,5,2,1,2025,2,-1)
print(df)
'''










'''
model = joblib.load('XGBRegressor.joblib')
df = get_df(2,0.797966	,3,3,5,2,1,2025,2,-1)
prediction = get_pred(model, df)
trans_val = transform(prediction, scaler,lambda_fitted, 2025,  10000000)
print(trans_val)
'''


'''
# print(get_intervals(trans_val))
print(f'Predicition = {trans_val}')
print(f'Min Value = {get_intervals(trans_val).iloc[0,1]}')
print(f'Max Value = {get_intervals(trans_val).iloc[0,2]}')
'''


# test_values = pd.DataFrame([[3,.6363,2,1,4.0,4,2,2020.0,1.0,4]], columns= ['locations', 'Price_per_m', 'rooms', 'bathrooms', 'finishes', 'views','floors', 'years', 'payments', 'property_age'])
#
# pred= get_predictions_and_intervals(model, test_values)
# pred = pd.DataFrame(pred, index= [0])
# pred.iloc[0,0]
#
#
# y_train = pd.read_csv("y_train.csv")
#
# scaler = RobustScaler()
# scaler.fit(y_train)





# data_inflated = pd.read_csv("data_inflated.csv")
#
# features = data_inflated.drop(columns=["prices", "core_inflation", "CPI", "Adjusted_Price"])
# features["Price_per_m"] = features["Price_per_m"] / 10000
#
# target = data_inflated["Adjusted_Price"]
# target = target / 10000000
# target, lambda_fitted = stats.boxcox(target)
#
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=42)
# y_train = pd.DataFrame(y_train)
# y_test = pd.DataFrame(y_test)
#


# y_test_scaled = scaler2.transform(y_test)
# def scaling(pred):
#
#
#
#     scaler = RobustScaler()
#     scaler.fit(y_train)
#     pred = pd.DataFrame([pred], index=[0])
#     pred = scaler.inverse_transform(pred)
#
#     return pred[0][0]


# Get Scaler and Return Scaler
# def scaling(pred):
#     y_train = pd.read_csv("y_train.csv")
#     y_train.drop(columns="Unnamed: 0", inplace=True)
#     y_train = pd.DataFrame(y_train)
#
#     scaler = RobustScaler()
#     scaler.fit(y_train)
#     pred = pd.DataFrame([pred], index=[0])
#     pred = scaler.inverse_transform(pred)
#
#     return pred[0][0]




# transform our prediction
# 0 - get_prediction
# 1 - scaling
# 2 - inv_boxcox
# 3 - cpi convert
# 4 - pred intervals


# 0 - get values and Return Scaler
# def get_pred(model, locations, Price_per_m, rooms, bathrooms, finishes, views, floors, years, payments, property_age):
#     test_values = pd.DataFrame([[locations, Price_per_m, rooms, bathrooms,
#                                  finishes, views, floors, years, payments, property_age]],
#                                columns=['locations', 'Price_per_m', 'rooms', 'bathrooms', 'finishes', 'views', 'floors',
#                                         'years', 'payments', 'property_age'])
#
#     return model.predict(test_values)[0]
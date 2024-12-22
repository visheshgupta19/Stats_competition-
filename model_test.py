#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np

team_data = pd.read_csv("../Stats_competition-/team_data_collected_df.csv")


# In[2]:


threshold = 6


# In[3]:


team_data


# In[4]:


team_data["Location"] = np.where(
    team_data["Location"] == "N", 0, np.where(team_data["Location"] == "H", 1, -1)
)


# In[5]:


columns_to_convert = [
    "Location",
    "ADJO",
    "ADJD",
    "EFG%",
    "TO%",
    "OR%",
    "FTR",
    "Opp EFG%",
    "Opp TO%",
    "Opp OR%",
    "Opp FTR",
]
for col in columns_to_convert:
    team_data[col] = pd.to_numeric(team_data[col], errors="coerce")


# In[6]:


team_data_cleaned = team_data.dropna()

X = team_data_cleaned[columns_to_convert]
y_team = team_data_cleaned["Team_score"]
y_opp = team_data_cleaned["Opponent_score"]

X_train, X_test, y_team_train, y_team_test = train_test_split(
    X, y_team, test_size=0.2, random_state=42
)
y_opp_train, y_opp_test = train_test_split(y_opp, test_size=0.2, random_state=42)

model_team = LinearRegression()
model_team.fit(X_train, y_team_train)
team_pred = model_team.predict(X_test)

model_opp = LinearRegression()
model_opp.fit(X_train, y_opp_train)
opp_pred = model_opp.predict(X_test)

team_rmse = mean_squared_error(y_team_test, team_pred, squared=False)
opp_rmse = mean_squared_error(y_opp_test, opp_pred, squared=False)
print(f"Team Score RMSE: {team_rmse}, Opponent Score RMSE: {opp_rmse}")


# In[7]:


team_accuracy = (abs(team_pred - y_team_test) <= threshold).mean() * 100
opp_accuracy = (abs(opp_pred - y_opp_test) <= threshold).mean() * 100
print(f"Team Score Accuracy: {team_accuracy:.2f}%")
print(f"Opponent Score Accuracy: {opp_accuracy:.2f}%")


# In[8]:


rf_model_team = RandomForestRegressor(random_state=42)
rf_model_team.fit(X_train, y_team_train)
rf_team_pred = rf_model_team.predict(X_test)

rf_model_opp = RandomForestRegressor(random_state=42)
rf_model_opp.fit(X_train, y_opp_train)
rf_opp_pred = rf_model_opp.predict(X_test)

team_rf_rmse = mean_squared_error(y_team_test, rf_team_pred, squared=False)
opp_rf_rmse = mean_squared_error(y_opp_test, rf_opp_pred, squared=False)
print(f"Team Score RMSE: {team_rf_rmse}, Opponent Score RMSE: {opp_rf_rmse}")


# In[9]:


team_accuracy = (abs(rf_team_pred - y_team_test) <= threshold).mean() * 100
opp_accuracy = (abs(rf_opp_pred - y_opp_test) <= threshold).mean() * 100
print(f"Team Score Accuracy: {team_accuracy:.2f}%")
print(f"Opponent Score Accuracy: {opp_accuracy:.2f}%")


# In[10]:


import xgboost as xgb

X_train, X_test, y_team_train, y_team_test = train_test_split(
    X, y_team, test_size=0.2, random_state=42
)
X_train_opp, X_test_opp, y_opp_train, y_opp_test = train_test_split(
    X, y_opp, test_size=0.2, random_state=42
)

dtrain_team = xgb.DMatrix(X_train, label=y_team_train)
dtest_team = xgb.DMatrix(X_test, label=y_team_test)

dtrain_opp = xgb.DMatrix(X_train_opp, label=y_opp_train)
dtest_opp = xgb.DMatrix(X_test_opp, label=y_opp_test)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

team_model = xgb.train(
    params,
    dtrain_team,
    num_boost_round=100,
    evals=[(dtest_team, "test")],
    early_stopping_rounds=10,
)

opp_model = xgb.train(
    params,
    dtrain_opp,
    num_boost_round=100,
    evals=[(dtest_opp, "test")],
    early_stopping_rounds=10,
)

xg_team_pred = team_model.predict(dtest_team)
xg_opp_pred = opp_model.predict(dtest_opp)

team_rmse = mean_squared_error(y_team_test, xg_team_pred, squared=False)
opp_rmse = mean_squared_error(y_opp_test, xg_opp_pred, squared=False)

print(f"Team Score RMSE: {team_rmse}")
print(f"Opponent Score RMSE: {opp_rmse}")


# In[11]:


team_accuracy = (abs(xg_team_pred - y_team_test) <= threshold).mean() * 100
opp_accuracy = (abs(xg_opp_pred - y_opp_test) <= threshold).mean() * 100
print(f"Team Score Accuracy: {team_accuracy:.2f}%")
print(f"Opponent Score Accuracy: {opp_accuracy:.2f}%")


# In[12]:


predict_games = pd.read_csv("../Stats_competition-/basketball_games_data.csv")


# In[13]:


predict_games["Location"] = np.where(
    predict_games["Location"] == "Neutral",
    0,
    np.where(predict_games["Location"] == "Home", 1, -1),
)


# In[14]:


predict_games


# In[15]:


X = predict_games[columns_to_convert]


# In[16]:


X


# In[17]:


team_pred_new = model_team.predict(X)
opp_pred_new = model_opp.predict(X)

predictions = pd.DataFrame(
    {
        "Predicted Team Score LR": team_pred_new,
        "Predicted Opponent Score LR": opp_pred_new,
    }
)


# In[18]:


rf_team_pred_new = rf_model_team.predict(X)
rf_opp_pred_new = rf_model_opp.predict(X)

predictions_rf = pd.DataFrame(
    {
        "Predicted Team Score RF": rf_team_pred_new,
        "Predicted Opponent Score RF": rf_team_pred_new,
    }
)


# In[19]:


predict_games = pd.concat([predict_games, predictions], axis=1)
predict_games = pd.concat([predict_games, predictions_rf], axis=1)


# In[21]:


predict_games[
    [
        "Team",
        "Opponent",
        "Predicted Team Score LR",
        "Predicted Opponent Score LR",
        "Predicted Team Score RF",
        "Predicted Opponent Score RF",
    ]
]


# In[ ]:

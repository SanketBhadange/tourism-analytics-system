import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Tourism Analytics System",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Tourism Experience Analytics & Recommendation System")

# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():

    BASE_DIR = os.path.dirname(__file__)
    file_path = os.path.join(BASE_DIR, "tourism_dataset.csv")

    df = pd.read_csv(file_path)

    return df

df = load_data()

st.success("Dataset loaded successfully")

# =========================
# FEATURE ENGINEERING
# =========================

le = LabelEncoder()

cols = ["Continent","Region","Country","CityName","VisitMode","AttractionType"]

for col in cols:
    df[col] = le.fit_transform(df[col].astype(str))

df["AttractionAvgRating"] = df.groupby("AttractionId")["Rating"].transform("mean")
df["UserAvgRating"] = df.groupby("UserId")["Rating"].transform("mean")

# =========================
# TRAIN MODELS
# =========================

@st.cache_resource
def train_models(df):

    features = [
        "UserId","VisitYear","VisitMonth","Continent","Region",
        "Country","CityName","AttractionType","AttractionId",
        "AttractionAvgRating","UserAvgRating"
    ]

    X = df[features]

    y_reg = df["Rating"]
    y_clf = df["VisitMode"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    reg_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    reg_model.fit(X_train, y_train)

    clf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    clf_model.fit(X_train, y_clf.loc[y_train.index])

    return reg_model, clf_model

reg_model, clf_model = train_models(df)

st.success("Models trained successfully")

# =========================
# RECOMMENDATION SYSTEM
# =========================

@st.cache_resource
def build_recommendation(df):

    user_item = df.pivot_table(
        index="UserId",
        columns="AttractionId",
        values="Rating"
    )

    user_item.fillna(0, inplace=True)

    similarity = cosine_similarity(user_item)

    return user_item, similarity

user_item, similarity = build_recommendation(df)

def recommend(user_id, n=5):

    if user_id >= len(similarity):
        return []

    sim_scores = similarity[user_id]

    similar_users = np.argsort(sim_scores)[::-1][1:6]

    rec = user_item.iloc[similar_users].mean()

    visited = user_item.iloc[user_id]

    rec = rec[visited == 0]

    rec = rec.dropna()

    rec = rec.sort_values(ascending=False)

    return rec.head(n).index.tolist()

# =========================
# UI TABS
# =========================

tab1, tab2, tab3 = st.tabs([
    "⭐ Predict Rating",
    "🚗 Predict Visit Mode",
    "🎯 Recommendation"
])

# =========================
# TAB 1
# =========================

with tab1:

    st.header("Predict Attraction Rating")

    user_id = st.number_input("User ID", 0, 10000, 1)
    attraction_id = st.number_input("Attraction ID", 0, 10000, 1)
    visit_year = st.number_input("Visit Year", 2000, 2030, 2024)
    visit_month = st.number_input("Visit Month", 1, 12, 1)

    continent = st.number_input("Continent Code", 0, 10, 1)
    region = st.number_input("Region Code", 0, 10, 1)
    country = st.number_input("Country Code", 0, 100, 1)
    city = st.number_input("City Code", 0, 1000, 1)
    attraction_type = st.number_input("Attraction Type", 0, 10, 1)

    attraction_avg = st.number_input("Attraction Avg Rating", 0.0, 5.0, 4.0)
    user_avg = st.number_input("User Avg Rating", 0.0, 5.0, 4.0)

    if st.button("Predict Rating"):

        features = np.array([[
            user_id, visit_year, visit_month, continent,
            region, country, city, attraction_type,
            attraction_id, attraction_avg, user_avg
        ]])

        pred = reg_model.predict(features)

        st.success(f"Predicted Rating: {pred[0]:.2f}")

# =========================
# TAB 2
# =========================

with tab2:

    st.header("Predict Visit Mode")

    if st.button("Predict Visit Mode"):

        features = np.array([[
            user_id, visit_year, visit_month, continent,
            region, country, city, attraction_type,
            attraction_id, attraction_avg, user_avg
        ]])

        pred = clf_model.predict(features)

        st.success(f"Visit Mode Code: {pred[0]}")

# =========================
# TAB 3
# =========================

with tab3:

    st.header("Get Recommendations")

    rec_user = st.number_input("Enter User ID", 0, 10000, 1)

    if st.button("Recommend"):

        recs = recommend(rec_user)

        if recs:

            for r in recs:
                st.write(f"Attraction ID: {r}")

        else:
            st.warning("No recommendations found")
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

st.title("Tourism Experience Analytics & Recommendation System")

# ========================
# LOAD DATA
# ========================

import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
file_path = os.path.join(BASE_DIR, "tourism_dataset.csv")

df = pd.read_csv(file_path)

# ========================
# FEATURE ENGINEERING
# ========================

le = LabelEncoder()

cols = ["Continent","Region","Country","CityName","VisitMode","AttractionType"]

for col in cols:
    df[col] = le.fit_transform(df[col].astype(str))

df["AttractionAvgRating"] = df.groupby("AttractionId")["Rating"].transform("mean")
df["UserAvgRating"] = df.groupby("UserId")["Rating"].transform("mean")

# ========================
# TRAIN MODELS
# ========================

@st.cache_resource
def train_models(df):

    features = [
        "UserId","VisitYear","VisitMonth","Continent","Region",
        "Country","CityName","AttractionType","AttractionId",
        "AttractionAvgRating","UserAvgRating"
    ]

    X = df[features]

    y_reg = df["Rating"]
    y_clf = df["VisitMode"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    reg_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    reg_model.fit(X_train, y_train)

    clf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    clf_model.fit(X_train, y_clf.loc[y_train.index])

    return reg_model, clf_model

reg_model, clf_model = train_models(df)

# ========================
# RECOMMENDATION SYSTEM
# ========================

@st.cache_resource
def build_recommendation(df):

    user_item = df.pivot_table(
        index="UserId",
        columns="AttractionId",
        values="Rating"
    )

    user_item.fillna(0, inplace=True)

    similarity = cosine_similarity(user_item)

    return user_item, similarity

user_item, similarity = build_recommendation(df)

def recommend(user_id, n=5):

    if user_id >= len(similarity):
        return []

    sim_scores = similarity[user_id]

    similar_users = np.argsort(sim_scores)[::-1][1:6]

    rec = user_item.iloc[similar_users].mean()

    visited = user_item.iloc[user_id]

    rec = rec[visited == 0]

    rec = rec.dropna()

    rec = rec.sort_values(ascending=False)

    return rec.head(n).index.tolist()

st.success("System Ready")

# ========================
# USER INPUT
# ========================

st.header("Prediction")

UserId = st.number_input("User ID", value=1)

VisitYear = st.number_input("Visit Year", value=2024)

VisitMonth = st.number_input("Visit Month", value=1)

Continent = st.number_input("Continent Code", value=0)

Region = st.number_input("Region Code", value=0)

Country = st.number_input("Country Code", value=0)

CityName = st.number_input("City Code", value=0)

AttractionType = st.number_input("Attraction Type", value=0)

AttractionId = st.number_input("Attraction ID", value=0)

AttractionAvgRating = st.number_input("Attraction Avg Rating", value=4.0)

UserAvgRating = st.number_input("User Avg Rating", value=4.0)

features = np.array([[
    UserId,VisitYear,VisitMonth,Continent,Region,
    Country,CityName,AttractionType,AttractionId,
    AttractionAvgRating,UserAvgRating
]])

# ========================
# PREDICT RATING
# ========================

if st.button("Predict Rating"):

    prediction = reg_model.predict(features)

    st.success(f"Predicted Rating: {prediction[0]:.2f}")

# ========================
# PREDICT VISIT MODE
# ========================

if st.button("Predict Visit Mode"):

    visit_mode = clf_model.predict(features)

    st.success(f"Predicted Visit Mode: {visit_mode[0]}")

# ========================
# RECOMMEND ATTRACTIONS
# ========================

st.header("Recommendation System")

rec_user = st.number_input("Enter User ID for Recommendations", value=1)

if st.button("Recommend Attractions"):

    recommendations = recommend(rec_user)

    if recommendations:

        st.success("Recommended Attraction IDs:")

        for r in recommendations:
            st.write(f"Attraction ID: {r}")

    else:

        st.warning("No recommendations available")

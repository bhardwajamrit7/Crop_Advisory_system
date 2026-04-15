import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Crop Advisory System", layout="wide")

# ---------------- LOAD ----------------
model = joblib.load("models/crop_recommendation_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

df = pd.read_csv("data/Crop_recommendation.csv")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌱 Crop Advisory System")
page = st.sidebar.radio("", ["🏠 Dashboard", "🌾 Prediction", "📊 Clustering"])

# ---------------- DASHBOARD ----------------
if page == "🏠 Dashboard":
    st.title("🌱 Crop Advisory Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "Random Forest")
    col2.metric("Accuracy", "99%")
    col3.metric("Data Points", len(df))

    st.markdown("---")

    st.subheader("📌 About")
    st.write("""
    This system helps farmers and researchers:
    - Predict best crop using ML
    - Analyze crop patterns using clustering
    - Visualize data insights effectively
    """)

# ---------------- PREDICTION ----------------
elif page == "🌾 Prediction":
    st.title("🌾 Crop Recommendation System")

    tab1, tab2 = st.tabs(["🧾 Input", "📊 Model Insights"])

    # -------- INPUT --------
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            N = st.number_input("Nitrogen", 0.0)
            P = st.number_input("Phosphorus", 0.0)
            K = st.number_input("Potassium", 0.0)
            temperature = st.number_input("Temperature", 0.0)

        with col2:
            humidity = st.number_input("Humidity", 0.0)
            ph = st.number_input("pH", 0.0)
            rainfall = st.number_input("Rainfall", 0.0)

        if st.button("🚀 Predict Crop"):
            with st.spinner("Predicting..."):
                input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
                input_scaled = scaler.transform(input_data)

                prediction = model.predict(input_scaled)
                crop = label_encoder.inverse_transform(prediction)

                st.success(f"🌱 Recommended Crop: **{crop[0]}**")

    # -------- MODEL INSIGHTS --------
    with tab2:
        st.subheader("📊 Feature Importance")

        features = ['N', 'P', 'K', 'Temp', 'Humidity', 'pH', 'Rainfall']
        importances = model.feature_importances_

        fig, ax = plt.subplots()
        ax.barh(features, importances)
        ax.set_title("Feature Importance")

        st.pyplot(fig)

        st.info("👉 Higher value = more influence on prediction")

# ---------------- CLUSTERING ----------------
elif page == "📊 Clustering":
    st.title("📊 Clustering Analysis")

    X = df[['N','P','K','temperature','humidity','ph','rainfall']]

    # Fit model
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    tab1, tab2, tab3 = st.tabs([
        "📌 PCA Visualization",
        "📉 Elbow Method",
        "🌾 Crop Distribution"
    ])

    # -------- PCA --------
    with tab1:
        st.subheader("PCA Cluster Visualization")

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        fig, ax = plt.subplots()

        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=df['cluster'],
            cmap='tab10'
        )

        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")

        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)

        # Cluster centers
        centers = pca.transform(kmeans.cluster_centers_)
        ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='X')

        st.pyplot(fig)

        st.info("PCA reduces 7D data into 2D for visualization")

    # -------- ELBOW --------
    with tab2:
        st.subheader("Elbow Method")

        wcss = []
        for i in range(1, 10):
            km = KMeans(n_clusters=i, random_state=42)
            km.fit(X)
            wcss.append(km.inertia_)

        fig, ax = plt.subplots()
        ax.plot(range(1, 10), wcss, marker='o')
        ax.set_xlabel("Clusters")
        ax.set_ylabel("WCSS")

        st.pyplot(fig)

    # -------- DISTRIBUTION --------
    with tab3:
        st.subheader("Crop Distribution Across Clusters")

        cluster_crop = df.groupby(['cluster', 'label']).size().unstack()

        st.bar_chart(cluster_crop)

        st.subheader("Cluster Summary Table")
        st.dataframe(cluster_crop)
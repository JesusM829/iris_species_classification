# ============================================
# Data Visualization and Dashboard Deployment
# Universidad de la Costa - Minería de Datos
# Integrantes: Jesús Martínez, Sebastián Teheran
# Grupo: 10032
# Profesor: José Escorcia Gutierrez, Ph.D.
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

st.set_page_config(page_title="Iris Species Classification", layout="wide")
st.title("Iris Species Classification")
st.markdown("**Integrantes:** Jesús Martínez, Sebastián Teheran  \n**Profesor:** José Escorcia Gutierrez, Ph.D.  \n**Universidad de la Costa**")

from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
df = iris.frame.copy()
df['species'] = df['target'].map({i: name for i, name in enumerate(iris.target_names)})
df = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'species']]
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

try:
    model = joblib.load("model_iris.joblib")
    scaler = joblib.load("scaler_iris.joblib")
except Exception as e:
    st.error("No se encontró el modelo guardado. Asegúrate de subir model_iris.joblib y scaler_iris.joblib al repositorio.")
    st.stop()

st.sidebar.header("Input measurements")
sepal_length = st.sidebar.number_input("Sepal length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
sepal_width = st.sidebar.number_input("Sepal width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
petal_length = st.sidebar.number_input("Petal length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
petal_width = st.sidebar.number_input("Petal width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

if st.sidebar.button("Predict species"):
    X_new = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    X_new_s = scaler.transform(X_new)
    pred = model.predict(X_new_s)[0]
    proba = model.predict_proba(X_new_s)[0]
    st.subheader("Predicted species:")
    st.markdown(f"**{pred}**")
    probs_df = pd.DataFrame({
        "species": model.classes_,
        "probability": proba
    }).sort_values("probability", ascending=False)
    st.table(probs_df)

    fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_length',
                        color='species', symbol='species', title='3D view: sample vs dataset')
    fig.add_scatter3d(x=[sepal_length], y=[sepal_width], z=[petal_length],
                      mode='markers', marker=dict(size=6, color='black', symbol='x'),
                      name='New sample')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.header("Model evaluation (computed on test set)")

try:
    eval_df = pd.read_csv("model_metrics.csv", index_col=0)
    st.table(eval_df.T)
except:
    st.write("No se encontraron métricas guardadas. Revisar el notebook de entrenamiento para ver resultados.")

st.markdown("---")
st.header("Exploratory Visualizations")
st.subheader("Pairplot (interactive subset)")
fig2 = px.scatter_matrix(df, dimensions=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], color='species')
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Feature distributions")
fig3 = px.histogram(df, x='petal_length', color='species', barmode='overlay')
st.plotly_chart(fig3, use_container_width=True)

st.markdown("### Uso")
st.write("1. Ingresa las medidas en la barra lateral. 2. Presiona 'Predict species'. 3. Observa la predicción, probabilidades y la ubicación del punto en la gráfica 3D.")

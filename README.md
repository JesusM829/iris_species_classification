**IRIS SPECIES CLASSIFICATION - Final Project**

**Curso:** Minería de Datos — Unidad 3: Visualización de Datos  
**Proyecto:** *Iris Species Classification*  
**Integrantes:** Martínez Martínez Jesús, Teheran Gonzales Sebastián  
**Grupo:** 10032  
**Profesor:** José Escorcia Gutierrez, Ph.D.  
**Universidad:** Universidad de la Costa  

---

**Descripción General**

Este proyecto corresponde al **proyecto final de la materia de Minería de Datos**, cuyo objetivo es **entrenar un modelo de clasificación** capaz de predecir la especie de una flor *Iris* (Setosa, Versicolor o Virginica) a partir de sus características morfológicas:  
- Longitud del sépalo  
- Ancho del sépalo  
- Longitud del pétalo  
- Ancho del pétalo  

El resultado se despliega en un **dashboard interactivo** construido con **Streamlit**, donde el usuario puede ingresar las medidas manualmente y obtener:
- La **especie predicha**.
- Las **probabilidades asociadas a cada especie**.
- Un **gráfico 3D** mostrando la posición de la flor en relación con el dataset.
- Métricas de rendimiento del modelo (**Accuracy, Precision, Recall, F1-score**).
- Visualizaciones exploratorias del dataset (pairplot interactivo, histogramas).

---

**Objetivos de Aprendizaje**

Al finalizar esta actividad, el estudiante será capaz de:
- Integrar conocimientos del curso en un flujo completo de minería de datos.
- Aplicar técnicas de preprocesamiento, modelado, validación y despliegue.
- Justificar la selección de modelos y parámetros.
- Comunicar resultados mediante un dashboard visual e interactivo.

---

**Flujo de Trabajo (Workflow)**

1. **Data Understanding:** Carga y análisis del dataset *Iris* desde `sklearn.datasets`.  
2. **Preprocesamiento:** División *train-test* estratificada y estandarización con `StandardScaler`.  
3. **Modelado:** Entrenamiento con **Random Forest** y ajuste de hiperparámetros con `GridSearchCV`.  
4. **Evaluación:** Cálculo de métricas (Accuracy, Precision, Recall, F1) y visualización de la matriz de confusión.  
5. **Interpretabilidad:** Análisis de importancia de variables.  
6. **Despliegue:** Creación del dashboard en **Streamlit Cloud** con entrada interactiva y visualización 3D.

---

**Archivos del Proyecto**

| Archivo | Descripción |
|----------|--------------|
| `project.py` | Script principal del dashboard (Streamlit App). |
| `model_iris.joblib` | Modelo Random Forest entrenado. |
| `scaler_iris.joblib` | Escalador de los datos (`StandardScaler`). |
| `model_metrics.csv` | Métricas de evaluación del modelo. |
| `requirements.txt` | Librerías necesarias para ejecutar el proyecto. |
| `README.md` | Descripción general, instalación e instrucciones. |

---

**Métricas del Modelo**

| Métrica | Valor (aproximado) |
|----------|--------------------|
| Accuracy | 0.97 |
| Precision | 0.97 |
| Recall | 0.97 |
| F1-score | 0.97 |

*(Los valores pueden variar ligeramente según la semilla aleatoria.)*

---

**Ejecución Local**

1. Clona el repositorio o descárgalo en tu equipo:
   ```bash
   git clone https://github.com/JesusM829/iris_species_classification.git
   cd iris_species_classification

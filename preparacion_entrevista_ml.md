# Repaso de Machine Learning para Entrevista: Redes Eléctricas de España

Aquí tienes un resumen de la arquitectura y los conceptos clave de Machine Learning aplicados en tu proyecto para que puedas defenderlos con seguridad en tu entrevista.

## 🛠️ 1. Frameworks y Librerías Utilizadas

*   **TensorFlow / Keras (`tensorflow`, `keras`):** Framework principal desarrollado por Google, utilizado para crear, entrenar y evaluar las redes neuronales de *Deep Learning* (modelos secuenciales: LSTM, GRU y RNN).
*   **Meta Prophet (`prophet`):** Librería específica para análisis y predicción de *Time Series* (Series Temporales) desarrollada originalmente por Facebook (Meta).
*   **Scikit-Learn (`scikit-learn`):** Utilizada fundamentalmente para dos propósitos:
    *   **Preprocesamiento:** Uso de *Scalers* para normalizar/estandarizar los datos antes de introducirlos en las redes neuronales (crítico para la convergencia en Deep Learning).
    *   **Métricas de Evaluación:** Para calcular el rendimiento de los modelos matemáticamente (MAE, RMSE y $R^2$).
*   **Streamlit y Plotly:** Utilizados para el despliegue del modelo (Front-end) y la visualización interactiva de las inferencias y datos históricos.

---

## 🧠 2. Tipos de Modelos de Predicción Aplicados

La aplicación se basa en datos históricos de la **Demanda** eléctrica (en kWh) y emplea dos enfoques predictivos principales:

### A. Modelo Estadístico: Prophet (Modelo Aditivo)

*   **Concepto:** Es un modelo aditivo donde la tendencia no lineal se ajusta a estacionalidades (diarias, semanales, anuales, etc.) y soporta la inclusión de festividades (holiday effects).
*   **Aplicación en el proyecto:** Se han entrenado distintos modelos de Prophet ajustados para diferentes frecuencias (*Diario, Semanal, Mensual, Trimestral, Semestral, Anual*).
*   **Puntos clave a mencionar en la entrevista:**
    *   Prophet es **muy robusto frente a valores atípicos (outliers)**, datos faltantes y cambios drásticos en la tendencia.
    *   **Comportamiento en ventanas largas:** Para frecuencias muy largas (como la Semestral o Anual), el modelo empieza a generar curvas más planas o lineales y obtiene un $R^2$ cercano a 1. Esto ocurre porque la librería no detecta suficiente varianza o estacionalidad representativa cuando hay tan pocos puntos agregados a ese nivel.

### B. Modelos de Deep Learning (Sequence Models)

Este enfoque "aprende" sobre secuencias de datos históricos para predecir el futuro basándose en patrones secuenciales anteriores. En el proyecto, los datos se agrupan en ventanas temporales (`window_size = 30` días) para alimentar la red.

Se han implementado tres arquitecturas de redes recurrentes:

1.  **RNN (Recurrent Neural Network):**
    *   *Concepto:* Es la arquitectura base para secuencias. La información de instantes pasados fluye hacia adelante.
    *   *Desventaja a mencionar:* Sufren del problema de **"Desvanecimiento del Gradiente"** (*Vanishing Gradient Problem*), lo que significa que en secuencias largas, la red tiende a "olvidar" la información que ocurrió en los primeros pasos de la secuencia.
2.  **LSTM (Long Short-Term Memory):**
    *   *Concepto:* Resuelve el problema del gradiente desvaneciente de las RNN normales utilizando un diseño interno más complejo basado en **puertas (gates)**: *Input Gate, Output Gate* y *Forget Gate*. Esto le permite "recordar" dependencias y variaciones a largo plazo en la serie temporal.
    *   *Dato del proyecto:* En la aplicación, este modelo alcanza casi un **82% de precisión** ($R^2$), demostrando un buen nivel de ajuste.
3.  **GRU (Gated Recurrent Unit):**
    *   *Concepto:* Es una variante más moderna de LSTM y computacionalmente más ligera (tiene menos puertas: *Update Gate* y *Reset Gate*).
    *   *Punto clave a mencionar:* En la práctica, rinden de forma muy similar a las LSTM, pero al requerir menos operaciones por celda, **las GRU se entrenan más rápido**. Es una buena práctica probar ambas y quedarse con la que converja mejor según el dataset.

***Nota técnica del código:** Las arquitecturas de Deep Learning en el proyecto están preparadas tanto para predicción de un solo paso a futuro (**One-Step**) como para predicciones encadenadas múltiples pasos hacia el futuro (**Multi-Step** a 1, 7, 14 o 24 días).*

---

## 📏 3. Evaluación Continua: Métricas

Es vital que en una entrevista demuestres cómo validas que tu modelo "es bueno". Estas son las métricas implementadas:

*   **MAE (Mean Absolute Error - Error Absoluto Medio):** Mide el promedio de todos los errores de predicción en términos absolutos. *Ejemplo:* Si el MAE es 15, significa que el modelo suele fallar por +/- 15 kWh.
*   **RMSE (Root Mean Square Error - Raíz del Error Cuadrático Medio):** Peniliza mucho más los fallos grandes porque eleva las diferencias al cuadrado antes de promediarlas. Si el modelo se equivoca gravemente en un solo punto, el RMSE se disparará más que el MAE.
*   **R² (R-cuadrado / Coeficiente de Determinación):** Indica qué porcentaje de la variabilidad en los datos originales es explicada por el modelo (se lee como un porcentaje hasta 1.0).

---

## 💡 Tips Adicionales para Respuestas en la Entrevista

*   **Pregunta trampa común:** *"¿Por qué escalaste (normalizaste/estandarizaste) los datos de la demanda antes de introducirlos en las redes neuronales de Keras, pero en Prophet no es tan estricto?"*
    *   **Respuesta ideal:** *"Las redes neuronales de Deep Learning utilizan optimizadores basados en el gradiente (como Adam o SGD) y funciones de activación (como Tanh o Sigmoide). Si les entregamos valores crudos en bruto (como decenas de miles de kWh), los gradientes pueden explotar o saturar las funciones de activación muy rápidamente, haciendo que la red no aprenda y no converja. Escalar los datos asegura que la pérdida descienda progresivamente de manera mucho más estable y rápida. Prophet, al ser un modelo estadístico aditivo basado en modelos generalizados, gestiona la magnitud original matemáticamente de otra manera."*

*   **Pregunta sobre "Multi-Step":** *"He visto que haces predicciones Multi-Step. ¿Qué problemas tiene hacer eso frente a predecir solo el día de mañana?"*
    *   **Respuesta ideal:** *"En la predicción One-Step el modelo siempre tiene la verdad real pasada. En el Multi-Step Autorregresivo, como predecimos el 'día + 2' usando la predicción (no el dato real) del 'día + 1', los errores se van acumulando de manera exponencial. Precisamente por esto es totalmente normal y esperado que predecir a 24 días tenga un margen de error bastante mayor que predecir a 1 o 7 días."*

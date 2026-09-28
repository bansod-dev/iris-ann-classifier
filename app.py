import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

@st.cache_resource
def load_model_and_objects():
    model = load_model("ann.keras")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    return model, scaler, encoder

model, scaler, encoder = load_model_and_objects()

st.title("🌸 Iris Flower Classifier")

st.write(
    "Predict the species of an Iris flower using "
    "a trained Artificial Neural Network."
)

st.divider()

st.subheader("Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1
    )

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1
    )

with col2:
    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1
    )

st.write("")

if st.button("🔍 Predict Species", use_container_width=True):

    input_data = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(
        input_scaled,
        verbose=0
    )

    predicted_class = np.argmax(
        prediction,
        axis=1
    )

    predicted_species = encoder.inverse_transform(
        predicted_class
    )[0]

    confidence = np.max(prediction) * 100

    st.divider()

    st.subheader("Prediction")

    st.success(
        f"🌸 **{predicted_species}**"
    )

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

    st.subheader("Class Probabilities")

    probabilities = prediction[0]

    for species, probability in zip(
        encoder.classes_,
        probabilities
    ):
        st.write(
            f"**{species}** — "
            f"{probability * 100:.2f}%"
        )

        st.progress(
            float(probability)
        )

st.divider()

st.caption(
    "Built with Python, TensorFlow, Scikit-learn and Streamlit"
)
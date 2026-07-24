import streamlit as st
import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image
import os

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Crop Disease Detection System",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Crop Disease Detection System")
st.write("Upload a crop leaf image to detect the disease.")
with st.sidebar:
    st.title("🌿 Crop Disease Detection")
    st.write("### AI Powered System")
    st.write("Model : MobileNetV2")
    st.write("Dataset : PlantVillage")
    st.write("Classes : 16")
    st.success("Deep Learning Project")

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_trained_model():
    model =load_model("mobilenet_final.keras")
    return model

model = load_trained_model()

# ---------------------------
# Load Class Names
# ---------------------------
dataset_path = "C:\\Users\\ASUS\\.cache\\kagglehub\\datasets\\emmarex\\plantdisease\\versions\\1\\PlantVillage"
classes = sorted(
    [
        folder
        for folder in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, folder))
    ]
)
disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "description": "A bacterial disease causing dark spots on pepper leaves.",
        "solution": "Use copper fungicide and remove infected leaves."
    },
    "Pepper_bell__healthy": {
        "description": "Healthy pepper plant.",
        "solution": "Maintain proper irrigation and nutrition."
    },
    "Potato___Early_blight": {
        "description": "Fungal disease causing brown spots.",
        "solution": "Spray fungicide and remove infected leaves."
    },
    "Potato___healthy": {
        "description": "Healthy potato plant.",
        "solution": "Continue proper crop management."
    },
    "Potato___Late_blight": {
        "description": "Serious fungal disease affecting leaves and tubers.",
        "solution": "Use recommended fungicides immediately."
    },
    "Tomato_Bacterial_spot": {
        "description": "Bacterial disease causing leaf spots.",
        "solution": "Avoid overhead watering and use copper spray."
    },
    "Tomato_Early_blight": {
        "description": "Fungal disease producing target-like spots.",
        "solution": "Remove infected leaves and spray fungicide."
    },
    "Tomato_healthy": {
        "description": "Healthy tomato plant.",
        "solution": "Maintain regular watering and monitoring."
    },
    "Tomato_Late_blight": {
        "description": "Severe fungal disease spreading quickly.",
        "solution": "Apply fungicide and destroy infected plants."
    },
    "Tomato_Leaf_Mold": {
        "description": "Leaf mold caused by fungal infection.",
        "solution": "Reduce humidity and improve air circulation."
    },
    "Tomato_Septoria_leaf_spot": {
        "description": "Leaf spot disease caused by fungus.",
        "solution": "Remove infected leaves and spray fungicide."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description": "Damage caused by spider mites.",
        "solution": "Spray neem oil or suitable miticide."
    },
    "Tomato__Target_Spot": {
        "description": "Fungal disease causing circular lesions.",
        "solution": "Apply fungicide and keep the field clean."
    },
    "Tomato__Tomato_mosaic_virus": {
        "description": "Viral disease causing mosaic leaf patterns.",
        "solution": "Remove infected plants and disinfect tools."
    },
    "Tomato_Tomato_YellowLeaf_Curl_Virus": {
        "description": "Virus causing yellow curled leaves.",
        "solution": "Control whiteflies and remove infected plants."
    }
}

# ---------------------------
# Upload Image
# ---------------------------
uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = image.resize((224, 224))

    img_array = np.array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)
if st.button("Predict Disease"):

    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    confidence = float(np.max(prediction))

    disease = classes[predicted_class]

    st.success("🌿 Disease Detected Successfully")

    st.markdown(f"## {disease.replace('_',' ')}")

    st.progress(confidence)

    st.metric(
        "Confidence",
        f"{confidence*100:.2f}%"
    )

    if disease in disease_info:
         st.info("📖 Disease Description")
    st.write(disease_info[disease].get("description", "Description not available."))

    st.warning("💊 Recommended Solution")
    st.write(disease_info[disease].get("solution", "Solution not available."))
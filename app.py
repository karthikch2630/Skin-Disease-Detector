import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

# Disable OneDNN optimizations for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the trained model
model = load_model("skin_disease_model.h5")

# Defining the classes
class_names = [
    "Cellulitis", "Impetigo", "Athlete's Foot", "Nail Fungus", 
    "Ringworm", "Cutaneous Larva Migrans", "Chickenpox", "Shingles"
]

# Disease information dictionary
disease_info = {
    "Cellulitis": {
        "description": "Cellulitis is a bacterial infection of the skin and tissues beneath the skin.",
        "medications": ["Antibiotics like penicillin, amoxicillin, or cephalexin"],
        "precautions": [
            "Keep the affected area clean and dry.",
            "Avoid scratching or picking at the skin.",
            "Seek medical attention for worsening symptoms."
        ]
    },
    "Impetigo": {
        "description": "Impetigo is a contagious bacterial skin infection, commonly affecting children.",
        "medications": ["Topical antibiotics like mupirocin, or oral antibiotics like erythromycin"],
        "precautions": [
            "Avoid close contact with others to prevent spreading.",
            "Wash hands regularly and maintain hygiene.",
            "Clean affected areas gently and cover them with a bandage."
        ]
    },
    "Athlete's Foot": {
        "description": "Athlete's Foot is a fungal infection that usually begins between the toes.",
        "medications": ["Topical antifungals like terbinafine or clotrimazole"],
        "precautions": [
            "Keep your feet clean and dry.",
            "Avoid walking barefoot in communal areas.",
            "Change socks regularly to reduce moisture buildup."
        ]
    },
    "Nail Fungus": {
        "description": "Nail fungus is a common condition that causes thickened, discolored, or brittle nails.",
        "medications": ["Topical treatments like ciclopirox or oral antifungals like itraconazole"],
        "precautions": [
            "Keep your nails trimmed and clean.",
            "Avoid sharing nail clippers or other personal items.",
            "Wear breathable shoes and avoid tight footwear."
        ]
    },
    "Ringworm": {
        "description": "Ringworm is a common fungal infection that appears as a red, itchy, circular rash.",
        "medications": ["Antifungal creams like clotrimazole, terbinafine, or oral antifungals."],
        "precautions": [
            "Keep the affected area clean and dry.",
            "Avoid sharing personal items like towels or clothing.",
            "Wear breathable fabrics to reduce moisture buildup."
        ]
    },
    "Cutaneous Larva Migrans": {
        "description": "A skin condition caused by hookworm larvae, typically acquired from contaminated soil or sand.",
        "medications": ["Antiparasitic drugs like albendazole or ivermectin"],
        "precautions": [
            "Avoid walking barefoot in sandy or moist soil.",
            "Wear protective footwear outdoors.",
            "Consult a doctor for antiparasitic treatment."
        ]
    },
    "Chickenpox": {
        "description": "Chickenpox is a viral infection causing an itchy, blister-like rash.",
        "medications": ["Antiviral drugs like acyclovir, or over-the-counter antihistamines for itching."],
        "precautions": [
            "Avoid scratching the blisters to prevent infection.",
            "Stay hydrated and rest well.",
            "Isolate to prevent spreading the infection to others."
        ]
    },
    "Shingles": {
        "description": "Shingles is a viral infection causing a painful rash, often in a single stripe on one side of the body.",
        "medications": ["Antiviral drugs like valacyclovir or acyclovir"],
        "precautions": [
            "Avoid close contact with others, especially those who haven't had chickenpox.",
            "Keep the rash clean and covered.",
            "Manage pain with prescribed medications or cold compresses."
        ]
    }
}

# Function to preprocess the image
def preprocess_image(img, target_size):
    img = img.resize(target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize to [0, 1] range
    return img

# Streamlit app starts here
def main():
    st.title("Skin Disease Detector")
    st.subheader("Upload an image of the skin condition, and the model will predict the disease.")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

    # Display uploaded image and predict button
    if uploaded_file is not None:
        # Open and display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Process and predict when the "Detect" button is clicked
        if st.button("Detect"):
            with st.spinner("Analyzing the image..."):
                # Preprocess the image
                processed_image = preprocess_image(image, target_size=(150, 150))

                # Predict the class
                prediction = model.predict(processed_image)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction)

                # Display the prediction
                st.success(f"Prediction: {predicted_class}")
                st.write(f"Confidence: {confidence * 100:.2f}%")

                # Retrieve additional information
                disease_details = disease_info.get(predicted_class, None)
                if disease_details:
                    st.subheader("Disease Information")
                    st.write(disease_details["description"])

                    st.subheader("Recommended Medications")
                    st.write(", ".join(disease_details["medications"]))

                    st.subheader("Precautions")
                    for precaution in disease_details["precautions"]:
                        st.write(f"- {precaution}")

                    # Include a warning
                    st.warning("This information is for educational purposes only. Please consult a dermatologist for a proper diagnosis and treatment.")

if __name__ == "__main__":
    main()

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

# Define class names
class_names = [
    "Cellulitis", "Impetigo", "Athlete's Foot", "Nail Fungus",
    "Ringworm", "Cutaneous Larva Migrans", "Chickenpox", "Shingles"
]

# Disease info
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

# Preprocess image
def preprocess_image(img, target_size):
    img = img.resize(target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Main app
def main():
    st.set_page_config(page_title="Skin Disease Detector", layout="centered")

    # Custom CSS
    st.markdown("""
        <style>
            .main { background-color: #f5f5f5; padding: 20px; }
            .title { font-size: 40px; color: #2c3e50; font-weight: bold; text-align: center; }
            .subtitle { font-size: 20px; text-align: center; color: #555; margin-top: -10px; }
            .footer { margin-top: 50px; text-align: center; font-size: 12px; color: #999; }
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown('<div class="title">🧬 Skin Disease Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload an image to predict and learn about skin conditions</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Expandable disease preview
    with st.expander("📷 See examples of detectable conditions"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.image("skin-disease-datasaet/train_set/BA- cellulitis/BA- cellulitis (2).jpeg", caption="Cellulitis", use_container_width=True)
            st.image("skin-disease-datasaet/train_set/BA-impetigo/4_BA-impetigo (34).jpg", caption="Impetigo", use_container_width=True)

        with col2:
            st.image("skin-disease-datasaet/train_set/FU-athlete-foot/FU-athlete-foot (2).jpg", caption="Athlete's Foot", use_container_width=True)
            st.image("skin-disease-datasaet/train_set/FU-nail-fungus/_4_1433.jpg", caption="Nail Fungus", use_container_width=True)

        with col3:
            st.image("skin-disease-datasaet/train_set/FU-ringworm/6_FU-ringworm (9).jpg", caption="Ringworm", use_container_width=True)
            st.image("skin-disease-datasaet/train_set/PA-cutaneous-larva-migrans/15_PA-cutaneous-larva-migrans (42).jpg", caption="Larva Migrans", use_container_width=True)
        with col4:
            st.image("skin-disease-datasaet/train_set/VI-chickenpox/7_VI-chickenpox (9).jpg", caption="Chickenpox", use_container_width=True)
            st.image("skin-disease-datasaet/train_set/VI-shingles/9_VI-shingles (20).jpg", caption="Shingles", use_container_width=True)
        st.markdown("---")

    # Upload and detect
    uploaded_file = st.file_uploader("📤 Upload a clear image of the skin condition (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

        if st.button("🔍 Detect"):
            with st.spinner("Analyzing the image..."):
                processed_image = preprocess_image(image, target_size=(150, 150))
                prediction = model.predict(processed_image)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction)

                st.success(f"✅ Prediction: **{predicted_class}**")
                st.write(f"📊 Confidence: **{confidence * 100:.2f}%**")

                disease_details = disease_info.get(predicted_class)
                if disease_details:
                    st.markdown("## 🦠 Detected Disease Information")

                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"""
                            <div style="background-color: #fcefee; padding: 15px; border-radius: 10px;">
                                <h3 style="color:#d6336c;">🩺 Disease Name: <strong>{predicted_class}</strong></h3>
                                <p style = "color: black;"><strong>Description:</strong> {disease_details['description']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.metric(label="📊 Confidence", value=f"{confidence * 100:.2f} %")

                    st.markdown("---")
                    with st.expander("💊 Recommended Medications"):
                        for med in disease_details["medications"]:
                            st.markdown(f"- {med}")

                    with st.expander("⚠️ Precautions to Follow"):
                        for precaution in disease_details["precautions"]:
                            st.markdown(f"- {precaution}")

                    st.info("📌 **Note:** This prediction is for educational purposes only. Please consult a certified dermatologist.")
    else:
        st.warning("👆 Please upload an image to get started.")

# Run
if __name__ == "__main__":
    main()

import streamlit as st
import numpy as np
from PIL import Image
import os
import requests
from streamlit_lottie import st_lottie
from deploy_mnist import MNISTDeployer

# --- 1. DYNAMIC PATH DISCOVERY ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'final_98_model')
IMAGE_DIR = os.path.join(BASE_DIR, '..', 'images')

# --- 2. SETTINGS & STYLING ---
st.set_page_config(page_title="Magic MNIST", page_icon="🪄", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background-color: #009688;
        color: white;
        font-weight: bold;
        border: none;
    }
    .prediction-card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. ASSETS & UTILITIES ---
def load_lottieurl(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_robot = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")

def safe_image_display(filename, caption):
    """Safely loads images using use_container_width."""
    img_path = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(img_path):
        st.image(img_path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Note: {filename} was not found.")

# --- 4. SIDEBAR ---
with st.sidebar:
    if lottie_robot:
        st_lottie(lottie_robot, height=180)
    st.title("Project Info")
    st.success("*This application is a full-stack Machine Learning demonstration. It bridges the gap between raw mathematical theory and interactive software, featuring a custom neural engine built entirely from scratch, no deep learning frameworks like TensorFlow or PyTorch were used.*")
    st.info("**Model Accuracy:** 98.38%")
    st.write("**Architecture:** 784 ⮕ 512 ⮕ 10")
    st.write("**Optimizer:** Custom ADAM")
    st.write("**Framework:** 100% NumPy")

# --- 5. HEADER & TECH SPECS ---
st.title(" Magic Digit Classifier")
st.write("A framework-free deep learning implementation built from scratch.")

with st.expander(" View Technical Specifications"):
    st.markdown("""
    #### **The Engineering Behind the Magic**
    * **Hidden Layer**: 512 neurons with **ReLU** activation.
    * **Output Layer**: 10 neurons with **Softmax** probability distribution.
    * **Optimization**: Custom **ADAM** implementation for dynamic weight updates.
    * **He Initialization**: Used to ensure gradient stability during backpropagation.
    """)

# --- 6. TABS INTERFACE ---
tab1, tab2 = st.tabs([" Live Classifier", " Performance Audit"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload Your Digit")
        uploaded_file = st.file_uploader("Upload a handwritten digit image", type=["png", "jpg", "jpeg"])
        
        if uploaded_file:
            img = Image.open(uploaded_file)
            # Using the updated parameter here
            st.image(img, caption="Original Input", use_container_width=True)

    with col2:
        st.subheader("Brain's Analysis")
        if uploaded_file is not None:
            if st.button("Run Inference"):
                with st.spinner('Neural Network is thinking...'):
                    temp_path = os.path.join(BASE_DIR, "temp_inference.png")
                    img.save(temp_path)
                    
                    try:
                        model = MNISTDeployer(model_dir=MODEL_PATH)
                        digit, conf = model.predict_from_file(temp_path)
                        
                        st.markdown(f"""
                        <div class="prediction-card">
                            <p style='color: #757575; font-size: 18px;'>Predicted Digit</p>
                            <h1 style='font-size: 100px; color: #009688; margin: 0;'>{digit}</h1>
                            <p style='font-size: 20px;'>Confidence: <b>{conf*100:.2f}%</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(conf)
                    except Exception as e:
                        st.error(f"Inference Error: {e}")
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
        else:
            st.info("Ready to classify! Please upload an image.")

with tab2:
    st.subheader("Scientific Performance Audit")
    st.write("Visual verification of the model's 98.38% peak accuracy.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        safe_image_display("learning_curves.png", "Training Dynamics: Loss vs. Accuracy")
        safe_image_display("final_matrix_perfect.png", "Confusion Matrix: Precision Audit")
    with col_b:
        safe_image_display("tsne_manifold.png", "t-SNE: Feature Cluster Separation")
        safe_image_display("weight_histograms.png", "Weight Health: He Initialization Stability")

st.markdown("---")
st.caption("© 2026 | MNIST Scratch Implementation | IIT Kanpur ")
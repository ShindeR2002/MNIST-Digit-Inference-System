import numpy as np
import os
from PIL import Image

class MNISTDeployer:
    """
    Production-ready inference agent for MNIST digit classification.
    Uses pre-trained weights to perform a forward pass in NumPy.
    """
    def __init__(self, model_dir=None):
        # Dynamically find the path to this script to ensure relative paths work
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        # Default to the 'final_98_model' folder if no directory is provided
        if model_dir is None:
            model_dir = os.path.join(base_path, '..', 'final_98_model')
        
        print(f"Loading weights from: {os.path.abspath(model_dir)}")
        
        try:
            # Load the serialized weights and biases (the model's "brain")
            self.W1 = np.load(os.path.join(model_dir, 'W1.npy'))
            self.b1 = np.load(os.path.join(model_dir, 'b1.npy'))
            self.W2 = np.load(os.path.join(model_dir, 'W2.npy'))
            self.b2 = np.load(os.path.join(model_dir, 'b2.npy'))
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing .npy weight files in {model_dir}. Please train the model first.")

    def predict_from_file(self, image_path):
        # Load and convert to grayscale
        img = Image.open(image_path).convert('L').resize((28, 28))
        pixels = np.array(img)

    # --- NEW: SMART INVERSION LOGIC ---
    # If the corners are bright, it's a white background; we must invert it.
    # MNIST MUST be white text on black background.
        if np.mean([pixels[0,0], pixels[0,-1], pixels[-1,0], pixels[-1,-1]]) > 127:
           pixels = 255 - pixels 
    
    # Final normalization to [0, 1]
        pixels = pixels.flatten() / 255.0
    
    # Forward Pass
        z1 = np.dot(self.W1, pixels.reshape(784, 1)) + self.b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(self.W2, a1) + self.b2
    
        probs = np.exp(z2 - np.max(z2)) / np.sum(np.exp(z2 - np.max(z2)))
        return int(np.argmax(probs)), float(np.max(probs))
# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Setup absolute paths for the deployment environment
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, '..', 'final_98_model')
    SAMPLE_PATH = os.path.join(BASE_DIR, 'sample_images', 'test_digit_7.png')

    print("="*40)
    print("MNIST INFERENCE AGENT STARTING")
    print("="*40)

    try:
        # 2. Initialize the agent
        agent = MNISTDeployer(model_dir=MODEL_PATH)
        
        # 3. Perform a test prediction if the sample exists
        if os.path.exists(SAMPLE_PATH):
            digit, conf = agent.predict_from_file(SAMPLE_PATH)
            
            print("\n" + "✅" + " SUCCESSFUL CLASSIFICATION")
            print(f"Target Image: {os.path.basename(SAMPLE_PATH)}")
            print(f"Predicted Digit: {digit}")
            print(f"Model Confidence: {conf*100:.2f}%")
        else:
            print(f"\n❌ FILE ERROR: Sample image not found at: {SAMPLE_PATH}")
            print("Ensure you generated sample images in your training notebook.")
            
    except Exception as e:
        print(f"\n❌ DEPLOYMENT CRASH: {e}")

    print("\n" + "="*40)
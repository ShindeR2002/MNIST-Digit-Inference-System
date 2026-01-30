import numpy as np
import os
from PIL import Image

class MNISTDeployer:
    def __init__(self, model_dir=None):
        # Determine paths relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if model_dir is None:
            model_dir = os.path.join(base_dir, '..', 'final_98_model')
        
        print(f"Loading weights from: {model_dir}")
        
        try:
            self.W1 = np.load(os.path.join(model_dir, 'W1.npy'))
            self.b1 = np.load(os.path.join(model_dir, 'b1.npy'))
            self.W2 = np.load(os.path.join(model_dir, 'W2.npy'))
            self.b2 = np.load(os.path.join(model_dir, 'b2.npy'))
            print("✅ Model weights loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            raise

def predict_from_file(self, image_path):
    # 1. Load and convert to grayscale
    img = Image.open(image_path).convert('L')
    
    # 2. Thresholding: Remove shadows and gray mid-tones
    # This turns pixels into either 0 or 255 (binary)
    img = img.point(lambda p: 255 if p > 140 else 0) 
    
    # 3. SMART INVERSION: Ensure it is white-on-black
    pixels = np.array(img)
    if np.mean(pixels) > 127:
        pixels = 255 - pixels
        
    # 4. Centering: Crop the bounding box of the digit and pad it
    coords = np.argwhere(pixels > 0)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        # Crop to the digit only
        digit_crop = pixels[y0:y1+1, x0:x1+1]
        img_crop = Image.fromarray(digit_crop)
        # Resize to 20x20 to leave a margin
        img_crop = img_crop.resize((20, 20), Image.LANCZOS)
        # Paste onto a new 28x28 black canvas
        new_img = Image.new('L', (28, 28), 0)
        new_img.paste(img_crop, (4, 4))
        pixels = np.array(new_img)
    else:
        # Fallback if image is empty
        img = img.resize((28, 28))
        pixels = np.array(img)

    # 5. Final Normalization and Prediction
    pixels = pixels.flatten() / 255.0
    z1 = np.dot(self.W1, pixels.reshape(784, 1)) + self.b1
    a1 = np.maximum(0, z1) 
    z2 = np.dot(self.W2, a1) + self.b2
    exp_z2 = np.exp(z2 - np.max(z2))
    probs = exp_z2 / np.sum(exp_z2)
    
    return int(np.argmax(probs)), float(np.max(probs))
if __name__ == "__main__":
    print("========================================")
    print("MNIST INFERENCE AGENT STARTING")
    print("========================================")
    try:
        deployer = MNISTDeployer()
        print("✅ Deployment Agent Ready for app.py")
    except Exception as e:
        print(f"❌ DEPLOYMENT CRASH: {e}")

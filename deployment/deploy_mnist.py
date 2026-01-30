import numpy as np
import os
from PIL import Image

class MNISTDeployer:
    def __init__(self, model_dir=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if model_dir is None:
            # Assumes model is in 'final_98_model' one level up
            model_dir = os.path.join(base_dir, '..', 'final_98_model')
        
        try:
            self.W1 = np.load(os.path.join(model_dir, 'W1.npy'))
            self.b1 = np.load(os.path.join(model_dir, 'b1.npy'))
            self.W2 = np.load(os.path.join(model_dir, 'W2.npy'))
            self.b2 = np.load(os.path.join(model_dir, 'b2.npy'))
        except Exception as e:
            raise FileNotFoundError(f"Weights missing in {model_dir}. Error: {e}")

    def predict_from_file(self, image_path):
        """Advanced preprocessing to handle real-world handwritten photos."""
        # 1. Grayscale and Initial Resize
        img = Image.open(image_path).convert('L')
        
        # 2. Thresholding: Remove shadows/noise from photos
        img = img.point(lambda p: 255 if p > 130 else 0) 
        
        # 3. Smart Inversion: Convert light-background paper to MNIST dark-background
        pixels = np.array(img)
        if np.mean(pixels) > 127:
            pixels = 255 - pixels
            
        # 4. Bounding Box Normalization & Centering
        coords = np.argwhere(pixels > 0)
        if coords.size > 0:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            digit_crop = pixels[y0:y1+1, x0:x1+1]
            img_crop = Image.fromarray(digit_crop)
            # Resize to 20x20 to maintain MNIST-style margins
            img_crop = img_crop.resize((20, 20), Image.LANCZOS)
            
            canvas = Image.new('L', (28, 28), 0)
            canvas.paste(img_crop, (4, 4)) 
            pixels = np.array(canvas)
        else:
            pixels = np.array(img.resize((28, 28)))

        # 5. Forward Pass through 512-neuron Hidden Layer
        pixels = pixels.flatten() / 255.0
        z1 = np.dot(self.W1, pixels.reshape(784, 1)) + self.b1
        a1 = np.maximum(0, z1) # ReLU
        z2 = np.dot(self.W2, a1) + self.b2
        
        # Softmax for probabilities
        exp_z2 = np.exp(z2 - np.max(z2))
        probs = exp_z2 / np.sum(exp_z2)
        
        return int(np.argmax(probs)), float(np.max(probs))

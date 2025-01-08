import cv2
import os
import numpy as np

def preprocess_image(image_path, target_size=(300, 300)):
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0 
    return image

def load_data(data_dir):
    
    images = []
    labels = []
    
    
    print(f"Files in {data_dir}: {os.listdir(data_dir)}")

    for root, dirs, files in os.walk(data_dir):
        print(f"Checking directory: {root}")  
        for file in files:
            print(f"Found file: {file}")  
            if file.endswith((".png", ".jpg", ".jpeg")):  
                image_path = os.path.join(root, file)
                print(f"Processing file: {image_path}")  
                images.append(preprocess_image(image_path))
                labels.append(0)  
    return np.array(images), np.array(labels)

if __name__ == "__main__":
    data_dir = r"C:\Users\erkoc\OneDrive\Desktop\FaceMaskDetection\data\images"
    images, labels = load_data(data_dir)
    print(f"Loaded {len(images)} images.")

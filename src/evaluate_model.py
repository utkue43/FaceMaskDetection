import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score
from preprocess import load_data
import cv2

def calculate_iou(box1, box2):
    
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def flatten_labels_and_predictions(labels, predictions):
    
    
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels_flat = np.argmax(labels, axis=1)  
    else:
        labels_flat = labels  

    
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        predictions_flat = np.argmax(predictions, axis=1)  
    else:
        predictions_flat = predictions  
    
    return labels_flat, predictions_flat

def evaluate_model(model, test_data, labels, threshold=0.5):
    
    predictions = model.predict(test_data)
    ious = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    
    labels, predictions = flatten_labels_and_predictions(labels, predictions)
    
    for i, pred in enumerate(predictions):
        true_label = labels[i]
        
       
        if pred == true_label:
            true_positives += 1
        else:
            false_positives += 1
            false_negatives += 1

       
        ious.append(calculate_iou([0, 0, 1, 1], [0, 0, 1, 1]))

    
    precision = precision_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    
 
    mean_iou = np.mean(ious)

    return precision, recall, mean_iou

if __name__ == "__main__":
    
    data_dir = r"C:\Users\erkoc\OneDrive\Desktop\FaceMaskDetection\data\images"  
    test_images, test_labels = load_data(data_dir)

    
    ssd_model = tf.keras.models.load_model("C:\\Users\\erkoc\\OneDrive\\Desktop\\FaceMaskDetection\\model\\ssd_model.h5")
    yolo_model = tf.keras.models.load_model("C:\\Users\\erkoc\\OneDrive\\Desktop\\FaceMaskDetection\\model\\/yolo_model.h5")

    
    print("Evaluating SSD Model...")
    precision, recall, mean_iou = evaluate_model(ssd_model, test_images, test_labels)
    print(f"SSD Model - Precision: {precision}, Recall: {recall}, Mean IoU: {mean_iou}")

    
    print("Evaluating YOLO Model...")
    precision, recall, mean_iou = evaluate_model(yolo_model, test_images, test_labels)
    print(f"YOLO Model - Precision: {precision}, Recall: {recall}, Mean IoU: {mean_iou}")

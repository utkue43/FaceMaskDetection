import cv2
import numpy as np
import tensorflow as tf


ssd_model = tf.keras.models.load_model("C:\\Users\\erkoc\\OneDrive\\Desktop\\FaceMaskDetection\\model\\ssd_model.h5")
yolo_model = tf.keras.models.load_model("C:\\Users\\erkoc\\OneDrive\\Desktop\\FaceMaskDetection\\model\\yolo_model.h5")

def preprocess_frame(frame):
    """Preprocess a frame for the object detection model."""
    frame_resized = cv2.resize(frame, (300, 300))  
    frame_normalized = frame_resized / 255.0  
    return np.expand_dims(frame_normalized, axis=0)  


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    input_frame = preprocess_frame(frame)

    
    ssd_predictions = ssd_model.predict(input_frame)
    yolo_predictions = yolo_model.predict(input_frame)

    
    print("SSD predictions shape:", ssd_predictions.shape)
    print("SSD predictions:", ssd_predictions)
    print("YOLO predictions shape:", yolo_predictions.shape)
    print("YOLO predictions:", yolo_predictions)

    
    if isinstance(ssd_predictions, np.ndarray):
       
        prediction = ssd_predictions[0]  
        class_id = np.argmax(prediction)  
        confidence = prediction[class_id]  
        
        print(f"SSD - Predicted class: {class_id}, Confidence: {confidence}")  

        
        if confidence > 0.3:  
            label = 'Mask' if class_id == 1 else 'No Mask'
            cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    
    if isinstance(yolo_predictions, np.ndarray):
        
        prediction = yolo_predictions[0]  
        class_id = np.argmax(prediction)  
        confidence = prediction[class_id]  
        
        print(f"YOLO - Predicted class: {class_id}, Confidence: {confidence}")  

        
        if confidence > 0.3:  
            label = 'Mask' if class_id == 1 else 'No Mask'
            cv2.putText(frame, label, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

  
    cv2.imshow('Real-Time Face Mask Detection', frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

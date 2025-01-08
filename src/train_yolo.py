import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from preprocess import load_data

def create_yolo_model(input_shape=(300, 300, 3), num_classes=3):
    """Create a YOLO model."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights='imagenet'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Flatten(),  
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

if __name__ == "__main__":
   
    data_dir = r"C:\Users\erkoc\OneDrive\Desktop\FaceMaskDetection\data\images"
    images, labels = load_data(data_dir)

    
    yolo_model = create_yolo_model()
    yolo_model.compile(optimizer='adam', 
                       loss='sparse_categorical_crossentropy', 
                       metrics=['accuracy'])

    
    yolo_model.fit(
        images, labels, 
        epochs=80, batch_size=32, 
        validation_split=0.2
    )

    
    yolo_model.save("C:\\Users\\erkoc\\OneDrive\\Desktop\\FaceMaskDetection\\model\\yolo_model.h5")
    print("YOLO model training completed and saved.")
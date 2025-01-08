import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from preprocess import load_data

def create_ssd_model(input_shape=(300, 300, 3), num_classes=3):
    """Create an SSD model."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights='imagenet'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

if __name__ == "__main__":
  

    data_dir = r"C:\\Users\\erkoc\\OneDrive\\Desktop\\FaceMaskDetection\\data"     

    images, labels = load_data(data_dir)

    # Create the model
    ssd_model = create_ssd_model()
    ssd_model.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])

    # Train the model
    ssd_model.fit(
        images, labels, 
        epochs=80, batch_size=32, 
        validation_split=0.2
    )

    # Save the model
    ssd_model.save("C:\\Users\\erkoc\\OneDrive\\Desktop\\FaceMaskDetection\\model\\ssd_model.h5")
    print("SSD model training completed and saved.")
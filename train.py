
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_builder import build_model

def train_model():
    train_dir = 'data'
    model = build_model()

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    model.fit(train_data, validation_data=val_data, epochs=5)
    model.save('models/rice_classifier.h5')
    return model
if __name__ == "__main__":
    train_model()


import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model(model=None):
    if model is None:
        model = load_model('models/rice_classifier.h5')

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    val_data = datagen.flow_from_directory(
        'data',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    preds = model.predict(val_data)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_data.classes
    labels = list(val_data.class_indices.keys())

    print(classification_report(y_true, y_pred, target_names=labels))

if __name__ == "__main__":
    evaluate_model()



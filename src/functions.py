import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image

def load_and_preprocess_data():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    return X_train, y_train, X_test, y_test

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, epochs=5, batch_size=64, validation_split=0.2):
    return model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

def evaluate_model(model, X_test, y_test):
    return model.evaluate(X_test, y_test)

def plot_training_history(history):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Entrenamiento', color='#ff7f0e')  # Naranja
    plt.plot(history.history['val_accuracy'], label='Validación', color='#1f77b4')  # Azul
    plt.title('Evolución de la Accuracy')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(visible=True, color='lightgray', linestyle='--', linewidth=0.5)
    plt.savefig('img/accuracy.jpg')
    # plt.show()

    plt.figure()
    plt.plot(history.history['loss'], label='Entrenamiento', color='#ff7f0e')  # Naranja
    plt.plot(history.history['val_loss'], label='Validación', color='#1f77b4')  # Azul
    plt.title('Evolución de la Pérdida (Loss)')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(visible=True, color='lightgray', linestyle='--', linewidth=0.5)
    plt.savefig('img/loss.jpg')
    # plt.show()

def plot_confusion_matrix(model, X_test, y_test):
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap='Blues')
    plt.title("Matriz de Confusión - MNIST")
    plt.savefig('img/confusion_matrix.jpg')
    # plt.show()
    return y_pred

def visualize_misclassified_samples(X_test, y_test, y_pred):
    misclassified_idx = np.where(y_pred != y_test)[0]
    print(f"Encontrados {len(misclassified_idx)} ejemplos mal clasificados.")
    np.random.shuffle(misclassified_idx)
    misclassified_samples = misclassified_idx[:5]

    plt.figure(figsize=(10,2))
    for i, idx in enumerate(misclassified_samples):
        img = X_test[idx]
        true_label = y_test[idx]
        pred_label = y_pred[idx]
        plt.subplot(1, 5, i+1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"Pred: {pred_label}, Real: {true_label}")
        plt.axis('off')
    plt.savefig('img/misclassified_samples.jpg')
    # plt.show()


def predict_and_save_digit_in_image(model, image_path, output_path="img/processed_turin_shroud.jpg"):
    """
    Carga una imagen externa, la procesa (umbralizado y resize),
    la guarda en disco, y devuelve el dígito que el modelo MNIST "ve".
    """
    # 1. Cargar en escala de grises
    img = Image.open(image_path).convert('L')
    
    # 2. Umbralizar (Threshold):
    # Si el pixel > 128 => píxel blanco (255), de lo contrario => píxel negro (0).
    threshold_value = 128
    img = img.point(lambda x: 255 if x > threshold_value else 0)

    # 3. Redimensionar a 28x28
    img = img.resize((28, 28))

    # 4. Guardar la imagen procesada (para ver cómo queda)
    img.save(output_path)
    print(f"Imagen procesada guardada en: {output_path}")

    # 5. Convertir a NumPy y normalizar [0,1]
    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=(0, -1))  # (1,28,28,1)
    
    # 6. Predecir dígito
    predictions = model.predict(img_arr)
    predicted_digit = np.argmax(predictions, axis=1)[0]
    
    return predicted_digit

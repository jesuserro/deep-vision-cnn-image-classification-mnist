import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Cargar el dataset MNIST
# ------------------------------------------
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Observa tamaños
print("X_train shape:", X_train.shape)  # (60000, 28, 28)
print("y_train shape:", y_train.shape)  # (60000, )
print("X_test shape:", X_test.shape)    # (10000, 28, 28)

# 2. Preprocesado: Normalización y cambio de dimensiones
# ------------------------------------------
# Normalizamos para que los valores estén entre 0 y 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Redimensionamos para que tengan un canal (28,28,1)
# (para redes CNN, necesitamos la dimensión "canal" al final)
X_train = np.expand_dims(X_train, axis=-1)  # (60000, 28, 28, 1)
X_test = np.expand_dims(X_test, axis=-1)    # (10000, 28, 28, 1)

# 3. Construcción de la CNN
# ------------------------------------------
model = tf.keras.Sequential([
    # Capa Convolucional 1
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    # Capa Convolucional 2
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    
    # Aplanado y MLP final
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # 10 clases (dígitos 0-9)
])

# 4. Compilar el modelo
# ------------------------------------------
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Resumen de la arquitectura
model.summary()

# 5. Entrenar el modelo
# ------------------------------------------
history = model.fit(X_train, y_train,
                    epochs=5,  # para ejemplo basta con 5, puedes aumentar
                    batch_size=64,
                    validation_split=0.2)  # 20% de train como validación

# 6. Evaluar en test
# ------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Pérdida en test:", test_loss)
print("Exactitud en test:", test_acc)

# 7. Graficar la historia de entrenamiento
# ------------------------------------------
plt.figure()
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Evolución de la Accuracy')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('img/accuracy.jpg')
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Evolución de la Pérdida (Loss)')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 8. Matriz de confusión
# ------------------------------------------
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap='Blues')
plt.title("Matriz de Confusión - MNIST")
# Save the confusion matrix plot as an image
plt.savefig('img/confusion_matrix.jpg')
plt.show()

# 9. Visualizar algunos ejemplos mal clasificados
# ------------------------------------------
misclassified_idx = np.where(y_pred != y_test)[0]
print(f"Encontrados {len(misclassified_idx)} ejemplos mal clasificados.")

# Muestra aleatoriamente 5 ejemplos mal clasificados
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
# Save the misclassified samples plot as an image
plt.savefig('img/misclassified_samples.jpg')
plt.show()

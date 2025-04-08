from functions import (
    load_and_preprocess_data,
    build_model,
    train_model,
    evaluate_model,
    plot_training_history,
    plot_confusion_matrix,
    visualize_misclassified_samples
)

# 1. Cargar y preprocesar datos
X_train, y_train, X_test, y_test = load_and_preprocess_data()

# 2. Construir el modelo
model = build_model()
model.summary()

# 3. Entrenar el modelo
history = train_model(model, X_train, y_train)

# 4. Evaluar el modelo
test_loss, test_acc = evaluate_model(model, X_test, y_test)
print("Pérdida en test:", test_loss)
print("Exactitud en test:", test_acc)

# 5. Graficar métricas de entrenamiento
plot_training_history(history)

# 6. Matriz de confusión
y_pred = plot_confusion_matrix(model, X_test, y_test)

# 7. Visualizar ejemplos mal clasificados
visualize_misclassified_samples(X_test, y_test, y_pred)
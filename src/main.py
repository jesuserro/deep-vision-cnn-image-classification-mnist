from functions import (
    load_and_preprocess_data,
    build_model,
    train_model,
    evaluate_model,
    plot_training_history,
    plot_confusion_matrix,
    visualize_misclassified_samples,
    predict_and_save_digit_in_image
)

# 1. Load and preprocess data
X_train, y_train, X_test, y_test = load_and_preprocess_data()

# 2. Build the model
model = build_model()
model.summary()

# 3. Train the model
history = train_model(model, X_train, y_train)

# 4. Evaluate the model
test_loss, test_acc = evaluate_model(model, X_test, y_test)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

# 5. Predict a digit in an image of the Shroud of Turin
predicted_digit = predict_and_save_digit_in_image(
    model, 
    image_path="img/turin-shroud_number.jpg",
    output_path="img/turin-shroud_number_processed.jpg"  # Nombre que prefieras
)

print(f"El modelo cree que el dígito es: {predicted_digit}")

# 6. Plot training metrics
plot_training_history(history)

# 7. Confusion matrix
y_pred = plot_confusion_matrix(model, X_test, y_test)

# 8. Visualize misclassified examples
visualize_misclassified_samples(X_test, y_test, y_pred)
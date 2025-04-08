from functions import (
    load_and_preprocess_data,
    build_model,
    train_model,
    evaluate_model,
    plot_training_history,
    plot_confusion_matrix,
    visualize_misclassified_samples,
    predict_digit_in_image
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
predicted_digit = predict_digit_in_image(model, "img/turin-shroud_number.jpg")

if predicted_digit == 3:
    print("The model detects a '3' in the image of the Shroud of Turin.")
else:
    print(f"The model believes the number in the image of the Shroud is a '{predicted_digit}'.")

# 6. Plot training metrics
plot_training_history(history)

# 7. Confusion matrix
y_pred = plot_confusion_matrix(model, X_test, y_test)

# 8. Visualize misclassified examples
visualize_misclassified_samples(X_test, y_test, y_pred)
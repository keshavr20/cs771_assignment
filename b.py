import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import time
def my_map(challenges):
    D = 32 * (32 + 1) // 2  # Dimensionality of the output vector
    phi = np.zeros((challenges.shape[0], D))

    for j in range(challenges.shape[0]):
        challenge_vector = challenges[j]
        index = 0
        for i in range(32):
            z_ij = np.cumprod(1 - 2 * challenge_vector[i:])
            phi[j, index:index+32-i] = z_ij
            index += 32-i

    return phi

# Load the data
def load_data(file_path):
    """
    Load data from the given file path.
    Assumes each line consists of input features followed by the response.
    """
    data = np.loadtxt(file_path)
    inputs = data[:, :-1]  # Input features are all columns except the last one
    responses = data[:, -1]   # Response is the last column
    return inputs, responses

def calculate_accuracy(predictions, true_labels):
    # Convert predictions to binary values (0 or 1) based on a threshold
    binary_predictions = (predictions > 0).astype(int)
    
    # Calculate accuracy
    accuracy = np.mean(binary_predictions == true_labels)
    
    return accuracy

def my_fit(challenges, responses, loss='hinge'):
    # Apply the mapping function to each challenge vector
    X_mapped = my_map(challenges)

    # Fit a linear SVC model with the specified loss function
    if loss == 'hinge':
        model = LinearSVC(loss='hinge')
    elif loss == 'squared_hinge':
        model = LinearSVC(loss='squared_hinge')
    else:
        raise ValueError("Invalid loss function. Use 'hinge' or 'squared_hinge'.")

    
    model.fit(X_mapped, responses)
    
    

    weights = model.coef_[0]  # Coefficients of the features
    bias = model.intercept_[0]  # Intercept term

    return weights, bias, model

# Example usage:
file_path = "train.dat"
inputs, responses = load_data(file_path)

# Training with hinge loss
start_time = time.time()
w_hinge, b_hinge, model_hinge = my_fit(inputs, responses, loss='hinge')
end_time = time.time()
training_time_hinge = end_time - start_time
# Training with squared hinge loss
start_time = time.time()
w_squared_hinge, b_squared_hinge, model_squared_hinge = my_fit(inputs, responses, loss='squared_hinge')
end_time = time.time()
training_time_sq_hinge = end_time - start_time
# Now, you can use the trained models to make predictions on the test set and evaluate accuracy
test_challenges, test_responses = load_data("test.dat")

# Predictions with hinge loss
mapped_test_features = my_map(test_challenges)
predictions_hinge = mapped_test_features.dot(w_hinge) + b_hinge
accuracy_hinge = calculate_accuracy(predictions_hinge, test_responses)

# Predictions with squared hinge loss
predictions_squared_hinge = mapped_test_features.dot(w_squared_hinge) + b_squared_hinge
accuracy_squared_hinge = calculate_accuracy(predictions_squared_hinge, test_responses)

print("Accuracy on Test Set (Hinge Loss):", accuracy_hinge)
print("Accuracy on Test Set (Squared Hinge Loss):", accuracy_squared_hinge)
print("Training Time (Hinge Loss):", training_time_hinge)
print("Training Time (Squared Hinge Loss):", training_time_sq_hinge)

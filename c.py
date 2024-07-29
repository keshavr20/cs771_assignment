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

def my_fit(challenges, responses):
    # Apply the mapping function to each challenge vector
    X_mapped = my_map(challenges)

    # Fit a logistic regression model
    #model = LogisticRegression(penalty='l2', solver='lbfgs',C=100)
    model = LogisticRegression(penalty='l1', solver='liblinear')
    #model = LinearSVC(penalty='l2', loss='squared_hinge', dual=False,tol=1e-5,C=10)
    #model = LinearSVC(penalty='l1', loss='squared_hinge', dual=False)
    model.fit(X_mapped, responses)
    weights = model.coef_[0]  # Coefficients of the features
    bias = model.intercept_[0]  # Intercept term

    return weights, bias
    #return model

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
    binary_predictions = (predictions >= 0.5).astype(int)
    
    # Calculate accuracy
    accuracy = np.mean(binary_predictions == true_labels)
    
    return accuracy

# Example usage:
file_path = "train.dat"
inputs, responses = load_data(file_path)

acc=0
start_time=time.time()
w,b=my_fit(inputs,responses)
end_time=time.time()
training_time=end_time-start_time
# Now, you can use the solver to map test challenges and perform predictions
# For example:
test_challenges,test_responses = load_data("test.dat")
mapped_test_features = my_map(test_challenges)
#print(mapped_test_features)
predictions = mapped_test_features.dot(w)+b
#binary_predictions = convert_to_binary(predictions)
pred = np.zeros_like( predictions )
pred[predictions > 0] = 1
acc += np.average( test_responses == pred )
#accuracy = calculate_accuracy(predictions, test_responses)
print("Accuracy on Test Set:", acc)
print("training_time:",training_time)
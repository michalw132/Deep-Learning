# Only allowed the following libraries:
# Scikit-learn, Numpy, Matplotlib, Pandas
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Utility functions
def sigmoid(v):
    return 1/(1+np.exp(-v))

def sigmoid_derivative(v):
    return sigmoid(v)*(1-sigmoid(v))


class ANN:
    def __init__(self, input_size, hidden_size=16, learning_rate=0.05, seed=42):
        np.random.seed(seed) # set the random seed

        # uniform distribution between -0.1 and 0.1
        self.W1 = np.random.uniform(-0.1, 0.1, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.uniform(-0.1, 0.1, (hidden_size, 1))
        self.b2 = np.zeros((1, 1))

        self.lr = learning_rate

    # Forward propagation
    def forward(self, x):
        # Hidden layer
        self.in_hidden = x @ self.W1 + self.b1
        self.out_hidden = sigmoid(self.in_hidden)

        # Output layer
        self.in_output = self.out_hidden @ self.W2 + self.b2
        self.out_output = sigmoid(self.in_output)

        return self.out_output
    
    # Binary cross-entropy loss    
    def crossEntrop(self,o,y):
        eps = 1e-15
        return -np.mean(y*np.log(o+eps) + (1-y)* np.log(1-o+eps))

    def backward(self, x, y):
        m = x.shape[0]

        # Output layer
        dz2 = self.out_output - y
        dW2 = (self.out_hidden.T @ dz2) / m
        db2 = np.mean(dz2, axis=0, keepdims=True)

        # Hidden layer
        da1 = dz2 @ self.W2.T
        dz1 = da1 * sigmoid_derivative(self.in_hidden)
        dW1 = (x.T @ dz1) / m
        db1 = np.mean(dz1, axis=0, keepdims=True)

        # Update weights
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, x, y, x_test, y_test, epochs=150, batch_size=32):
        n = x.shape[0]
        losses = []
        test_accs = []

        for epoch in range(epochs):
            idx = np.random.permutation(n)
            xs = x[idx]
            ys = y[idx]

            # mini-batch loop
            for i in range(0, n, batch_size):
                x_mini = xs[i:i+batch_size, :]
                y_mini = ys[i:i+batch_size, :]

                # forward pass
                out = self.forward(x_mini)

                # backpropagation
                self.backward(x_mini, y_mini)

            full_out = self.forward(x)
            epoch_loss = self.crossEntrop(full_out, y)
            losses.append(epoch_loss)

            # compute test accuracy each epoch for task 2
            test_preds = self.predict(x_test)
            test_acc = np.mean(test_preds == y_test)
            test_accs.append(test_acc)

        return losses, test_accs

    def predict(self, x):
        return (self.forward(x) >= 0.5).astype(int)
        

# Main training function
def train_german_credit_ann(file_path="german_credit_simplified.txt"):
    # Load data
    data = pd.read_csv(file_path, sep=" ", header=None)

    # Split data set
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1, 1)

    # 80/20 split with static seed for reproducability
    # stratify to preserve class balance between train/test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)

    # Normalization
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Build the model
    model = ANN(
        input_size=24,
        hidden_size=16,
        learning_rate=0.05, # alter for Task 3
        seed=42
    )

    # Train model
    losses, test_accs = model.train(x_train, y_train, x_test, y_test, epochs=150, batch_size=32)

    # Outputs 
    final_train_loss = losses[-1]
    test_preds = model.predict(x_test)
    test_acc = np.mean(test_preds == y_test)

    print("Final training loss:", final_train_loss)
    print("Test accuracy:", test_acc, "\n")

    # Task 2 outputs
    z = np.arange(len(losses))

    plt.plot(z, losses, label="train loss", color='red')
    plt.plot(z, test_accs, label="test accuracy", color='blue')

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Loss and Test Accuracy")

    plt.legend(loc='best')
    plt.show()
    
    # Run task 3
    run_task_3(x_train, y_train, x_test, y_test)


# Task 3: Hyperparameter investigation
def run_task_3(x_train, y_train, x_test, y_test):
    print("Task 3: Hyperparameter investigation")
    learning_rate_options = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001] # other values to try: 5, 2, 0.5
    plt.figure(figsize=(10, 6))

    for rate in learning_rate_options:
        print(f"Training model with learning_rate={rate}")
        
        # Re-initialize the model for each time
        model = ANN(
            input_size=24,
            hidden_size=16,
            learning_rate=rate,
            seed=42
        )
        
        # Train and collect test accuracies
        _, test_acc = model.train(x_train, y_train, x_test, y_test, epochs=150, batch_size=32)
        
        # Plot test accuracy curve for this value
        plt.plot(range(150), test_acc, label=f'learning rate: {rate}')

        test_preds = model.predict(x_test)
        print("Test accuracy:", np.mean(test_preds == y_test))

    # Graph
    plt.title("Task 3: Impact of Learning Rate on Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Test Set Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Run the ANN
if __name__ == "__main__":
    train_german_credit_ann()
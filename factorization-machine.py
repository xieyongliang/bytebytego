import torch
import numpy as np

class FactorizationMachine(torch.nn.Module):
    def __init__(self, n_features, k):
        """
        n_features: Number of input features (size of X)
        k: Size of the embedding for second-order interactions
        """
        super(FactorizationMachine, self).__init__()
        
        # First-order weights (linear term)
        self.w = torch.nn.Parameter(torch.randn(n_features))
        
        # Second-order embeddings for each feature
        self.V = torch.nn.Parameter(torch.randn(n_features, k))  # Feature embedding matrix
        
        # Global bias term (optional)
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, X):
        """
        X: input feature vector of shape (batch_size, n_features)
        """
        # First-order term: sum(w_i * x_i)
        first_order = torch.matmul(X, self.w)  # Shape: (batch_size,)

        # Second-order term: sum interactions between pairs of features
        # Formula: sum_{i=1}^{n} sum_{j=i+1}^{n} (V_i . V_j) * x_i * x_j
        # Efficient implementation using vectorization:
        
        # Step 1: (X * V) --> element-wise multiplication between X and V
        X_embedded = torch.matmul(X, self.V)  # Shape: (batch_size, k)

        # (x1*v1)**2+(x2*v2)**2+ ... + (xk*vk)**2
        # (x1**2,x2**2, ..., xk**2) . (v1**2, v2**k, ..., vk**2)
        
        # Step 2: sum of squared embeddings
        sum_squared = torch.sum(X_embedded ** 2, dim=1)  # Shape: (batch_size,)
        
        # Step 3: squared sum of embeddings
        squared_sum = torch.sum(X ** 2 @ (self.V ** 2), dim=1)  # Shape: (batch_size,)
        
        # Second-order interaction term
        second_order = 0.5 * (sum_squared - squared_sum)  # Shape: (batch_size,)

        interaction_term = 0.5 * torch.sum(torch.matmul(X, self.V) ** 2 - torch.matmul(X ** 2, self.V ** 2), dim=1, keepdim=True)
        
        print(second_order, interaction_term)
        
        # Total prediction: sum of first-order, second-order, and bias
        y_hat = first_order + second_order + self.bias
        
        return y_hat

# Example usage
n_features = 10  # Number of features in the input vector
k = 5  # Size of the latent embedding vector for second-order interactions

# Initialize the Factorization Machine model
fm_model = FactorizationMachine(n_features, k)

# Create an example input feature vector (batch of size 2)
X = torch.randn(2, n_features)  # Shape: (batch_size=2, n_features=10)

# Forward pass
with torch.no_grad():
    output = fm_model(X)
print("FM Prediction:", output)

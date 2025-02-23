import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
VOCAB_SIZE = 10000  # Vocabulary size
EMBEDDING_DIM = 100  # Embedding dimension
CONTEXT_SIZE = 4  # Number of context words (2 on each side)
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# Example data (context-target pairs)
data = [
    ([1, 2, 4, 5], 3),  # Context: [1, 2, 4, 5], Target: 3
    ([2, 3, 5, 6], 4),  # Context: [2, 3, 5, 6], Target: 4
    # Add more data here
]

# CBOW Model
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, inputs):

        # Look up embeddings for context words
        embeds = self.embeddings(inputs)  # Shape: (batch_size, context_size, embedding_dim)

        # Average the embeddings
        h = torch.mean(embeds, dim=1)  # Shape: (batch_size, embedding_dim)

        # Predict the target word
        out = self.linear(h)  # Shape: (batch_size, vocab_size)

        return out

# Initialize model, loss function, and optimizer
model = CBOW(VOCAB_SIZE, EMBEDDING_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for context, target in data:
        # Convert context and target to tensors
        context_tensor = torch.tensor(context, dtype=torch.long)
        target_tensor = torch.tensor(torch.tensor(target).unsqueeze(dim=0), dtype=torch.long)
        context_tensor = context_tensor.unsqueeze(dim=0)

        # Forward pass
        output = model(context_tensor)
        loss = criterion(output, target_tensor)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {total_loss/len(data)}")
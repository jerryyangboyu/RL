import torch.nn as nn
import torch


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 3, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(675, 225), # output x and y coordinate
        )

    def forward(self, x):
        return self.network(x)


# Initialize the network
network = QNetwork()

# Generate a random input tensor of size (batch_size, channels, height, width)
# Note: PyTorch expects the input in the form of (N, C, H, W)
# where N is the batch size, C is the number of channels,
# H is the height, and W is the width of the image.
input_tensor = torch.rand(1, 3, 15, 15)  # Batch size of 1

# Pass the input tensor through the network
output = network(input_tensor)

# Print the output
print("Output shape:", output.shape)
print("Output:", output)

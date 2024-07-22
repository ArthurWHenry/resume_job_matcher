import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # Example input size of 768 and output size of 1
        self.fc = nn.Linear(768, 1)

    def forward(self, x):
        return self.fc(x)


# Instantiate the model
model = SimpleModel()

# Save the model
torch.save(model.state_dict(), './models/model.pth')

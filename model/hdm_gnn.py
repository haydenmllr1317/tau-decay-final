import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import pandas as pd

###################################################################################################
### HYPERPARAMETERS ###
epochs = 100
###################################################################################################

df = pd.read_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/all_nommass_reco_data_boosted.csv')




# Example data for a single triplet
node_features = torch.tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]], dtype=torch.float)
edge_index = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2], [1, 0], [2, 1]], dtype=torch.long)

# Example target vector for the fourth particle
target = torch.tensor([x4, y4, z4], dtype=torch.float)

# Create a PyTorch Geometric Data object
data = Data(x=node_features, edge_index=edge_index.t().contiguous(), y=target)

class GNNRegressor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNRegressor, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.mean(x, dim=0)  # Aggregate node features (e.g., mean pooling)
        x = self.fc(x)
        return x

# Instantiate the model, loss function, and optimizer
model = GNNRegressor(input_dim=3, hidden_dim=64, output_dim=3)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
def train(model, data, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

# Example data loader for batch processing
loader = DataLoader([data], batch_size=1, shuffle=True)

# Training the model
for epoch in range(epochs):
    for batch in loader:
        loss = train(model, batch, criterion, optimizer)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# Predict the three-vector of the fourth particle
model.eval()
with torch.no_grad():
    prediction = model(data)
    print("Predicted vector:", prediction)
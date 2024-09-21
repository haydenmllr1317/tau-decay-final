import pickle
# import ROOT
# from fast_histogram import histogram1d
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

class FourMomentum:
    def __init__(self, E, px, py, pz):
        self.vector = np.array([px, py, pz, E])

    @property
    def E(self):
        return self.vector[3]

    @property
    def px(self):
        return self.vector[0]

    @property
    def py(self):
        return self.vector[1]

    @property
    def pz(self):
        return self.vector[2]

    def __add__(self, other):
        return FourMomentum(*(self.vector + other.vector))

    def __sub__(self, other):
        return FourMomentum(*(self.vector - other.vector))

    def dot(self, other):
        return self.E * other.E - self.px * other.px - self.py * other.py - self.pz * other.pz

    def mass(self):
        mass_squared = self.dot(self)
        return np.sqrt(np.abs(mass_squared))







df = pd.read_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/reco_alldata.csv')
#df_gen = pd.read_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/gen_alldata.csv')
# print(df_gen.shape)
print(df.shape)

torch.set_num_threads(2)

df[['pi1_cosphi']] = np.cos(df[['pi1_phi']].values)
df[['pi1_sinphi']] = np.sin(df[['pi1_phi']].values)
df[['pi2_cosphi']] = np.cos(df[['pi2_phi']].values)
df[['pi2_sinphi']] = np.sin(df[['pi2_phi']].values)
df[['pi3_cosphi']] = np.cos(df[['pi3_phi']].values)
df[['pi3_sinphi']] = np.sin(df[['pi3_phi']].values)
df[['neu_cosphi']] = np.cos(df[['neu_phi']].values)
df[['neu_sinphi']] = np.sin(df[['neu_phi']].values)



X = df[['pi1_pt', 'pi1_eta', 'pi1_cosphi', 'pi1_sinphi',
        'pi2_pt', 'pi2_eta', 'pi2_cosphi', 'pi2_sinphi',
        'pi3_pt', 'pi3_eta', 'pi3_cosphi', 'pi3_sinphi',]].values
Y = df[['neu_pt', 'neu_eta', 'neu_cosphi', 'neu_sinphi']].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

batch_size = 376

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = CustomDataset(X_train, Y_train)
test_dataset = CustomDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=9433, shuffle=False)

class SimpleDNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc99 = nn.Linear(256, 256)
        # self.fc100 = nn.Linear(324, 428)
        # self.fc101 = nn.Linear(428,256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 96)
        # self.fc6 = nn.Linear(64, 48)
        #self.fc7 = nn.Linear(64, 48)
        self.fc8 = nn.Linear(96, 48)
        self.fc9 = nn.Linear(48, 24)
        #self.fc10 = nn.Linear(16, 12)
        self.fc11 = nn.Linear(24, 16)
        #self.fc12 = nn.Linear(8, 6)
        self.fc13 = nn.Linear(16, output_dim)
        # self.dropout_a = nn.Dropout(p=0.5)
        self.dropout_b = nn.Dropout(p=0.2)
        # self.dropout_c = nn.Dropout(p=0.1)
        # she had 12 layers, up to 2560 neurons, all relu with droppout from 0.3, up to 0.5, and then progressively down to 0.05

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.dropout_a(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout_a(x)
        x = F.relu(self.fc3(x))
        x = self.dropout_b(x)
        x = F.relu(self.fc99(x))
        # x = self.dropout_b(x)
        # x = F.tanh(self.fc100(x))
        # x = self.dropout_b(x)
        # x = F.tanh(self.fc101(x))
        x = F.relu(self.fc4(x))
        # x = self.dropout_b(x)
        x = F.relu(self.fc5(x))
        # x = self.dropout_c(x)
        # x = F.relu(self.fc6(x))
        # x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        # x = F.relu(self.fc10(x))
        x = F.relu(self.fc11(x))
        # x = F.relu(self.fc12(x))
        x = self.fc13(x)

        return x

pion_mass = 0.13957
def taumass(features, outputs, l):
    tensor = torch.empty(l, 1)
    for i in range(l):
        pi1 = FourMomentum(features[i,0], features[i,1], features[i,2], pion_mass)
        pi2 = FourMomentum(features[i,3], features[i,4], features[i,5], pion_mass)
        pi3 = FourMomentum(features[i,6], features[i,7], features[i,8], pion_mass)
        neu = FourMomentum(outputs[i,0], outputs[i,2], outputs[i,3], 0.0)
        taum = (pi1 + pi2 + pi3 + neu).mass()
        tensor[i,0] = torch.tensor(taum)
    return tensor.requires_grad_(True)


# Instantiate the model
input_dim = X_train.shape[1]
output_dim = Y_train.shape[1]
#print(output_dim)
model = SimpleDNN(input_dim, output_dim)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

training_loss = []
eval_loss = []

# Training loop
num_epochs = 100
model.train()
for epoch in range(num_epochs):
    for features, labels in train_loader:
        # Forward pass
        outputs = model(features)
        # loss = criterion(outputs.squeeze(), labels)
        # print(features.shape)
        # print(labels.shape)
        loss = criterion(outputs.squeeze(), labels) # + 0.1*criterion(taumass(features.detach(), outputs.squeeze().detach(), outputs.shape[0]), torch.full((features.shape[0], 1), 1.7769, requires_grad=True))
        # print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(name, param.grad)
        optimizer.step()
        outputs = model(features)
        # if epoch%10 == 0:
        #     # loss_eval = criterion(outputs.squeeze(), labels)
        #     loss_eval = criterion(taumass(features.detach(), outputs.squeeze().detach(), features.shape[0]), torch.full((features.shape[0], 1), 1.7769, requires_grad=True))
        #     print(loss_eval.item())
    if epoch%10 == 0:
        model.eval()
        with torch.no_grad():
            for features_eval, labels_eval in train_loader:
                outputs_eval = model(features_eval)
                loss_eval = criterion(outputs.squeeze(), labels) # + 0.1*criterion(taumass(features_eval.detach(), outputs_eval.squeeze().detach(), features_eval.shape[0]), torch.full((features_eval.shape[0], 1), 1.7769, requires_grad=True))
        training_loss.append(loss.item())
        eval_loss.append(loss_eval.item())
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        print(f'Eval [{epoch+1}/{num_epochs}], Loss: {loss_eval.item():.4f}')
        model.train()
    # scheduler.step(loss)


# Evaluation
model.eval()
count = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        count = count + param.numel()
print("Total param count:" + str(count))
with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)
        loss = criterion(outputs.squeeze(), labels)
        #print(loss)
    print(f'Testing Loss: {loss.item():.4f}')

index = list(range(len(training_loss)))
scaled_index = [i*10 for i in index]
eval_index = list(range(len(eval_loss)))
eval_scaled_index = [i*10 for i in eval_index]
plt.figure(figsize = (5,5))
plt.plot(scaled_index,training_loss, color='red')
plt.plot(eval_scaled_index, eval_loss, color='blue')
plt.title('Training vs Eval Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('training_vs_eval_loss.png')

# Create a scatter plot for the specified column
plt.figure(figsize=(5, 5))
plt.scatter(Y_test[:, 0], outputs[:, 0])
plt.title(f'Prediction of pT')
plt.xlabel('Validation Set pT')
plt.ylabel('Prediction of pT from Model')
plt.savefig('pt_plot.png')
plt.figure(figsize=(5, 5))
plt.scatter(Y_test[:, 1], outputs[:, 1])
plt.title(f'Prediction of eta')
plt.xlabel('Validation Set eta')
plt.ylabel('Prediction of eta from Model')
plt.savefig('eta_plot.png')
plt.figure(figsize=(5, 5))
plt.scatter(Y_test[:, 2], outputs[:, 2])
plt.title(f'Prediction of cosphi')
plt.xlabel('Validation Set cosphi')
plt.ylabel('Prediction of cosphi from Model')
plt.savefig('cosphi_plot.png')
plt.figure(figsize=(5, 5))
plt.scatter(Y_test[:, 3], outputs[:, 3])
plt.title(f'Prediction of sinphi')
plt.xlabel('Validation Set sinphi')
plt.ylabel('Prediction of phi from Model')
plt.savefig('sinphi_plot.png')
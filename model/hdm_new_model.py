import pickle
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

#######################################################################################################################
# HYPERPARAMETERS
num_epochs = 400
batch_size = 5000
test_batch_size = 7075
testing_size = 0.3 # percentage of data used for testing (currenlty split 50/50 into val and testing portions)
learning_rate = 0.01
lr_patience = 20 # how many epochs of plateud loss before learning rate is decreased
lr_delta = 0.2 # factor by which learning rate is decreased if patience is exhausted
weight_decay_rate = 0.0001
#######################################################################################################################

# globals for keeping track losses
training_losses = []
val_losses = []
min_loss = 999.0
epochs_since_min = 0

# first we have to create a class for fourmomentum vectors
class FourMomentum:
    def __init__(self, E, px, py, pz):
        self.vector = np.array([E, px, py, pz])

    @property
    def E(self):
        return self.vector[0]

    @property
    def px(self):
        return self.vector[1]

    @property
    def py(self):
        return self.vector[2]

    @property
    def pz(self):
        return self.vector[3]

    def __add__(self, other):
        return FourMomentum(*(self.vector + other.vector))

    def __sub__(self, other):
        return FourMomentum(*(self.vector - other.vector))

    def dot(self, other):
        return self.E * other.E - self.px * other.px - self.py * other.py - self.pz * other.pz

    def mass(self):
        mass_squared = self.dot(self)
        if mass_squared < 0:
            print('subzero mass is an issue')
        return np.sqrt(mass_squared)
    
# method for computing energy:
def compute_energy(px, py, pz, mass):
    E = torch.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return E

# switching inputs from polar to cartesian coordinates
def to_cartesian_features(features):
    lst = []
    lst.append(features[:,0]*features[:,2])
    lst.append(features[:,0]*features[:,3])
    lst.append(features[:,0]*np.sinh(features[:,1]))
    lst.append(features[:,4]*features[:,6])
    lst.append(features[:,4]*features[:,7])
    lst.append(features[:,4]*np.sinh(features[:,5]))
    lst.append(features[:,8]*features[:,10])
    lst.append(features[:,8]*features[:,11])
    lst.append(features[:,8]*np.sinh(features[:,9]))
    return lst

# switching outputs from polar to cartesian coordinates
def to_cartesian_outputs(outputs):
    lst = []
    lst.append(outputs[:,0]*outputs[:,2])
    lst.append(outputs[:,0]*outputs[:,3])
    lst.append(outputs[:,0]*np.sinh(outputs[:,1]))
    return lst

# method for computing tau mass from 3 pions and a neutrino   
pion_mass = 0.13957
def taumass(features, outputs, l): # features are pion info, outputs are predicted neutrino info, l is length of data
    tensor = torch.empty(l, 1)
    l_features = to_cartesian_features(features)
    l_outputs = to_cartesian_outputs(outputs)
    for i in range(l):
        pi1 = FourMomentum(compute_energy(l_features[0][i], l_features[1][i], l_features[2][i], pion_mass), l_features[0][i], l_features[1][i], l_features[2][i])
        pi2 = FourMomentum(compute_energy(l_features[3][i], l_features[4][i], l_features[5][i], pion_mass), l_features[3][i], l_features[4][i], l_features[5][i])
        pi3 = FourMomentum(compute_energy(l_features[6][i], l_features[7][i], l_features[8][i], pion_mass), l_features[6][i], l_features[7][i], l_features[8][i])
        neu = FourMomentum(compute_energy(l_outputs[0][i], l_outputs[1][i], l_outputs[2][i], 0.0), l_outputs[0][i], l_outputs[1][i], l_outputs[2][i])
        taum = (pi1 + pi2 + pi3 + neu).mass()
        tensor[i,0] = torch.tensor(taum)
    return tensor.requires_grad_(True)

# now we input our CSV file as a data frame
df = pd.read_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/df_boosted_highest_pt_pion_spherical.csv')
print('The shape of the input data is:' + str(df.shape))

# this should limit us to using only 2 CPU threads on whatever computer we are SSHed into
torch.set_num_threads(2)

# 


# now we switch from phi to cossphi and sinphi, the double brackets maintains the data frame structure of the columns
# (single brackets would have made it a pandas sequence instead)
df[['pi1_cosphi']] = np.cos(df[['pi1_phi']].values)
df[['pi1_sinphi']] = np.sin(df[['pi1_phi']].values)
df[['pi2_cosphi']] = np.cos(df[['pi2_phi']].values)
df[['pi2_sinphi']] = np.sin(df[['pi2_phi']].values)
df[['pi3_cosphi']] = np.cos(df[['pi3_phi']].values)
df[['pi3_sinphi']] = np.sin(df[['pi3_phi']].values)
df[['neu_cosphi']] = np.cos(df[['neu_phi']].values)
df[['neu_sinphi']] = np.sin(df[['neu_phi']].values)

# # now we are taking each of our 12 input variables and normalizing them to have mean 0 and RMS 1
# # there are many ways to do this, including built in functions, that might be better
# df[['pi1_pt']] = (df[['pi1_pt']] - df[['pi1_pt']].mean())/df[['pi1_pt']].std()
# print("the rms should be 1 and it is:" + str(np.sqrt(np.mean(df['pi1_pt']**2))))
# df[['pi2_pt']] = (df[['pi2_pt']] - df[['pi2_pt']].mean())/df[['pi2_pt']].std()
# df[['pi3_pt']] = (df[['pi3_pt']] - df[['pi3_pt']].mean())/df[['pi3_pt']].std()
# df[['pi1_eta']] = (df[['pi1_eta']] - df[['pi1_eta']].mean())/df[['pi1_eta']].std()
# df[['pi2_eta']] = (df[['pi2_eta']] - df[['pi2_eta']].mean())/df[['pi2_eta']].std()
# df[['pi3_eta']] = (df[['pi3_eta']] - df[['pi3_eta']].mean())/df[['pi3_eta']].std()
# df[['pi1_cosphi']] = (df[['pi1_cosphi']] - df[['pi1_cosphi']].mean())/df[['pi1_cosphi']].std()
# df[['pi1_sinphi']] = (df[['pi1_sinphi']] - df[['pi1_sinphi']].mean())/df[['pi1_sinphi']].std()
# df[['pi2_cosphi']] = (df[['pi2_cosphi']] - df[['pi2_cosphi']].mean())/df[['pi2_cosphi']].std()
# df[['pi2_sinphi']] = (df[['pi2_sinphi']] - df[['pi2_sinphi']].mean())/df[['pi2_sinphi']].std()
# df[['pi3_cosphi']] = (df[['pi3_cosphi']] - df[['pi3_cosphi']].mean())/df[['pi3_cosphi']].std()
# df[['pi3_sinphi']] = (df[['pi3_sinphi']] - df[['pi3_sinphi']].mean())/df[['pi3_sinphi']].std()

# we specify that input columns and output columns of our model
X = df[['pi1_pt', 'pi1_eta', 'pi1_cosphi', 'pi1_sinphi',
        'pi2_pt', 'pi2_eta', 'pi2_cosphi', 'pi2_sinphi',
        'pi3_pt', 'pi3_eta', 'pi3_cosphi', 'pi3_sinphi',]].values
Y = df[['neu_pt', 'neu_eta', 'neu_cosphi', 'neu_sinphi']].values

# we split into train, val, and testing data sets
X_train, X_other, Y_train, Y_other = train_test_split(X, Y, test_size=testing_size, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_other,Y_other, test_size=0.5, random_state=42)

# creates a class to have a set way we read in datasets, makes it compatable with dataloader
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
# creates our datasets in the appropriate format
train_dataset = CustomDataset(X_train, Y_train)
val_dataset = CustomDataset(X_val, Y_val)
test_dataset = CustomDataset(X_test, Y_test)

# loads the data in a way that is easy for pytorch nn models
# combines dataset with a sampler and provides iterable over the dataset
# loads and shuffles data into batches
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

# model class
class DNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 640)
        self.fc2 = nn.Linear(640,640)
        self.fc3 = nn.Linear(640,640)
        self.fc4 = nn.Linear(640,640)
        self.fc5 = nn.Linear(640,640)
        self.fcn = nn.Linear(640, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.05)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.fcn(x) # no activation function on output layer for regression tasks
        return x

# instantiating the model
input_dim = X_train.shape[1] # should be 12, the number of features
output_dim = Y_train.shape[1] # should be 4, the number of labels
model = DNN(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_rate)
# weight decay discourages weight that are particularly large, it is intended to prevent overfitting
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=lr_delta, patience=lr_patience, verbose=True)

# TRAINING!
def train_model(model):
    global min_loss
    global epochs_since_min
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
        
            loss = criterion(outputs.squeeze(), labels) # + 0.05*criterion(taumass(features.detach(), outputs.squeeze().detach(), outputs.shape[0]), torch.full((features.shape[0], 1), 1.7769, requires_grad=True))
            # print(loss)
            loss.backward()
            optimizer.step()
            total_loss += loss * features.size(0)

        if epoch%10 == 0:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for features_val, labels_val in val_loader:
                    outputs_val = model(features_val)
                    val_loss = criterion(outputs_val.squeeze(), labels_val) # + 0.05*criterion(taumass(features_val.detach(), outputs_val.squeeze().detach(), outputs_val.shape[0]), torch.full((features_val.shape[0], 1), 1.7769, requires_grad=True))
                    total_val_loss += val_loss * features_val.size(0)
            avg_loss = total_loss.item()/len(train_loader.dataset)
            avg_val_loss = total_val_loss.item()/len(val_loader.dataset)

            training_losses.append(avg_loss)
            val_losses.append(avg_val_loss)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            print(f'Val [{epoch+1}/{num_epochs}], Loss: {val_loss.item():.4f}')
            if avg_val_loss < 0.95*min_loss:
                min_loss = avg_val_loss
            else:
                epochs_since_min += 1
                if epochs_since_min >= 4:
                    return
        scheduler.step(loss)
    return
train_model(model)

# TESTING!
model.eval()
test_loss = 0.0
with torch.no_grad():
    for test_features, test_labels in test_loader:
        test_outputs = model(test_features)
        loss = criterion(test_outputs.squeeze(), test_labels) # + 0.05*criterion(taumass(test_features.detach(), test_outputs.squeeze().detach(), test_outputs.shape[0]), torch.full((test_features.shape[0], 1), 1.7769, requires_grad=True))
        test_loss += loss.item() * test_features.size(0)
test_loss /= len(test_loader.dataset)
print(f'Testing Loss: {test_loss:.4f}')

# parameter counting
count = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        count = count + param.numel()
print("Total param count:" + str(count))

# HISTOGRAMS:

# Loss Curve
scaled_index = [i*10 for i in range(len(training_losses))]
val_scaled_index = [i*10 for i in range(len(val_losses))]
plt.figure(figsize = (5,5))
plt.plot(scaled_index,training_losses, color='red', label = 'Training Losses')
plt.plot(val_scaled_index, val_losses, color='blue', label = 'Val Losses')
plt.title('Training vs Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_vs_val_loss.png')

# values of labels vs outputs
plt.figure(figsize=(5, 5))
plt.scatter(Y_test[:, 0], test_outputs[:, 0])
plt.title(f'Prediction of pT')
plt.xlabel('Validation Set pT')
plt.ylabel('Prediction of pT from Model')
plt.savefig('pt_plot.png')
plt.figure(figsize=(5, 5))
plt.scatter(Y_test[:, 1], test_outputs[:, 1])
plt.title(f'Prediction of eta')
plt.xlabel('Validation Set eta')
plt.ylabel('Prediction of eta from Model')
plt.savefig('eta_plot.png')
plt.figure(figsize=(5, 5))
plt.scatter(Y_test[:, 2], test_outputs[:, 2])
plt.title(f'Prediction of cosphi')
plt.xlabel('Validation Set cosphi')
plt.ylabel('Prediction of cosphi from Model')
plt.savefig('cosphi_plot.png')
plt.figure(figsize=(5, 5))
plt.scatter(Y_test[:, 3], test_outputs[:, 3])
plt.title(f'Prediction of sinphi')
plt.xlabel('Validation Set sinphi')
plt.ylabel('Prediction of phi from Model')
plt.savefig('sinphi_plot.png')
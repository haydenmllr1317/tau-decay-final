import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau


#######################################################################################################################
# HYPERPARAMETERS
portion_for_testing_and_eval = 0.3
batch_size = 5000
num_epochs = 400
learning_rate = 0.01
weight_decay = 0.0001
learning_rate_delta = 0.2
learning_rate_patience = 30
epochs_until_kill = 17 # (in increments of 10 epochs, meaning inputting 4 here gives you 40 epochs of tolerance)
#######################################################################################################################

# import dataframe
#df = pd.read_csv('/afs/cern.ch/user/h/hmiller/private/tau-decay-ml/CMSSW_13_3_0/src/df_boosted_highest_pt_pion_spherical.csv')
df = pd.read_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/reco_nommass_vis_tau_COM_frame.csv')
print('Dataframe succesfully created.')

df_X = pd.DataFrame()
df_Y = pd.DataFrame()

df_X['pi1_pt'] = df['pi1_pt']
df_X['pi2_pt'] = df['pi2_pt']
df_X['pi3_pt'] = df['pi3_pt']

df_X['pi1_eta'] = df['pi1_eta'] 
df_X['pi2_eta'] = df['pi2_eta'] 
df_X['pi3_eta'] = df['pi3_eta']

df_X['pi1_cosphi'] = np.cos(df['pi1_phi'])
df_X['pi2_cosphi'] = np.cos(df['pi2_phi'])
df_X['pi3_cosphi'] = np.cos(df['pi3_phi'])

df_X['pi1_sinphi'] = np.sin(df['pi1_phi'])
df_X['pi2_sinphi'] = np.sin(df['pi2_phi'])
df_X['pi3_sinphi'] = np.sin(df['pi3_phi'])

df_Y['neu_pt'] = df['neu_pt']
df_Y['neu_eta'] = df['neu_eta'] 
df_Y['neu_cosphi'] = np.cos(df['neu_phi'])
df_Y['neu_sinphi'] = np.sin(df['neu_phi'])

# now we are taking each of our 12 input variables and normalizing them to have mean 0 and RMS 1
# there are many ways to do this, including built in functions, that might be better
df_X['pi1_pt']= (df_X['pi1_pt'] - df_X['pi1_pt'].mean())/df_X['pi1_pt'].std()
print("the rms should be 1 and it is:" + str(np.sqrt(np.mean(df_X['pi1_pt']**2))))
df_X['pi2_pt'] = (df_X['pi2_pt'] - df_X['pi2_pt'].mean())/df_X['pi2_pt'].std()
df_X['pi3_pt'] = (df_X['pi3_pt'] - df_X['pi3_pt'].mean())/df_X['pi3_pt'].std()
df_X['pi1_eta'] = (df_X['pi1_eta'] - df_X['pi1_eta'].mean())/df_X['pi1_eta'].std()
df_X['pi2_eta'] = (df_X['pi2_eta'] - df_X['pi2_eta'].mean())/df_X['pi2_eta'].std()
df_X['pi3_eta'] = (df_X['pi3_eta'] - df_X['pi3_eta'].mean())/df_X['pi3_eta'].std()
df_X['pi1_cosphi'] = (df_X['pi1_cosphi'] - df_X['pi1_cosphi'].mean())/df_X['pi1_cosphi'].std()
df_X['pi1_sinphi'] = (df_X['pi1_sinphi'] - df_X['pi1_sinphi'].mean())/df_X['pi1_sinphi'].std()
df_X['pi2_cosphi'] = (df_X['pi2_cosphi'] - df_X['pi2_cosphi'].mean())/df_X['pi2_cosphi'].std()
df_X['pi2_sinphi'] = (df_X['pi2_sinphi'] - df_X['pi2_sinphi'].mean())/df_X['pi2_sinphi'].std()
df_X['pi3_cosphi'] = (df_X['pi3_cosphi'] - df_X['pi3_cosphi'].mean())/df_X['pi3_cosphi'].std()
df_X['pi3_sinphi'] = (df_X['pi3_sinphi'] - df_X['pi3_sinphi'].mean())/df_X['pi3_sinphi'].std()
df_Y['neu_pt'] = (df_Y['neu_pt'] - df_Y['neu_pt'].mean())/df_Y['neu_pt'].std()
df_Y['neu_eta'] = (df_Y['neu_eta'] - df_Y['neu_eta'].mean())/df_Y['neu_eta'].std()
df_Y['neu_cosphi'] = (df_Y['neu_cosphi'] - df_Y['neu_cosphi'].mean())/df_Y['neu_cosphi'].std()
df_Y['neu_sinphi'] = (df_Y['neu_sinphi'] - df_Y['neu_sinphi'].mean())/df_Y['neu_sinphi'].std()

print('Succesfully normalized all data.')

X_train, X_other, Y_train, Y_other = train_test_split(df_X, df_Y, test_size=portion_for_testing_and_eval, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_other, Y_other, test_size=0.5, random_state=42)

X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_val = torch.tensor(X_val.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
Y_train = torch.tensor(Y_train.values, dtype=torch.float32)
Y_val = torch.tensor(Y_val.values, dtype=torch.float32)
Y_test = torch.tensor(Y_test.values, dtype=torch.float32)

train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
test_dataset = TensorDataset(X_test, Y_test)
batch_size = batch_size

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print('Succesfully loaded data.')

torch.set_num_threads(2)

class DNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 512)
        # self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(512, 256)
        # self.layer5 = nn.Linear(256, 256)
        # self.layer6 = nn.Linear(256,256)
        # self.layer7 = nn.Linear(256,64)
        self.layer8 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=0.05)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        # x = self.dropout_1(x)
        x = self.relu(self.layer2(x))
        # x = self.dropout_1(x)
        # x = self.relu(self.layer3(x))
        # x = self.dropout_1(x)
        x = self.relu(self.layer4(x))
        # x = self.dropout_1(x)
        # x = self.relu(self.layer5(x))
        # x = self.dropout_1(x)
        # x = self.relu(self.layer6(x))
        # x = self.dropout_1(x)
        # x = self.relu(self.layer7(x))
        x = self.layer8(x)
        return x

# pion_mass = 0.13957

# def compute_energy(px, py, pz, mass):
#     E = torch.sqrt(px**2 + py**2 + pz**2 + mass**2)
#     return E

# def custom_loss(pion_momenta, neutrino_gen, model_outputs):
#     pi1_px, pi1_py, pi1_pz = pion_momenta[:, 0], pion_momenta[:, 3], pion_momenta[:, 6]
#     pi2_px, pi2_py, pi2_pz = pion_momenta[:, 1], pion_momenta[:, 4], pion_momenta[:, 7]
#     pi3_px, pi3_py, pi3_pz = pion_momenta[:, 2], pion_momenta[:, 5], pion_momenta[:, 8]

#     neu_gen_px, neu_gen_py, neu_gen_pz = neutrino_gen[:, 0], neutrino_gen[:, 1], neutrino_gen[:, 2]
#     neu_pred_px, neu_pred_py, neu_pred_pz = model_outputs[:, 0], model_outputs[:, 1], model_outputs[:, 2]

#     pi1_E = compute_energy(pi1_px, pi1_py, pi1_pz, pion_mass)
#     pi2_E = compute_energy(pi2_px, pi2_py, pi2_pz, pion_mass)
#     pi3_E = compute_energy(pi3_px, pi3_py, pi3_pz, pion_mass)

#     neu_gen_E = torch.sqrt(neu_gen_px**2 + neu_gen_py**2 + neu_gen_pz**2)
#     neu_pred_E = torch.sqrt(neu_pred_px**2 + neu_pred_py**2 + neu_pred_pz**2)

#     tau_gen_px = pi1_px + pi2_px + pi3_px + neu_gen_px
#     tau_gen_py = pi1_py + pi2_py + pi3_py + neu_gen_py
#     tau_gen_pz = pi1_pz + pi2_pz + pi3_pz + neu_gen_pz
#     tau_gen_E = pi1_E + pi2_E + pi3_E + neu_gen_E

#     tau_pred_px = pi1_px + pi2_px + pi3_px + neu_pred_px
#     tau_pred_py = pi1_py + pi2_py + pi3_py + neu_pred_py
#     tau_pred_pz = pi1_pz + pi2_pz + pi3_pz + neu_pred_pz
#     tau_pred_E = pi1_E + pi2_E + pi3_E + neu_pred_E

#     tau_gen_mass = torch.sqrt(tau_gen_E**2 - (tau_gen_px**2 + tau_gen_py**2 + tau_gen_pz**2))
#     tau_pred_mass = torch.sqrt(tau_pred_E**2 - (tau_pred_px**2 + tau_pred_py**2 + tau_pred_pz**2))

#     loss = torch.mean((tau_gen_mass - tau_pred_mass)**2)
#     return loss

# criterion1 = custom_loss
mse = nn.MSELoss()
input_dim = X_train.shape[1]
output_dim = Y_train.shape[1]
model = DNN(input_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
num_epochs = num_epochs
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=learning_rate_delta, patience=learning_rate_patience, verbose=1)


training_losses = []
val_losses = []
min_loss = 999.0
epochs_since_min = 0

print('Beginning training process.')

# TRAINING!
def train_model(model):
    global min_loss
    global epochs_since_min
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = mse(outputs,batch_y)
            # loss = 0.1*criterion1(batch_X, batch_y, outputs) + 0.9*mse(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        training_loss = running_loss / len(train_loader)
        training_losses.append(training_loss)

        val_running_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                outputs_val = model(batch_X_val)
                val_loss = mse(outputs_val,batch_y_val)
                # val_loss = 0.1*criterion1(batch_X_val, batch_y_val, outputs_val) + 0.9*mse(outputs_val, batch_y_val)
                val_running_loss += val_loss.item()

        val_loss = val_running_loss / len(val_loader)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {training_loss:.4f}, Validation Loss: {val_loss:.4f}')
            if val_loss < 0.95*min_loss:
                min_loss = val_loss
            else:
                epochs_since_min += 1
                if epochs_since_min >= epochs_until_kill:
                    return
        scheduler.step(loss)
    return
train_model(model)


# training vs validation losses through the run
plt.figure(figsize=(10, 5))
plt.plot(training_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.savefig('trainings_vs_val_loss.png')

# model testing
model.eval()
count = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        count = count + param.numel()
print("Total param count:" + str(count))
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = mse(test_outputs,Y_test)
    # test_loss = 0.1*criterion1(X_test, Y_test, test_outputs) + 0.9*mse(test_outputs, Y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

# pt, eta, cosphi, sinphi, prediction vs testing set pt
plt.figure(figsize=(5, 5))
plt.scatter(Y_test[:, 0], test_outputs[:, 0])
plt.title(f'Prediction of pT')
plt.xlabel('Test Set pT')
plt.ylabel('Prediction of pT from Model')
plt.savefig('pt.png')
plt.figure(figsize=(5, 5))
plt.scatter(Y_test[:, 1], test_outputs[:, 1])
plt.title(f'Prediction of eta')
plt.xlabel('Test Set eta')
plt.ylabel('Prediction of eta from Model')
plt.savefig('eta.png')
plt.figure(figsize=(5, 5))
plt.scatter(Y_test[:, 2], test_outputs[:, 2])
plt.title(f'Prediction of cosphi')
plt.xlabel('Test Set cosphi')
plt.ylabel('Prediction of cosphi from Model')
plt.savefig('cosphi.png')
plt.figure(figsize=(5, 5))
plt.scatter(Y_test[:, 2], test_outputs[:, 2])
plt.title(f'Prediction of sinphi')
plt.xlabel('Test Set sinphi')
plt.ylabel('Prediction of sinphi from Model')
plt.savefig('sinphi.png')



outputs_array = test_outputs.numpy()
Y_test_array = Y_test.numpy()
X_test_array = X_test.numpy()


df_x_test = pd.DataFrame(X_test_array, columns=['pi1_pt','pi1_eta','pi1_cosphi','pi1_sinphi','pi2_pt','pi2_eta','pi2_cosphi','pi2_sinphi','pi3_pt','pi3_eta','pi3_cosphi','pi3_sinphi'])
df_y_test = pd.DataFrame(Y_test_array, columns=['neu_pt','neu_eta','neu_cosphi', 'neu_sinphi'])
df_y_prediction = pd.DataFrame(outputs_array, columns=['neu_pt','neu_eta','neu_cosphi', 'neu_sinphi'])

df_tau_mass = pd.concat([df_x_test, df_y_test, df_y_prediction], axis = 1)
df_tau_mass.to_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/model_output.csv')
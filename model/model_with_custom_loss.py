import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt

# Loading and preparing the dataset
df_toUse = pd.read_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/df_boosted_highest_pt_pion_spherical.csv')

new_df_X = pd.DataFrame()
new_df_Y = pd.DataFrame()

new_df_X['px_pi_1m'] = df_toUse['pi1_pt'] * np.cos(df_toUse['pi1_phi'])
new_df_X['px_pi_2m'] = df_toUse['pi2_pt'] * np.cos(df_toUse['pi2_phi'])
new_df_X['px_pi_3m'] = df_toUse['pi3_pt'] * np.cos(df_toUse['pi3_phi'])

new_df_X['py_pi_1m'] = df_toUse['pi1_pt'] * np.sin(df_toUse['pi1_phi'])
new_df_X['py_pi_2m'] = df_toUse['pi2_pt'] * np.sin(df_toUse['pi2_phi'])
new_df_X['py_pi_3m'] = df_toUse['pi3_pt'] * np.sin(df_toUse['pi3_phi'])

new_df_X['pz_pi_1m'] = df_toUse['pi1_pt'] * np.sinh(df_toUse['pi1_eta'])
new_df_X['pz_pi_2m'] = df_toUse['pi2_pt'] * np.sinh(df_toUse['pi2_eta'])
new_df_X['pz_pi_3m'] = df_toUse['pi3_pt'] * np.sinh(df_toUse['pi3_eta'])

new_df_Y['px_neu'] = df_toUse['neu_pt'] * np.cos(df_toUse['neu_phi'])
new_df_Y['py_neu'] = df_toUse['neu_pt'] * np.sin(df_toUse['neu_phi'])
new_df_Y['pz_neu'] = df_toUse['neu_pt'] * np.sinh(df_toUse['neu_eta'])

df_toUse_neutrino_X = new_df_X
df_toUse_neutrino_Y = new_df_Y

X_train, X_test1, y_train, y_test1 = train_test_split(df_toUse_neutrino_X, df_toUse_neutrino_Y, test_size=0.2, random_state=20)
X_test, X_val, y_test, y_val = train_test_split(X_test1, y_test1, test_size=0.5, random_state=42)

X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_val = torch.tensor(X_val.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)
batch_size = 4000

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

torch.set_num_threads(2)

class SimpleLinearNN2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLinearNN2, self).__init__()
        self.layer1 = nn.Linear(input_dim, 640)
        self.layer2 = nn.Linear(640, 640)
        self.layer3 = nn.Linear(640, 640)
        self.layer4 = nn.Linear(640, 640)
        self.layer5 = nn.Linear(640, 640)
        self.layer6 = nn.Linear(640, output_dim)
        self.relu = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=0.05)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout_1(x)
        x = self.relu(self.layer2(x))
        x = self.dropout_1(x)
        x = self.relu(self.layer3(x))
        x = self.dropout_1(x)
        x = self.relu(self.layer4(x))
        x = self.dropout_1(x)
        x = self.relu(self.layer5(x))
        x = self.layer6(x)
        return x

pion_mass = 0.13957

def compute_mass(px, py, pz, mass):
    E = torch.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return E

def custom_loss(pion_momenta, neutrino_gen, model_outputs):
    pi1_px, pi1_py, pi1_pz = pion_momenta[:, 0], pion_momenta[:, 3], pion_momenta[:, 6]
    pi2_px, pi2_py, pi2_pz = pion_momenta[:, 1], pion_momenta[:, 4], pion_momenta[:, 7]
    pi3_px, pi3_py, pi3_pz = pion_momenta[:, 2], pion_momenta[:, 5], pion_momenta[:, 8]

    neu_gen_px, neu_gen_py, neu_gen_pz = neutrino_gen[:, 0], neutrino_gen[:, 1], neutrino_gen[:, 2]
    neu_pred_px, neu_pred_py, neu_pred_pz = model_outputs[:, 0], model_outputs[:, 1], model_outputs[:, 2]

    pi1_E = compute_mass(pi1_px, pi1_py, pi1_pz, pion_mass)
    pi2_E = compute_mass(pi2_px, pi2_py, pi2_pz, pion_mass)
    pi3_E = compute_mass(pi3_px, pi3_py, pi3_pz, pion_mass)

    neu_gen_E = torch.sqrt(neu_gen_px**2 + neu_gen_py**2 + neu_gen_pz**2)
    neu_pred_E = torch.sqrt(neu_pred_px**2 + neu_pred_py**2 + neu_pred_pz**2)

    tau_gen_px = pi1_px + pi2_px + pi3_px + neu_gen_px
    tau_gen_py = pi1_py + pi2_py + pi3_py + neu_gen_py
    tau_gen_pz = pi1_pz + pi2_pz + pi3_pz + neu_gen_pz
    tau_gen_E = pi1_E + pi2_E + pi3_E + neu_gen_E

    tau_pred_px = pi1_px + pi2_px + pi3_px + neu_pred_px
    tau_pred_py = pi1_py + pi2_py + pi3_py + neu_pred_py
    tau_pred_pz = pi1_pz + pi2_pz + pi3_pz + neu_pred_pz
    tau_pred_E = pi1_E + pi2_E + pi3_E + neu_pred_E

    tau_gen_mass = torch.sqrt(tau_gen_E**2 - (tau_gen_px**2 + tau_gen_py**2 + tau_gen_pz**2))
    tau_pred_mass = torch.sqrt(tau_pred_E**2 - (tau_pred_px**2 + tau_pred_py**2 + tau_pred_pz**2))

    loss = torch.mean((tau_gen_mass - tau_pred_mass)**2)
    return loss

criterion1 = custom_loss
criterion2 = nn.MSELoss()
model = SimpleLinearNN2(9, 3)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
num_epochs = 300

training_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = 0.5*criterion1(batch_X, batch_y, outputs) + 0.5*criterion2(outputs, batch_y)
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
            val_loss = criterion1(batch_X_val, batch_y_val, outputs_val) + criterion2(outputs_val, batch_y_val)
            val_running_loss += val_loss.item()

    val_loss = val_running_loss / len(val_loader)
    val_losses.append(val_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {training_loss:.4f}, Validation Loss: {val_loss:.4f}')

# Plot the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(training_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('loss_plot2.png')

# torch.save(model.state_dict(), 'our_data_model_custom_loss2.pth')


###################################################################################################

# model.load_state_dict(torch.load('our_data_model_custom_loss2.pth'))

model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = 0.5*criterion1(X_test, y_test, test_outputs) + 0.5*criterion2(test_outputs, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

plt.figure(figsize=(5, 5))
plt.scatter(y_test[:, 0], test_outputs[:, 0])
plt.title(f'Prediction of Px')
plt.xlabel('Validation Set Px')
plt.ylabel('Prediction of Px from Model')
plt.savefig('px_plot2.png')
plt.figure(figsize=(5, 5))
plt.scatter(y_test[:, 1], test_outputs[:, 1])
plt.title(f'Prediction of Py')
plt.xlabel('Validation Set Py')
plt.ylabel('Prediction of Py from Model')
plt.savefig('py_plot2.png')
plt.figure(figsize=(5, 5))
plt.scatter(y_test[:, 2], test_outputs[:, 2])
plt.title(f'Prediction of Pz')
plt.xlabel('Validation Set Pz')
plt.ylabel('Prediction of Pz from Model')
plt.savefig('pz_plot2.png')

# # Step 2: Convert the tensor to a NumPy array
# outputs_array = test_outputs.numpy()
# y_test_array = y_test.numpy()
# X_test_array = X_test.numpy()

# # Step 3: Convert the NumPy array to a Pandas DataFrame
# df_x_test = pd.DataFrame(X_test_array, columns=['px_pi_1m','px_pi_2m','px_pi_3m','py_pi_1m','py_pi_2m','py_pi_3m','pz_pi_1m','pz_pi_2m','pz_pi_3m'])
# df_y_test = pd.DataFrame(y_test_array, columns=['px_neu_test','py_neu_test','pz_neu_test'])
# df_y_prediction = pd.DataFrame(outputs_array, columns=['px_neu_pred','py_neu_pred','pz_neu_pred'])

# df_tau_mass = pd.concat([df_x_test, df_y_test, df_y_prediction], axis = 1)
# df_tau_mass.to_csv('model_output_tau_mass_custom_loss2.csv')
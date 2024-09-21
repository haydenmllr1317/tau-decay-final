import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from fast_histogram import histogram1d

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Loading and preparing the dataset
df_toUse = pd.read_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/all_nommass_reco_data_boosted.csv')

#df_toUse = pd.concat([df_toUse_nominal, df_toUse_15], axis = 0)

new_df_X = pd.DataFrame()
new_df_Y = pd.DataFrame()


"""
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
"""

max_pt_neu = df_toUse['neu_pt'].max()
# print(max_pt_neu)
new_df_X['pt_pi_1m'] = df_toUse['pi1_pt']
new_df_X['pt_pi_2m'] = df_toUse['pi2_pt']
new_df_X['pt_pi_3m'] = df_toUse['pi3_pt']

new_df_X['eta_pi_1m'] = df_toUse['pi1_eta']
new_df_X['eta_pi_2m'] = df_toUse['pi2_eta']
new_df_X['eta_pi_3m'] = df_toUse['pi3_eta']

new_df_X['cos_phi_pi_1m'] = np.cos(df_toUse['pi1_phi'])
new_df_X['cos_phi_pi_2m'] = np.cos(df_toUse['pi2_phi'])
new_df_X['cos_phi_pi_3m'] = np.cos(df_toUse['pi3_phi'])

new_df_X['sin_phi_pi_1m'] = np.sin(df_toUse['pi1_phi'])
new_df_X['sin_phi_pi_2m'] = np.sin(df_toUse['pi2_phi'])
new_df_X['sin_phi_pi_3m'] = np.sin(df_toUse['pi3_phi'])

new_df_Y['pt_neu'] = df_toUse['neu_pt']
new_df_Y['eta_neu'] = df_toUse['neu_eta']
new_df_Y['cos_phi_neu'] = np.cos(df_toUse['neu_phi'])
new_df_Y['sin_phi_neu'] = np.sin(df_toUse['neu_phi'])
"""
def plot_histograms():
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # pt histograms
    axes[0, 0].hist(new_df_X['pt_pi_1m'], bins=50, alpha=0.5, label='pt_pi_1m')
    axes[0, 0].hist(new_df_X['pt_pi_2m'], bins=50, alpha=0.5, label='pt_pi_2m')
    axes[0, 0].hist(new_df_X['pt_pi_3m'], bins=50, alpha=0.5, label='pt_pi_3m')
    axes[0, 0].hist(new_df_Y['pt_neu'], bins=50, alpha=0.5, label='pt_neu')
    axes[0, 0].set_title('pt Histograms')
    axes[0, 0].legend()

    # eta histograms
    axes[0, 1].hist(new_df_X['eta_pi_1m'], bins=50, alpha=0.5, label='eta_pi_1m')
    axes[0, 1].hist(new_df_X['eta_pi_2m'], bins=50, alpha=0.5, label='eta_pi_2m')
    axes[0, 1].hist(new_df_X['eta_pi_3m'], bins=50, alpha=0.5, label='eta_pi_3m')
    axes[0, 1].hist(new_df_Y['eta_neu'], bins=50, alpha=0.5, label='eta_neu')
    axes[0, 1].set_title('eta Histograms')
    axes[0, 1].legend()

    # cos_phi histograms
    axes[1, 0].hist(new_df_X['cos_phi_pi_1m'], bins=50, alpha=0.5, label='cos_phi_pi_1m')
    axes[1, 0].hist(new_df_X['cos_phi_pi_2m'], bins=50, alpha=0.5, label='cos_phi_pi_2m')
    axes[1, 0].hist(new_df_X['cos_phi_pi_3m'], bins=50, alpha=0.5, label='cos_phi_pi_3m')
    axes[1, 0].hist(new_df_Y['cos_phi_neu'], bins=50, alpha=0.5, label='cos_phi_neu')
    axes[1, 0].set_title('cos_phi Histograms')
    axes[1, 0].legend()

    # sin_phi histograms
    axes[1, 1].hist(new_df_X['sin_phi_pi_1m'], bins=50, alpha=0.5, label='sin_phi_pi_1m')
    axes[1, 1].hist(new_df_X['sin_phi_pi_2m'], bins=50, alpha=0.5, label='sin_phi_pi_2m')
    axes[1, 1].hist(new_df_X['sin_phi_pi_3m'], bins=50, alpha=0.5, label='sin_phi_pi_3m')
    axes[1, 1].hist(new_df_Y['sin_phi_neu'], bins=50, alpha=0.5, label='sin_phi_neu')
    axes[1, 1].set_title('sin_phi Histograms')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('data_analysis.png')

plot_histograms()

"""
# df_toUse['pi1_phi'] = df_toUse['pi1_phi'] % (2 * np.pi)
# df_toUse['pi2_phi'] = df_toUse['pi2_phi'] % (2 * np.pi)
# df_toUse['pi3_phi'] = df_toUse['pi3_phi'] % (2 * np.pi)
# df_toUse['neu_phi'] = df_toUse['neu_phi'] % (2 * np.pi)

# new_df_X['pt_pi_1m'] = df_toUse['pi1_pt']
# new_df_X['pt_pi_2m'] = df_toUse['pi2_pt']
# new_df_X['pt_pi_3m'] = df_toUse['pi3_pt']

# new_df_X['eta_pi_1m'] = df_toUse['pi1_eta']
# new_df_X['eta_pi_2m'] = df_toUse['pi2_eta']
# new_df_X['eta_pi_3m'] = df_toUse['pi3_eta']

# new_df_X['phi_pi_1m'] = df_toUse['pi1_phi']
# new_df_X['phi_pi_2m'] = df_toUse['pi2_phi']
# new_df_X['phi_pi_3m'] = df_toUse['pi3_phi']

# new_df_Y['pt_neu'] = df_toUse['neu_pt']
# new_df_Y['eta_neu'] = df_toUse['neu_eta']
# new_df_Y['phi_neu'] = df_toUse['neu_phi']

new_df_Y['sign'] = df_toUse['sign'].replace({'m': 0, 'p': 1})
new_df_Y['upsilon?'] = df_toUse['upsilon?'].replace({'n': 0, 'y': 1})




df_toUse_neutrino_X = new_df_X
df_toUse_neutrino_Y = new_df_Y

# print(df_toUse_neutrino_Y.keys())
# print(df_toUse_neutrino_X.keys())

X_train, X_test1, y_train, y_test1 = train_test_split(df_toUse_neutrino_X, df_toUse_neutrino_Y, test_size=0.2, shuffle=False)
X_test, X_val, y_test, y_val = train_test_split(X_test1, y_test1, test_size=0.5, shuffle=False)

scaler_train_X = StandardScaler()
scaler_val_X = StandardScaler()
scaler_test_X = StandardScaler()

scaler_train_y = StandardScaler()
scaler_val_y = StandardScaler()
scaler_test_y = StandardScaler()

features_to_scale = ['pt_neu', 'eta_neu', 'cos_phi_neu', 'sin_phi_neu']

X_train = scaler_train_X.fit_transform(X_train)
X_val = scaler_val_X.fit_transform(X_val)
X_test = scaler_test_X.fit_transform(X_test)

y_train_scaled = scaler_train_y.fit_transform(y_train[features_to_scale])
y_train = np.concatenate([y_train_scaled, y_train[['sign', 'upsilon?']]], axis=1)
y_val_scaled = scaler_val_y.fit_transform(y_val[features_to_scale])
y_val = np.concatenate([y_val_scaled, y_val[['sign', 'upsilon?']]], axis=1)
y_test_scaled = scaler_test_y.fit_transform(y_test[features_to_scale])
y_test = np.concatenate([y_test_scaled, y_test[['sign', 'upsilon?']]], axis=1)


X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# print(y_test[4,:])

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

def polar_to_cartesian(pt, eta, cos_phi, sin_phi):
    px = pt * cos_phi
    py = pt * sin_phi
    pz = pt * torch.sinh(eta)
    return px, py, pz

def custom_loss(pion_momenta, neutrino_gen, model_outputs):
    pi1_pt, pi1_eta, pi1_cos_phi, pi1_sin_phi = pion_momenta[:, 0], pion_momenta[:, 3], pion_momenta[:, 6], pion_momenta[:, 9]
    pi2_pt, pi2_eta, pi2_cos_phi, pi2_sin_phi = pion_momenta[:, 1], pion_momenta[:, 4], pion_momenta[:, 7], pion_momenta[:, 10]
    pi3_pt, pi3_eta, pi3_cos_phi, pi3_sin_phi = pion_momenta[:, 2], pion_momenta[:, 5], pion_momenta[:, 8], pion_momenta[:, 11]

    neu_gen_pt, neu_gen_eta, neu_gen_cos_phi, neu_gen_sin_phi = neutrino_gen[:, 0], neutrino_gen[:, 1], neutrino_gen[:, 2], neutrino_gen[:, 3]
    neu_pred_pt, neu_pred_eta, neu_pred_cos_phi, neu_pred_sin_phi = model_outputs[:, 0], model_outputs[:, 1], model_outputs[:, 2], model_outputs[:, 3]

    pi1_px, pi1_py, pi1_pz = polar_to_cartesian(pi1_pt, pi1_eta, pi1_cos_phi, pi1_sin_phi)
    pi2_px, pi2_py, pi2_pz = polar_to_cartesian(pi2_pt, pi2_eta, pi2_cos_phi, pi2_sin_phi)
    pi3_px, pi3_py, pi3_pz = polar_to_cartesian(pi3_pt, pi3_eta, pi3_cos_phi, pi3_sin_phi)

    neu_gen_px, neu_gen_py, neu_gen_pz = polar_to_cartesian(neu_gen_pt, neu_gen_eta, neu_gen_cos_phi, neu_gen_sin_phi)
    neu_pred_px, neu_pred_py, neu_pred_pz = polar_to_cartesian(neu_pred_pt, neu_pred_eta, neu_pred_cos_phi, neu_pred_sin_phi)

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

    tau_gen_mass = torch.sqrt(tau_gen_E**2 - (tau_gen_px**2 + tau_gen_py**2 + tau_gen_pz**2)) / 1.776
    tau_pred_mass = torch.sqrt(tau_pred_E**2 - (tau_pred_px**2 + tau_pred_py**2 + tau_pred_pz**2)) / 1.776

    loss = torch.mean((tau_gen_mass - tau_pred_mass)**2)
    return loss


def compute_weighted_mse_loss(neutrino_gen, model_outputs):
    # Initialize the scalers
    loss_scaler_predictions = StandardScaler()
    loss_scaler_gen_info = StandardScaler()

    # Detach the tensors from the computation graph and convert to numpy
    neutrino_gen_np = neutrino_gen.detach().cpu().numpy()
    model_outputs_np = model_outputs.detach().cpu().numpy()

    # Fit and transform the data
    neutrino_gen_scaled = loss_scaler_gen_info.fit_transform(neutrino_gen_np)
    model_outputs_scaled = loss_scaler_predictions.fit_transform(model_outputs_np)

    # Convert the scaled data back to tensors
    neutrino_gen_scaled_tensor = torch.tensor(neutrino_gen_scaled, dtype=torch.float32, device=neutrino_gen.device)
    model_outputs_scaled_tensor = torch.tensor(model_outputs_scaled, dtype=torch.float32, device=model_outputs.device)

    # Enable gradient tracking on the scaled tensors
    neutrino_gen_scaled_tensor.requires_grad_(True)
    model_outputs_scaled_tensor.requires_grad_(True)

    # Compute the MSE loss
    mse_loss = nn.MSELoss()(neutrino_gen_scaled_tensor, model_outputs_scaled_tensor)

    return mse_loss

criterion1 = custom_loss
criterion2 = nn.MSELoss()
criterion3 = compute_weighted_mse_loss
model = SimpleLinearNN2(12, 4)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
num_epochs = 170

training_losses = []
val_losses = []

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for batch_X, batch_y in train_loader:
#         optimizer.zero_grad()
#         outputs = model(batch_X)
#         loss = criterion2(batch_y[:, :4], outputs) #+ 0.5*criterion1(batch_X, batch_y, outputs)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()

#     training_loss = running_loss / len(train_loader)
#     training_losses.append(training_loss)

#     val_running_loss = 0.0
#     model.eval()
#     with torch.no_grad():
#         for batch_X_val, batch_y_val in val_loader:
#             outputs_val = model(batch_X_val)
#             val_loss = criterion2(batch_y_val[:,:4], outputs_val) #+ 0.5*criterion1(batch_X_val, batch_y_val, outputs_val)
#             val_running_loss += val_loss.item()

#     val_loss = val_running_loss / len(val_loader)
#     val_losses.append(val_loss)

#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {training_loss:.4f}, Validation Loss: {val_loss:.4f}')

# plt.figure(figsize=(10, 5))
# plt.plot(training_losses, label='Training Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.savefig('hdm_training_vs_validation.png')

# torch.save(model.state_dict(), 'hdm_trained_model.pth')

# torch.save(X_test, 'testing_set.pt')
# torch.save(y_test, 'testing_answers.pt')



##################################################################################################

model.load_state_dict(torch.load('hdm_trained_model.pth'))


# make a list of sign 

model.load_state_dict(torch.load('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/model/hdm_trained_model.pth'))
X_Test = torch.load('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/model/testing_set.pt')
Y_Test = torch.load('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/model/testing_answers.pt')

print('loaded stuff in')

net_m_loss = 0.0
m_amount = 0
net_p_loss = 0.0
p_amount = 0

column_names = ['pi1_pt', 'pi1_eta', 'pi1_cosphi', 'pi1_sinphi',
                'pi2_pt', 'pi2_eta', 'pi2_cosphi', 'pi2_sinphi',
                'pi3_pt', 'pi3_eta', 'pi3_cosphi', 'pi3_sinphi',
                'neu1_pt', 'neu1_eta', 'neu1_cosphi', 'neu1_sinphi',
                'pi4_pt', 'pi4_eta', 'pi4_cosphi', 'pi4_sinphi',
                'pi5_pt', 'pi5_eta', 'pi5_cosphi', 'pi5_sinphi',
                'pi6_pt', 'pi6_eta', 'pi6_cosphi', 'pi6_sinphi',
                'neu2_pt', 'neu2_eta', 'neu2_cosphi', 'neu2_sinphi']

tau_column_names = ['pi1_pt', 'pi1_eta', 'pi1_cosphi', 'pi1_sinphi',
                'pi2_pt', 'pi2_eta', 'pi2_cosphi', 'pi2_sinphi',
                'pi3_pt', 'pi3_eta', 'pi3_cosphi', 'pi3_sinphi',
                'neu_pt', 'neu_eta', 'neu_cosphi', 'neu_sinphi']

df = pd.DataFrame(columns = tau_column_names)
df_taup = pd.DataFrame(columns = tau_column_names)
df_taum = pd.DataFrame(columns=tau_column_names)

df_fifteen = pd.read_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/all_15GeV_reco_data_boosted.csv')
df_fifteen_X = pd.DataFrame()
df_fifteen_Y = pd.DataFrame()

df_fifteen_X['pt_pi_1m'] = df_fifteen['pi1_pt']
df_fifteen_X['pt_pi_2m'] = df_fifteen['pi2_pt']
df_fifteen_X['pt_pi_3m'] = df_fifteen['pi3_pt']

df_fifteen_X['eta_pi_1m'] = df_fifteen['pi1_eta']
df_fifteen_X['eta_pi_2m'] = df_fifteen['pi2_eta']
df_fifteen_X['eta_pi_3m'] = df_fifteen['pi3_eta']

df_fifteen_X['cos_phi_pi_1m'] = np.cos(df_fifteen['pi1_phi'])
df_fifteen_X['cos_phi_pi_2m'] = np.cos(df_fifteen['pi2_phi'])
df_fifteen_X['cos_phi_pi_3m'] = np.cos(df_fifteen['pi3_phi'])

df_fifteen_X['sin_phi_pi_1m'] = np.sin(df_fifteen['pi1_phi'])
df_fifteen_X['sin_phi_pi_2m'] = np.sin(df_fifteen['pi2_phi'])
df_fifteen_X['sin_phi_pi_3m'] = np.sin(df_fifteen['pi3_phi'])

df_fifteen_Y['pt_neu'] = df_fifteen['neu_pt']
df_fifteen_Y['eta_neu'] = df_fifteen['neu_eta']
df_fifteen_Y['cos_phi_neu'] = np.cos(df_fifteen['neu_phi'])
df_fifteen_Y['sin_phi_neu'] = np.sin(df_fifteen['neu_phi'])


df_fifteen_Y['sign'] = df_fifteen['sign'].replace({'m': 0, 'p': 1})
df_fifteen_Y['upsilon?'] = df_fifteen['upsilon?'].replace({'n': 0, 'y': 1})

# print(df_fifteen_X.keys())
# print(df_fifteen_Y.keys())


scaler_fifteen_X = StandardScaler()

scaler_fifteen_y = StandardScaler()

features_to_scale = ['pt_neu', 'eta_neu', 'cos_phi_neu', 'sin_phi_neu']

X_fifteen_train, X_fifteen_test, y_fifteen_train, y_fifteen_test1 = train_test_split(df_fifteen_X, df_fifteen_Y, test_size=0.99, shuffle=False)

X_fifteen_test = scaler_fifteen_X.fit_transform(X_fifteen_test)
y_fifteen_test = scaler_fifteen_y.fit_transform(y_fifteen_test1[features_to_scale])
y_fifteen_test = np.concatenate([y_fifteen_test, y_fifteen_test1[['sign', 'upsilon?']]], axis=1)


X_fifteen_test = torch.tensor(X_fifteen_test, dtype=torch.float32)
y_fifteen_test = torch.tensor(y_fifteen_test, dtype=torch.float32)

print('beginning eval')
model.eval()
with torch.no_grad():
    outputs = model(X_fifteen_test)
    test_loss = criterion2(outputs, y_fifteen_test[:,:4])
    print(f'Test Loss: {test_loss.item():.4f}')
    X_fifteen_test = scaler_fifteen_X.inverse_transform(X_fifteen_test)
    outputs = scaler_fifteen_y.inverse_transform(outputs)
    for i in range(int(outputs.size/4)):
        if y_fifteen_test[i,4].item() == 1 and y_fifteen_test[i,5].item() == 1:
                add_row_1 = [X_fifteen_test[i,0].item(), X_fifteen_test[i,3].item(), X_fifteen_test[i,6].item(), X_fifteen_test[i,9].item(),
                X_fifteen_test[i,1].item(), X_fifteen_test[i,4].item(), X_fifteen_test[i,7].item(), X_fifteen_test[i,10].item(),
                X_fifteen_test[i,2].item(), X_fifteen_test[i,5].item(), X_fifteen_test[i,8].item(), X_fifteen_test[i,11].item(),
                outputs[i,0].item(), outputs[i,1].item(), outputs[i,2].item(), outputs[i,3].item()]
                new_row_1 = pd.DataFrame([add_row_1], columns = df.columns)
                df = pd.concat([df, new_row_1], ignore_index=True)
                add_row_2 = [X_fifteen_test[i+1,0].item(), X_fifteen_test[i+1,3].item(), X_fifteen_test[i+1,6].item(), X_fifteen_test[i+1,9].item(),
                X_fifteen_test[i+1,1].item(), X_fifteen_test[i+1,4].item(), X_fifteen_test[i+1,7].item(), X_fifteen_test[i+1,10].item(),
                X_fifteen_test[i+1,2].item(), X_fifteen_test[i+1,5].item(), X_fifteen_test[i+1,8].item(), X_fifteen_test[i+1,11].item(),
                outputs[i+1,0].item(), outputs[i+1,1].item(), outputs[i+1,2].item(), outputs[i+1,3].item()]
                new_row_2 = pd.DataFrame([add_row_2], columns = df.columns)
                df = pd.concat([df, new_row_2], ignore_index=True)
        
    #     add_row = [X_fifteen_test[i,0].item(), X_fifteen_test[i,3].item(), X_fifteen_test[i,6].item(), X_fifteen_test[i,9].item(),
    #             X_fifteen_test[i,1].item(), X_fifteen_test[i,4].item(), X_fifteen_test[i,7].item(), X_fifteen_test[i,10].item(),
    #             X_fifteen_test[i,2].item(), X_fifteen_test[i,5].item(), X_fifteen_test[i,8].item(), X_fifteen_test[i,11].item(),
    #             outputs[i,0].item(), outputs[i,1].item(), outputs[i,2].item(), outputs[i,3].item()]
    #     new_row = pd.DataFrame([add_row], columns=df_taum.columns)
    #     df_taum = pd.concat([df_taum, new_row], ignore_index=True)



    # # print(X_test)
    # outputs = model(X_Test)
    # X_Test = scaler_test_X.inverse_transform(X_Test)
    # outputs = scaler_test_y.inverse_transform(outputs)
    # print(outputs.size)
    # for i in range(int(outputs.size/4)):
    #     # print(Y_Test[i,5].item())
    #     if Y_Test[i,4].item() == 0:
    #        add_row = [X_Test[i,0].item(), X_Test[i,3].item(), X_Test[i,6].item(), X_Test[i,9].item(),
    #             X_Test[i,1].item(), X_Test[i,4].item(), X_Test[i,7].item(), X_Test[i,10].item(),
    #             X_Test[i,2].item(), X_Test[i,5].item(), X_Test[i,8].item(), X_Test[i,11].item(),
    #             outputs[i,0].item(), outputs[i,1].item(), outputs[i,2].item(), outputs[i,3].item()]
    #        new_row = pd.DataFrame([add_row], columns=df_taum.columns)
    #        df_taum = pd.concat([df_taum, new_row], ignore_index=True)
    #     if Y_Test[i,4].item() == 1:
    #         add_row = [X_Test[i,0].item(), X_Test[i,3].item(), X_Test[i,6].item(), X_Test[i,9].item(),
    #             X_Test[i,1].item(), X_Test[i,4].item(), X_Test[i,7].item(), X_Test[i,10].item(),
    #             X_Test[i,2].item(), X_Test[i,5].item(), X_Test[i,8].item(), X_Test[i,11].item(),
    #             outputs[i,0].item(), outputs[i,1].item(), outputs[i,2].item(), outputs[i,3].item()]
    #         new_row = pd.DataFrame([add_row], columns=df_taup.columns)
    #         df_taup = pd.concat([df_taup, new_row], ignore_index=True)
    #         if Y_Test[i,5].item() == 1:
    #             # print(Y_Test[i+1,5].item())
    #             if Y_Test[i+1,4].item() != 0: raise RuntimeError('Our whole idea is wrong')
    #             add_row_1 = [X_Test[i,0].item(), X_Test[i,3].item(), X_Test[i,6].item(), X_Test[i,9].item(),
    #             X_Test[i,1].item(), X_Test[i,4].item(), X_Test[i,7].item(), X_Test[i,10].item(),
    #             X_Test[i,2].item(), X_Test[i,5].item(), X_Test[i,8].item(), X_Test[i,11].item(),
    #             outputs[i,0].item(), outputs[i,1].item(), outputs[i,2].item(), outputs[i,3].item()]
    #             new_row_1 = pd.DataFrame([add_row_1], columns = df.columns)
    #             df = pd.concat([df, new_row_1], ignore_index=True)
    #             add_row_2 = [X_Test[i+1,0].item(), X_Test[i+1,3].item(), X_Test[i+1,6].item(), X_Test[i+1,9].item(),
    #             X_Test[i+1,1].item(), X_Test[i+1,4].item(), X_Test[i+1,7].item(), X_Test[i+1,10].item(),
    #             X_Test[i+1,2].item(), X_Test[i+1,5].item(), X_Test[i+1,8].item(), X_Test[i+1,11].item(),
    #             outputs[i+1,0].item(), outputs[i+1,1].item(), outputs[i+1,2].item(), outputs[i+1,3].item()]
    #             new_row_2 = pd.DataFrame([add_row_2], columns = df.columns)
    #             df = pd.concat([df, new_row_2], ignore_index=True)

print('saving csvs...')
# print(df)
df.to_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/15GeV_upsilons.csv', index=False)
# df_taup.to_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/taup.csv', index=False)
# df_taum.to_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/15GeV_all_taus.csv', index=False)
print('done.')


# pt_loss = abs(y_test[:3,0] - test_outputs[:,0])
# eta_loss = abs(y_test[:3,1] - test_outputs[:,1])
# cos_phi_loss = abs(y_test[:3,2] - test_outputs[:,2])
# sin_phi_loss = abs(y_test[:3,3] - test_outputs[:,3])

# print(pt_loss.mean())
# print(eta_loss.mean())
# print(cos_phi_loss.mean())
# print(sin_phi_loss.mean())



# test_outputs = scaler_test_y.inverse_transform(test_outputs)
# y_test = scaler_test_y.inverse_transform(y_test)

# pt_loss = abs(y_test[:,0] - test_outputs[:,0])
# eta_loss = abs(y_test[:,1] - test_outputs[:,1])
# cos_phi_loss = abs(y_test[:,2] - test_outputs[:,2])
# sin_phi_loss = abs(y_test[:,3] - test_outputs[:,3])

# print(pt_loss.mean())
# print(eta_loss.mean())
# print(cos_phi_loss.mean())
# print(sin_phi_loss.mean())

# max_pt_test = max(y_test[:,0])
# max_eta_test = max(y_test[:,1])
# max_cos_test = max(y_test[:,2])
# max_sin_test = max(y_test[:,3]) 

# max_pt_pred = max(test_outputs[:,0])
# max_eta_pred = max(test_outputs[:,1])
# max_cos_pred = max(test_outputs[:,2])
# max_sin_pred = max(test_outputs[:,3])

# scaled_test_pt = y_test[:,0] / max_pt_test
# scaled_test_eta = y_test[:,1] / max_eta_test
# scaled_test_cos = y_test[:,2] / max_cos_test
# scaled_test_sin = y_test[:,3] / max_sin_test

# scaled_pred_pt = test_outputs[:,0] / max_pt_pred
# scaled_pred_eta = test_outputs[:,1] / max_eta_pred
# scaled_pred_cos = test_outputs[:,2] / max_cos_pred
# scaled_pred_sin = test_outputs[:,3] / max_sin_pred


plt.figure(figsize=(5, 5))
plt.scatter(y_fifteen_test[:, 0], outputs[:, 0])
plt.title(f'Prediction of Pt')
plt.xlabel('Validation Set Pt')
plt.ylabel('Prediction of Pt from Model')
plt.savefig('norm_pt_proper_boosted_pt_plot_sph.png')
plt.figure(figsize=(5, 5))
plt.scatter(y_fifteen_test[:, 1], outputs[:, 1])
plt.title(f'Prediction of eta')
plt.xlabel('Validation Set eta')
plt.ylabel('Prediction of eta from Model')
plt.savefig('norm_pt_proper_boosted_eta_plot_sph.png')
plt.figure(figsize=(5, 5))
plt.scatter(y_fifteen_test[:, 3], outputs[:, 3])
plt.title(f'Prediction of sin(phi)')
plt.xlabel('Validation Set sin(phi)')
plt.ylabel('Prediction of sin(phi) from Model')
plt.savefig('norm_pt_proper_boosted_sin_phi_plot_sph.png')
plt.figure(figsize=(5, 5))
plt.scatter(y_fifteen_test[:, 2], outputs[:, 2])
plt.title(f'Prediction of cos(phi)')
plt.xlabel('Validation Set cos(phi)')
plt.ylabel('Prediction of cos(phi) from Model')
plt.savefig('norm_pt_proper_boosted_phi_plot_sph.png')

# plt.figure(figsize=(5, 5))
# plt.scatter(scaled_test_pt, scaled_pred_pt)
# plt.title(f'Prediction of Pt (scaled)')
# plt.xlabel('Validation Set Pt')
# plt.ylabel('Prediction of Pt from Model')
# plt.savefig('norm_pt_scaled_proper_boosted_pt_plot_sph.png')
# plt.figure(figsize=(5, 5))
# plt.scatter(scaled_test_eta, scaled_pred_eta)
# plt.title(f'Prediction of eta (scaled)')
# plt.xlabel('Validation Set eta')
# plt.ylabel('Prediction of eta from Model')
# plt.savefig('norm_pt_scaled_proper_boosted_eta_plot_sph.png')
# plt.figure(figsize=(5, 5))
# plt.scatter(scaled_test_sin, scaled_pred_sin)
# plt.title(f'Prediction of sin(phi) (scaled)')
# plt.xlabel('Validation Set sin(phi)')
# plt.ylabel('Prediction of sin(phi) from Model')
# plt.savefig('norm_pt_scaled_proper_boosted_sin_phi_plot_sph.png')
# plt.figure(figsize=(5, 5))
# plt.scatter(scaled_test_cos, scaled_pred_cos)
# plt.title(f'Prediction of cos(phi) (scaled)')
# plt.xlabel('Validation Set cos(phi)')
# plt.ylabel('Prediction of cos(phi) from Model')
# plt.savefig('norm_pt_scaled_proper_boosted_cos_phi_plot_sph.png')
# outputs_array = test_outputs
# y_test_array = y_test
# X_test_array = scaler_test_X.inverse_transform(X_test)

# print(outputs_array)
# print(y_test_array)

# # Step 3: Convert the NumPy array to a Pandas DataFrame
# #df_x_test = pd.DataFrame(X_test_array, columns=['pion1_pT','pion2_pT','pion3_pT','pion1_eta','pion2_eta','pion3_eta','pion1_cos_phi','pion2_cos_phi','pion3_cos_phi','pion1_sin_phi','pion2_sin_phi','pion3_sin_phi'])
# #df_y_prediction = pd.DataFrame(outputs_array, columns=['neutrino_pT','neutrino_eta','neutrino_cos_phi','neutrino_sin_phi'])

# df_x_test = pd.DataFrame(X_test_array, columns=['pion1_pT','pion2_pT','pion3_pT','pion1_eta','pion2_eta','pion3_eta','pion1_cos_phi','pion2_cos_phi','pion3_cos_phi','pion1_sin_phi','pion2_sin_phi','pion3_sin_phi'])
# df_y_prediction = pd.DataFrame(outputs_array, columns=['neutrino_pT','neutrino_eta','neutrino_cos_phi', 'neutrino_sin_phi'])
# df_y_test = pd.DataFrame(y_test_array, columns=['neutrino_pT_test','neutrino_eta_test','neutrino_cos_phi_test', 'neutrino_sin_phi_test'])

# df_tau_mass = pd.concat([df_x_test, df_y_prediction, df_y_test], axis = 1)
# df_tau_mass.to_csv('norm_pt_proper_boosted_reco_tau_mass_for_histograms.csv')
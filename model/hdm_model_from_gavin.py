import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from fast_histogram import histogram1d
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#import matplotlib.pyplot as plt
split_tau_columns = ['pi1_from_tau_pt', 'pi1_from_tau_eta', 'pi1_from_tau_phi','pi2_from_tau_pt', 'pi2_from_tau_eta', 'pi2_from_tau_phi', 'pi3_from_tau_pt', 'pi3_from_tau_eta', 'pi3_from_tau_phi',
                                        'neutrino_from_tau_pt','neutrino_from_tau_eta','neutrino_from_tau_phi', 'tau_no_neutrino_mass', 'tau_with_neutrino_mass']
split_antitau_columns = ['pi1_from_antitau_pt', 'pi1_from_antitau_eta', 'pi1_from_antitau_phi','pi2_from_antitau_pt', 'pi2_from_antitau_eta', 'pi2_from_antitau_phi', 'pi3_from_antitau_pt', 'pi3_from_antitau_eta', 'pi3_from_antitau_phi',
                                        'neutrino_from_antitau_pt','neutrino_from_antitau_eta','neutrino_from_antitau_phi', 'tau_no_neutrino_mass', 'tau_with_neutrino_mass']
column_names_both = ['pi1_from_tau_pt', 'pi1_from_tau_eta', 'pi1_from_tau_phi','pi2_from_tau_pt', 'pi2_from_tau_eta', 'pi2_from_tau_phi', 'pi3_from_tau_pt', 'pi3_from_tau_eta', 'pi3_from_tau_phi',
                                        'pi1_from_antitau_pt', 'pi1_from_antitau_eta', 'pi1_from_antitau_phi', 'pi2_from_antitau_pt', 'pi2_from_antitau_eta', 'pi2_from_antitau_phi','pi3_from_antitau_pt', 'pi3_from_antitau_eta', 'pi3_from_antitau_phi',
                                        'neutrino_from_tau_pt','neutrino_from_tau_eta','neutrino_from_tau_phi','neutrino_from_antitau_pt','neutrino_from_antitau_eta','neutrino_from_antitau_phi', 'tau_no_neutrino_mass', 'tau_with_neutrino_mass',
                                        'antitau_no_neutrino_mass', 'antitau_with_neutrino_mass', 'upsilon_no_neutrino_mass', 'upsilon_with_neutrino_mass']
# df_tau_total = pd.DataFrame(columns = split_tau_columns)
# df_antitau_total = pd.DataFrame(columns = split_antitau_columns)
# for i in range(30):
#     if i != 2 and i != 6:
#         df1 = pd.read_csv(f'/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing/individual_datasets_15GeV/both_proper_decay_info{i}.csv')
#         df2 = pd.read_csv(f'/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing/individual_datasets_15GeV/tau_proper_decay_info{i}.csv')
#         df3 = pd.read_csv(f'/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing/individual_datasets_15GeV/anti_proper_decay_info{i}.csv')
#         df_tau_split = df1[split_tau_columns]
#         df_antitau_split = df1[split_antitau_columns]
#         df_tau = pd.concat([df2, df_tau_split], ignore_index=True)
#         df_antitau = pd.concat([df3, df_antitau_split], ignore_index=True)
#         df_tau_total = pd.concat([df_tau_total, df_tau], ignore_index=True)
#         df_antitau_total = pd.concat([df_antitau_total, df_antitau], ignore_index=True)
# for i in range(10):
#     df1 = pd.read_csv(f'/isilon/export/home/gpitt3/tau-decay-ml/preprocessing/both_proper_decay_info{i}.csv')
#     df2 = pd.read_csv(f'/isilon/export/home/gpitt3/tau-decay-ml/preprocessing/tau_proper_decay_info{i}.csv')
#     df3 = pd.read_csv(f'/isilon/export/home/gpitt3/tau-decay-ml/preprocessing/anti_proper_decay_info{i}.csv')
#     df_tau_split = df1[split_tau_columns]
#     df_antitau_split = df1[split_antitau_columns]
#     df_tau = pd.concat([df2, df_tau_split], ignore_index=True)
#     df_antitau = pd.concat([df3, df_antitau_split], ignore_index=True)
#     df_tau_total = pd.concat([df_tau_total, df_tau], ignore_index=True)
#     df_antitau_total = pd.concat([df_antitau_total, df_antitau], ignore_index=True)
df_tau_total = df_tau_total.drop_duplicates(['pi1_from_tau_pt'])
df_antitau_total = df_antitau_total.drop_duplicates(['pi1_from_antitau_pt'])
# print(df_tau_total)
# print(df_antitau_total)
df_antitau_total.columns = df_tau_total.columns
df_tau_total = pd.concat([df_tau_total, df_antitau_total], ignore_index=True)
#print(df_tau_total)
bins = 80  # Number of bins
range_min = 0
range_max = 3
"""
hist1 = histogram1d(df_tau_total['tau_with_neutrino_mass'], bins=bins, range=[range_min, range_max])
hist2 = histogram1d(df_tau_total['tau_no_neutrino_mass'], bins=bins, range=[range_min, range_max])
bin_edges = np.linspace(range_min, range_max, bins + 1)
# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], hist1, width=np.diff(bin_edges), edgecolor='blue')
plt.title('Mass of Reconstructed Tau (with neutrino)')
plt.xlabel('Invariant Mass (GeV)')
plt.ylabel('Frequency')
plt.savefig('/isilon/export/home/gpitt3/tau-decay-ml/hist_tau_mass_with_neutrino.png')
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], hist2, width=np.diff(bin_edges), edgecolor='blue')
plt.title('Mass of Reconstructed Tau (without neutrino)')
plt.xlabel('Invariant Mass (GeV)')
plt.ylabel('Frequency')
plt.savefig('/isilon/export/home/gpitt3/tau-decay-ml/hist_tau_mass_no_neutrino.png')
"""
df_toUse = df_tau_total
new_df_X = pd.DataFrame()
new_df_Y = pd.DataFrame()
new_df_X['px_pi_1m'] = df_toUse['pi1_from_tau_pt']*np.cos(df_toUse['pi1_from_tau_phi'])
new_df_X['px_pi_2m'] = df_toUse['pi2_from_tau_pt']*np.cos(df_toUse['pi2_from_tau_phi'])
new_df_X['px_pi_3m'] = df_toUse['pi3_from_tau_pt']*np.cos(df_toUse['pi3_from_tau_phi'])
new_df_X['py_pi_1m'] = df_toUse['pi1_from_tau_pt']*np.sin(df_toUse['pi1_from_tau_phi'])
new_df_X['py_pi_2m'] = df_toUse['pi2_from_tau_pt']*np.sin(df_toUse['pi2_from_tau_phi'])
new_df_X['py_pi_3m'] = df_toUse['pi3_from_tau_pt']*np.sin(df_toUse['pi3_from_tau_phi'])
new_df_X['pz_pi_1m'] = df_toUse['pi1_from_tau_pt']*np.sinh(df_toUse['pi1_from_tau_eta'])
new_df_X['pz_pi_2m'] = df_toUse['pi2_from_tau_pt']*np.sinh(df_toUse['pi2_from_tau_eta'])
new_df_X['pz_pi_3m'] = df_toUse['pi3_from_tau_pt']*np.sinh(df_toUse['pi3_from_tau_eta'])
new_df_Y['px_neu'] = df_toUse['neutrino_from_tau_pt']*np.cos(df_toUse['neutrino_from_tau_phi'])
new_df_Y['py_neu'] = df_toUse['neutrino_from_tau_pt']*np.sin(df_toUse['neutrino_from_tau_phi'])
new_df_Y['pz_neu'] = df_toUse['neutrino_from_tau_pt']*np.sinh(df_toUse['neutrino_from_tau_eta'])
"""
new_df_X['px_pi_1p'] = df_toUse['pi1_from_antitau_pt']*np.cos(df_toUse['pi1_from_antitau_phi'])
new_df_X['px_pi_2p'] = df_toUse['pi2_from_antitau_pt']*np.cos(df_toUse['pi2_from_antitau_phi'])
new_df_X['px_pi_3p'] = df_toUse['pi3_from_antitau_pt']*np.cos(df_toUse['pi3_from_antitau_phi'])
new_df_X['py_pi_1p'] = df_toUse['pi1_from_antitau_pt']*np.sin(df_toUse['pi1_from_antitau_phi'])
new_df_X['py_pi_2p'] = df_toUse['pi2_from_antitau_pt']*np.sin(df_toUse['pi2_from_antitau_phi'])
new_df_X['py_pi_3p'] = df_toUse['pi3_from_antitau_pt']*np.sin(df_toUse['pi3_from_antitau_phi'])
new_df_X['pz_pi_1p'] = df_toUse['pi1_from_antitau_pt']*np.sinh(df_toUse['pi1_from_antitau_eta'])
new_df_X['pz_pi_2p'] = df_toUse['pi2_from_antitau_pt']*np.sinh(df_toUse['pi2_from_antitau_eta'])
new_df_X['pz_pi_3p'] = df_toUse['pi3_from_antitau_pt']*np.sinh(df_toUse['pi3_from_antiau_eta'])
new_df_Y['px_antineu'] = df_toUse['antineutrino_from_tau_pt']*np.cos(df_toUse['antineutrino_from_tau_phi'])
new_df_Y['py_antineu'] = df_toUse['antineutrino_from_tau_pt']*np.sin(df_toUse['antineutrino_from_tau_phi'])
new_df_Y['pz_antineu'] = df_toUse['antineutrino_from_tau_pt']*np.sinh(df_toUse['antineutrino_from_tau_eta'])
"""
df_toUse_neutrino_X = new_df_X
df_toUse_neutrino_Y = new_df_Y
mean_x = df_toUse_neutrino_X.mean()
std_x = df_toUse_neutrino_X.std()
mean_y = df_toUse_neutrino_Y.mean()
std_y = df_toUse_neutrino_Y.std()
#print(mean_x)
#print(std_x)
#print(mean_y)
#print(std_y)
X_train, X_test, y_train, y_test = train_test_split(df_toUse_neutrino_X, df_toUse_neutrino_Y, test_size=0.2, random_state=42)
"""
scaler_X_train = StandardScaler()
scaler_X_test = StandardScaler()
scaler_y_train = StandardScaler()
scaler_y_test = StandardScaler()
X_train = scaler_X_train.fit_transform(X_train)
X_test = scaler_X_test.fit_transform(X_test)
y_train = scaler_y_train.fit_transform(y_train)
y_test = scaler_y_test.fit_transform(y_test)
"""
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
batch_size = 90
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
torch.set_num_threads(2)
class SimpleLinearNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLinearNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 25)
        self.layer2 = nn.Linear(25, 40)
        self.layer3 = nn.Linear(40, 100)
        self.layer4 = nn.Linear(100,40)
        self.layer5 = nn.Linear(40,25)
        self.layer6 = nn.Linear(25, 16)
        self.layer7 = nn.Linear(16, 8)
        self.layer8 = nn.Linear(8, 6)
        self.layer9 = nn.Linear(6, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, x):
        x = self.relu(self.layer1(x))
        #x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.dropout(x)
        x = self.relu(self.layer4(x))
        x = self.dropout(x)
        x = self.relu(self.layer5(x))
        #x = self.layer4(x)
        x = self.relu(self.layer6(x))
        x = self.relu(self.layer7(x))
        x = self.relu(self.layer8(x))
        x = self.relu(self.layer9(x))
        #x = self.layer4(x)
        return x
criterion = nn.MSELoss()
model = SimpleLinearNN(9, 3)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
num_epochs = 400
losses = []
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    losses.append(loss.item())
torch.save(model.state_dict(), 'our_data_model_norm.pth')
# Step 5: Evaluate the model
# i = 0
# for parameter in model.parameters():
#     i=i+1
# print(i)
model.load_state_dict(torch.load('our_data_model_norm.pth'))
criterion = nn.MSELoss()
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')
#test_outputs = (test_outputs * y_test_stds) + y_test_means
#y_test = (y_test * y_test_stds) + y_test_means
"""
test_outputs = scaler_y_test.inverse_transform(test_outputs)
y_test = scaler_y_test.inverse_transform(y_test)
"""
# print(test_outputs)
total_params = sum(p.numel() for p in model.parameters())
print("Total Parameters:")
print(total_params)
# Create a scatter plot for the specified column
# plt.figure(figsize=(5, 5))
# plt.scatter(y_test[:, 0], test_outputs[:, 0])
# plt.title(f'Prediction of Px')
# plt.xlabel('Validation Set Px')
# plt.ylabel('Prediction of Px from Model')
# plt.savefig('px_plot.png')
# plt.figure(figsize=(5, 5))
# plt.scatter(y_test[:, 1], test_outputs[:, 1])
# plt.title(f'Prediction of Py')
# plt.xlabel('Validation Set Py')
# plt.ylabel('Prediction of Py from Model')
# plt.savefig('py_plot.png')
# plt.figure(figsize=(5, 5))
# plt.scatter(y_test[:, 2], test_outputs[:, 2])
# plt.title(f'Prediction of Pz')
# plt.xlabel('Validation Set Pz')
# plt.ylabel('Prediction of Pz from Model')
# plt.savefig('pz_plot.png')
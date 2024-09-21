import pickle
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

df = pd.read_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/reco_alldata.csv')
#df_gen = pd.read_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/gen_alldata.csv')
# print(df_gen.shape)
print(df.shape)

torch.set_num_threads(2)

df['pi1_px'] = df['pi1_pt']*np.cos(df['pi1_phi'])
df['pi2_px'] = df['pi2_pt']*np.cos(df['pi2_phi'])
df['pi3_px'] = df['pi3_pt']*np.cos(df['pi3_phi'])
df['pi1_py'] = df['pi1_pt']*np.sin(df['pi1_phi'])
df['pi2_py'] = df['pi2_pt']*np.sin(df['pi2_phi'])
df['pi3_py'] = df['pi3_pt']*np.sin(df['pi3_phi'])
df['pi1_pz'] = df['pi1_pt']*np.sinh(df['pi1_eta'])
df['pi2_pz'] = df['pi2_pt']*np.sinh(df['pi2_eta'])
df['pi3_pz'] = df['pi3_pt']*np.sinh(df['pi3_eta'])
df['neu_px'] = df['neu_pt']*np.cos(df['neu_phi'])
df['neu_py'] = df['neu_pt']*np.sin(df['neu_phi'])
df['neu_pz'] = df['neu_pt']*np.sinh(df['neu_eta'])

X = df[['pi1_px', 'pi1_py', 'pi1_pz',
        'pi2_px', 'pi2_py', 'pi2_pz',
        'pi3_px', 'pi3_py', 'pi3_pz']].values
Y = df[['neu_px', 'neu_py', 'neu_pz']].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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

train_loader = DataLoader(train_dataset, batch_size=752, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=9433, shuffle=False)

class SimpleDNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        # nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        self.fc2 = nn.Linear(64, 256)
        # nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        self.fc3 = nn.Linear(256, 2560)
        # # nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        # self.fc99 = nn.Linear(256, 624)
        # # nn.init.kaiming_normal_(self.fc99.weight, mode='fan_in', nonlinearity='relu')
        # self.fc100 = nn.Linear(624, 428)
        # # nn.init.kaiming_normal_(self.fc100.weight, mode='fan_in', nonlinearity='relu')
        # self.fc101 = nn.Linear(428,256)
        # # nn.init.kaiming_normal_(self.fc101.weight, mode='fan_in', nonlinearity='relu')
        self.fc4 = nn.Linear(2560, 1028)
        # nn.init.kaiming_normal_(self.fc4.weight, mode='fan_in', nonlinearity='relu')
        self.fc5 = nn.Linear(1028, 512)
        # nn.init.kaiming_normal_(self.fc5.weight, mode='fan_in', nonlinearity='relu')
        self.fc6 = nn.Linear(512, 128)
        # nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        self.fc7 = nn.Linear(128, 64)
        # nn.init.kaiming_normal_(self.fc7.weight, mode='fan_in', nonlinearity='relu')
        self.fc8 = nn.Linear(64, 36)
        # nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        # self.fc9 = nn.Linear(32, 24)
        # nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        # #self.fc10 = nn.Linear(16, 12)
        # nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        # self.fc11 = nn.Linear(24, 16)
        # nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        # self.fc12 = nn.Linear(16, 8)
        # nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        self.fc13 = nn.Linear(36, output_dim)
        #nn.init.kaiming_normal_(self.fc13.weight, mode='fan_in', nonlinearity='relu')
        # self.dropout_a = nn.Dropout(p=0.5)
        # self.dropout_b = nn.Dropout(p=0.2)
        # self.dropout_c = nn.Dropout(p=0.1)
        # she had 12 layers, up to 2560 neurons, all relu with droppout from 0.3, up to 0.5, and then progressively down to 0.05

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.dropout_a(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout_a(x)
        x = F.relu(self.fc3(x))
        # x = self.dropout_b(x)
        # x = F.relu(self.fc99(x))
        # x = self.dropout_b(x)
        # x = F.tanh(self.fc100(x))
        # x = self.dropout_b(x)
        # x = F.tanh(self.fc101(x))
        x = F.relu(self.fc4(x))
        # x = self.dropout_b(x)
        x = F.relu(self.fc5(x))
        # x = self.dropout_c(x)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        # x = F.relu(self.fc9(x))
        # x = F.relu(self.fc10(x))
        # x = F.relu(self.fc11(x))
        # x = F.relu(self.fc12(x))
        x = self.fc13(x)

        return x

# Instantiate the model
input_dim = X_train.shape[1]
output_dim = Y_train.shape[1]
#print(output_dim)
model = SimpleDNN(input_dim, output_dim)

# Loss and optimizer
#criterion = nn.BCELoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

training_loss = []
eval_loss = []

# Training loop
num_epochs = 50
model.train()
for epoch in range(num_epochs):
    for features, labels in train_loader:
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        outputs = model(features)
        if epoch%10 == 0:
            loss_eval = criterion(outputs.squeeze(), labels)
    
    if epoch%10 == 0:
        # print(loss.detach().item())
        # print(loss_eval.detach().item())
        training_loss.append(loss.detach().item())
        eval_loss.append(loss_eval.detach().item())
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        print(f'Eval [{epoch+1}/{num_epochs}], Loss: {loss_eval.item():.4f}')
#scheduler.step(loss)


# Evaluation
model.eval()
count = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        #print(f'Layer: {name} | Size: {param.size()} | Number of parameters: {param.numel()}')
        count = count + param.numel()
print("Total param count:" + str(count))
with torch.no_grad():
    # correct = 0
    # total = 0
    # for features, labels in test_loader:
    #     outputs = model(features)
    #     predicted = (outputs.squeeze() > 0.5).float()
    #     total += labels.size(0)
    #     correct += (predicted == labels).sum().item()

    # accuracy = correct / total
    # print(f'Accuracy of the model on the test set: {accuracy * 100:.2f}%')

    for features, labels in test_loader:
        outputs = model(features)
        loss = criterion(outputs.squeeze(), labels)
        print(loss)

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
# print(Y_test[:,0].size)
# print(outputs[:,0].size())
plt.scatter(Y_test[:, 0], outputs[:, 0])
plt.title(f'Prediction of px')
plt.xlabel('Validation Set px')
plt.ylabel('Prediction of px from Model')
plt.savefig('px_plot.png')
plt.figure(figsize=(5, 5))
plt.scatter(Y_test[:, 1], outputs[:, 1])
plt.title(f'Prediction of py')
plt.xlabel('Validation Set py')
plt.ylabel('Prediction of py from Model')
plt.savefig('py_plot.png')
plt.figure(figsize=(5, 5))
plt.scatter(Y_test[:, 2], outputs[:, 2])
plt.title(f'Prediction of pz')
plt.xlabel('Validation Set pz')
plt.ylabel('Prediction of pz from Model')
plt.savefig('pz_plot.png')
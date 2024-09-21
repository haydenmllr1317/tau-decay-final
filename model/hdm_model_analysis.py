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

print("beginning")

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
    

model = SimpleLinearNN2(12,4)
criterion = nn.MSELoss()


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

df = pd.DataFrame(columns = column_names)

print('beginning eval')
model.eval()
with torch.no_grad():
    outputs = model(X_Test)
    X_Test = StandardScaler().inverse_transform(X_Test)
    outputs = StandardScaler().inverse_transform(outputs)
    for i in range(outputs.size(0)):
       if Y_Test[i,5].item() == 1 and Y_Test[i,4].item() == 1:
           add_row = [X_Test[i,0].item(), X_Test[i,1].item(), X_Test[i,2].item(), X_Test[i,3].item(),
                X_Test[i,4].item(), X_Test[i,5].item(), X_Test[i,6].item(), X_Test[i,7].item(),
                X_Test[i,8].item(), X_Test[i,9].item(), X_Test[i,10].item(), X_Test[i,11].item(),
                outputs[i,0].item(), outputs[i,1].item(), outputs[i,2].item(), outputs[i,3].item(),
                X_Test[i+1,0].item(), X_Test[i+1,1].item(), X_Test[i+1,2].item(), X_Test[i+1,3].item(),
                X_Test[i+1,4].item(), X_Test[i+1,5].item(), X_Test[i+1,6].item(), X_Test[i+1,7].item(),
                X_Test[i+1,8].item(), X_Test[i+1,9].item(), X_Test[i+1,10].item(), X_Test[i+1,11].item(),
                outputs[i+1,0].item(), outputs[i+1,1].item(), outputs[i+1,2].item(), outputs[i+1,3].item()]
           new_row = pd.DataFrame([add_row], columns = df.columns)
           df = pd.concat([df, new_row], ignore_index=True)

print('saving csv...')
print(df)
df.to_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/upsilons.csv', index=False)
print('done.')



#         if Y_Test[i, 5].item() == 0:
#             print(criterion(test_outputs[i, :], Y_Test[i, :4]).item())
#             net_m_loss += criterion(test_outputs[i, :], Y_Test[i, :4]).item()
#             m_amount += 1
#         if Y_Test[i, 5].item() == 1:
#             net_p_loss += criterion(test_outputs[i, :], Y_Test[i, :4]).item()
#             p_amount += 1
#     m_loss = net_m_loss/m_amount
#     p_loss = net_p_loss/p_amount

# print(m_loss)
# print(p_loss)
import ROOT
from fast_histogram import histogram1d
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


pion_mass = 0.13957
df = pd.read_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/15GeV_all_taus.csv')
test_mass = []
# pred_mass = []
# print(df_tau_mass)t
for i in range(len(df)):
    pi1_lv = ROOT.TLorentzVector()
    pi2_lv = ROOT.TLorentzVector()
    pi3_lv = ROOT.TLorentzVector()
    neu_test_lv = ROOT.TLorentzVector()
    # neu_pred_lv = ROOT.TLorentzVector()
    """
    pi1_lv.SetPx(df_tau_mass.iloc[i]['px_pi_1m'])
    pi1_lv.SetPy(df_tau_mass.iloc[i]['py_pi_1m'])
    pi1_lv.SetPz( df_tau_mass.iloc[i]['pz_pi_1m'])
    pi1_lv.SetMass(pion_mass)
    """
    pi1_lv.SetPtEtaPhiM(df.iloc[i]['pi1_pt'], df.iloc[i]['pi1_eta'], math.atan2(df.iloc[i]['pi1_sinphi'], df.iloc[i]['pi1_cosphi']), pion_mass)
    pi2_lv.SetPtEtaPhiM(df.iloc[i]['pi2_pt'], df.iloc[i]['pi2_eta'], math.atan2(df.iloc[i]['pi2_sinphi'], df.iloc[i]['pi2_cosphi']), pion_mass)
    pi3_lv.SetPtEtaPhiM(df.iloc[i]['pi3_pt'], df.iloc[i]['pi3_eta'], math.atan2(df.iloc[i]['pi3_sinphi'], df.iloc[i]['pi3_cosphi']), pion_mass)
    neu_test_lv.SetPtEtaPhiM(df.iloc[i]['neu_pt'], df.iloc[i]['neu_eta'], math.atan2(df.iloc[i]['neu_sinphi'], df.iloc[i]['neu_cosphi']), 0.0)

    # print(pi1_lv.Pt())
    # print(pi1_lv.Eta())
    # print(pi1_lv.Phi())
    # pi1_lv.SetPtEtaPhiM(df.iloc[i]['pi1_pt'], df.iloc[i]['pi1_eta'], df.iloc[i]['pi1_phi'], pion_mass)
    # pi2_lv.SetPtEtaPhiM(df.iloc[i]['pi2_pt'], df.iloc[i]['pi2_eta'], df.iloc[i]['pi2_phi'], pion_mass)
    # pi3_lv.SetPtEtaPhiM(df.iloc[i]['pi3_pt'], df.iloc[i]['pi3_eta'], df.iloc[i]['pi3_phi'], pion_mass)
    # neu_test_lv.SetPtEtaPhiM(df.iloc[i]['neu_pt'], df.iloc[i]['neu_eta'], df.iloc[i]['neu_phi'], 0.0)


    tau_lv_test_mass = (pi1_lv + pi2_lv + pi3_lv + neu_test_lv).M()
    test_mass.append(tau_lv_test_mass)
# print(test_mass)
bins = 80
range = (1.0, 2.0)
hist1 = histogram1d(test_mass, bins=bins, range=range)
# Bin edges
bin_edges = np.linspace(range[0], range[1], bins + 1)
# Plot histograms
plt.figure(figsize=(10, 6))
# Plot first histogram
plt.hist(bin_edges[:-1], bins=bin_edges, weights=hist1, alpha=0.5, color='blue')
# Add labels and title
plt.xlabel('Tau Mass (GeV)')
plt.ylabel('Frequency')
plt.title('Reconstructed Tau Mass using Predicted neu Momentum')
plt.legend()
# Show plot
plt.savefig('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/histograms/tau_mass_15GeV_plot.png')
# hist2 = histogram1d(test_mass, bins=bins, range=range)
# plt.figure(figsize=(10, 6))
# # Plot first histogram
# plt.hist(bin_edges[:-1], bins=bin_edges, weights=hist2, alpha=0.5, color='red')
# # Add labels and title
# plt.xlabel('Tau Mass (GeV)')
# plt.ylabel('Frequency')
# plt.title('Reconstructed Tau Mass using Gen Level Neutrinos')
# plt.legend()
# # Show plot
# plt.savefig('/isilon/export/home/gpitt3/tau-decay-ml/tau_mass_plot_test.png')

mean_mass = sum(test_mass)/len(test_mass)
print('mean:' + str(mean_mass))
print('std:' + str(np.std(test_mass)))
# mean_test_mass = (sum(pred_mass) / len(pred_mass))
# std_test_mass = np.std(pred_mass)
# print(mean_test_mass, std_test_mass)
# mass_difference = [np.abs(a_i - b_i) for a_i, b_i in zip(test_mass, pred_mass)]
# print(sum(mass_difference) / len(mass_difference))





# 10:26
# This is the code for taking in the model output information and putting it into a csv:
# outputs_array = test_outputs.numpy()
# y_test_array = y_test.numpy()
# X_test_array = X_test.numpy()
# # Step 3: Convert the NumPy array to a Pandas DataFrame
# df_x_test = pd.DataFrame(X_test_array, columns=['px_pi_1m','px_pi_2m','px_pi_3m','py_pi_1m','py_pi_2m','py_pi_3m','pz_pi_1m','pz_pi_2m','pz_pi_3m'])
# df_y_test = pd.DataFrame(y_test_array, columns=['px_neu_test','py_neu_test','pz_neu_test'])
# df_y_prediction = pd.DataFrame(outputs_array, columns=['px_neu_pred','py_neu_pred','pz_neu_pred'])
# df_tau_mass = pd.concat([df_x_test, df_y_test, df_y_prediction], axis = 1)
# df_tau_mass.to_csv('model_output_tau_mass.csv')
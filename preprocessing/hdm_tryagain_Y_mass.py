import ROOT
from fast_histogram import histogram1d
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

pion_mass = 0.13957
df = pd.read_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/fixed_unboosted_upsilons.csv')
test_mass = []
# pred_mass = []
# print(df_tau_mass)
print(len(df))
for i in range(0,len(df),2):
    # if df.iloc[i]['sign'] == 'p' and df.iloc[i]['upsilon?'] == 'y':
    # print(i)
        pi1_lv = ROOT.TLorentzVector()
        pi2_lv = ROOT.TLorentzVector()
        pi3_lv = ROOT.TLorentzVector()
        neu1_test_lv = ROOT.TLorentzVector()
        pi4_lv = ROOT.TLorentzVector()
        pi5_lv = ROOT.TLorentzVector()
        pi6_lv = ROOT.TLorentzVector()
        neu2_test_lv = ROOT.TLorentzVector()

        pi1_lv.SetPtEtaPhiM(df.iloc[i]['pi1_pt'], df.iloc[i]['pi1_eta'], df.iloc[i]['pi1_phi'], pion_mass)
        pi2_lv.SetPtEtaPhiM(df.iloc[i]['pi2_pt'], df.iloc[i]['pi2_eta'], df.iloc[i]['pi2_phi'], pion_mass)
        pi3_lv.SetPtEtaPhiM(df.iloc[i]['pi3_pt'], df.iloc[i]['pi3_eta'], df.iloc[i]['pi3_phi'], pion_mass)
        neu1_test_lv.SetPtEtaPhiM(df.iloc[i]['neu_pt'], df.iloc[i]['neu_eta'], df.iloc[i]['neu_phi'], 0.0)
        pi4_lv.SetPtEtaPhiM(df.iloc[i+1]['pi1_pt'], df.iloc[i+1]['pi1_eta'], df.iloc[i+1]['pi1_phi'], pion_mass)
        pi5_lv.SetPtEtaPhiM(df.iloc[i+1]['pi2_pt'], df.iloc[i+1]['pi2_eta'], df.iloc[i+1]['pi2_phi'], pion_mass)
        pi6_lv.SetPtEtaPhiM(df.iloc[i+1]['pi3_pt'], df.iloc[i+1]['pi3_eta'], df.iloc[i+1]['pi3_phi'], pion_mass)
        neu2_test_lv.SetPtEtaPhiM(df.iloc[i+1]['neu_pt'], df.iloc[i+1]['neu_eta'], df.iloc[i+1]['neu_phi'], 0.0)

        # pi1_lv.SetPtEtaPhiM(df.iloc[i]['pi1_pt'], df.iloc[i]['pi1_eta'], math.atan2(df.iloc[i]['pi1_sinphi'], df.iloc[i]['pi1_cosphi']), pion_mass)
        # pi2_lv.SetPtEtaPhiM(df.iloc[i]['pi2_pt'], df.iloc[i]['pi2_eta'], math.atan2(df.iloc[i]['pi2_sinphi'], df.iloc[i]['pi2_cosphi']), pion_mass)
        # pi3_lv.SetPtEtaPhiM(df.iloc[i]['pi3_pt'], df.iloc[i]['pi3_eta'], math.atan2(df.iloc[i]['pi3_sinphi'], df.iloc[i]['pi3_cosphi']), pion_mass)
        # neu1_test_lv.SetPtEtaPhiM(df.iloc[i]['neu_pt'], df.iloc[i]['neu_eta'], math.atan2(df.iloc[i]['neu_sinphi'], df.iloc[i]['neu_cosphi']), 0.0)
        # pi4_lv.SetPtEtaPhiM(df.iloc[i+1]['pi1_pt'], df.iloc[i+1]['pi1_eta'], math.atan2(df.iloc[i+1]['pi1_sinphi'], df.iloc[i+1]['pi1_cosphi']), pion_mass)
        # pi5_lv.SetPtEtaPhiM(df.iloc[i+1]['pi2_pt'], df.iloc[i+1]['pi2_eta'], math.atan2(df.iloc[i+1]['pi2_sinphi'], df.iloc[i+1]['pi2_cosphi']), pion_mass)
        # pi6_lv.SetPtEtaPhiM(df.iloc[i+1]['pi3_pt'], df.iloc[i+1]['pi3_eta'], math.atan2(df.iloc[i+1]['pi3_sinphi'], df.iloc[i+1]['pi3_cosphi']), pion_mass)
        # neu2_test_lv.SetPtEtaPhiM(df.iloc[i+1]['neu_pt'], df.iloc[i+1]['neu_eta'], math.atan2(df.iloc[i+1]['neu_sinphi'], df.iloc[i+1]['neu_cosphi']), 0.0)


        # upsilon_mass = (pi1_lv + pi2_lv + pi3_lv + neu1_test_lv + pi4_lv + pi5_lv + pi6_lv + neu2_test_lv).M()
        upsilon_mass = (pi1_lv + pi2_lv + pi3_lv + pi4_lv + pi5_lv + pi6_lv).M()
        # upsilon_mass = (pi4_lv + pi5_lv + pi6_lv + neu2_test_lv).M()

        test_mass.append(upsilon_mass)
# print(test_mass)
bins = 80
range = (5,15)
hist1 = histogram1d(test_mass, bins=bins, range=range)
# Bin edges
bin_edges = np.linspace(range[0], range[1], bins + 1)
# Plot histograms
plt.figure(figsize=(10, 6))
# Plot first histogram
plt.hist(bin_edges[:-1], bins=bin_edges, weights=hist1, alpha=0.5, color='blue')
# Add labels and title
plt.xlabel('Visible Upsilon Mass (GeV)')
plt.ylabel('Frequency')
plt.title('Nominal Upsilon Mass without Neutrinos')
plt.legend()
# Show plot
plt.savefig('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/histograms/no_neutrino_Y_mass.png')
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
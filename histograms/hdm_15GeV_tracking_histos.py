import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/weird_eta_problems_15GeV.csv')

print(df.size)
plt.hist(df[['gen_pi1_from_tau_pt','gen_pi2_from_tau_pt','gen_pi3_from_tau_pt']].min(axis=1), bins=50, alpha=0.5, label='Min pT of Unmatched Pions', color='blue')
print('first histo made')
# plt.hist(sample_df2['unmatched_pion_pt'], bins=df2_bins, alpha=0.5, label='All pT of Nonmatched RECO Pions', color='orange')
# print('second histo made')

plt.xlabel('pT')
plt.ylabel('Frequency')
plt.title('pT of all GEN Pions')
plt.savefig('15GeV_pions_pt_matched_cmon.png')
print('done.')
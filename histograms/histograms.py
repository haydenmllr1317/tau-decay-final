from fast_histogram import histogram1d
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

gen_information = pd.read_csv('/isilon/export/home/gpitt3/tau-decay-ml/preprocessing/gen_information_first_miniaod.csv')
model_information = pd.read_csv('/isilon/export/home/gpitt3/tau-decay-ml/preprocessing/reco_information_first_miniaod.csv')


# Parameters for the histogram
bins = 80  # Number of bins
range_min = 0
range_max = 180



# Compute the histogram
hist1 = histogram1d(model_information['tau_no_neutrino_mass'], bins=bins, range=[range_min, range_max])
# Create bin edges
bin_edges = np.linspace(range_min, range_max, bins + 1)

# Plotting the histogram
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], hist1, width=np.diff(bin_edges), edgecolor='blue')
plt.title('Invariant Mass of Upsilon')
plt.xlabel('Invariant Mass')
plt.ylabel('Frequency')
plt.savefig('/isilon/export/home/gpitt3/tau-decay-ml/hist.png')




import pandas as pd
import os

# Directory containing the CSV files
csv_directory = '/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/40k_Varied_Y_Mass_Files'

# List to hold each dataframe
dfs = []
dfs_gen = []

# Loop through all files in the directory
for filename in os.listdir(csv_directory):
    if filename.endswith('GEN.csv'):
        file_path = os.path.join(csv_directory, filename)
        df = pd.read_csv(file_path)
        dfs_gen.append(df)
    if filename.endswith('Y.csv'):
        file_path = os.path.join(csv_directory, filename)
        df = pd.read_csv(file_path)
        dfs.append(df)

# Concatenate all dataframes in the list
combined_df= pd.concat(dfs, ignore_index=True)
combined_df_gen = pd.concat(dfs_gen, ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df_gen.to_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/200k_varied_Y_GEN.csv', index=False)
combined_df.to_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/200k_varied_Y.csv', index=False)

print("All CSV files combined")
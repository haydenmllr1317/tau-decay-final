import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

pion_mass = 0.13957
df = pd.read_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/unboosted_upsilons.csv')
print(len(df))
for i in range(len(df)):
    df.iloc[i]['pi1_phi'] = math.atan2(df.iloc[i]['pi1_sinphi'], df.iloc[i]['pi1_cosphi'])
    df.iloc[i]['pi2_phi'] = math.atan2(df.iloc[i]['pi2_sinphi'], df.iloc[i]['pi2_cosphi'])
    df.iloc[i]['pi3_phi'] = math.atan2(df.iloc[i]['pi3_sinphi'], df.iloc[i]['pi3_cosphi'])
    df.iloc[i]['neu_phi'] = math.atan2(df.iloc[i]['neu_sinphi'], df.iloc[i]['neu_cosphi'])
df.drop('pi1_cosphi', axis=1)
df.drop('pi1_sinphi', axis=1)
df.drop('pi2_cosphi', axis=1)
df.drop('pi2_sinphi', axis=1)
df.drop('pi3_cosphi', axis=1)
df.drop('pi3_sinphi', axis=1)
df.drop('neu_cosphi', axis=1)
df.drop('neu_sinphi', axis=1)

df.to_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/fixed_unboosted_upsilons.csv')
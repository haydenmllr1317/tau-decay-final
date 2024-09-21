import pandas as pd
import ROOT
from ROOT import TLorentzVector, TVector3
import numpy as np
import math

# Load the data frame (processed pions)
df = pd.read_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/upsilons.csv')
df_og = pd.read_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/all_nommass_reco_data.csv')

# Rodrigues formula for rotation (inverse rotations will use negative angles)
def rotation_from_axis_and_angle(k, theta, v):
    return (v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta)))

pion_mass = 0.13957

j = 37783

# Inverse transformation
for i in range(len(df)):
    # Create Lorentz Vectors from the processed data
    pi1 = TLorentzVector()
    pi2 = TLorentzVector()
    pi3 = TLorentzVector()
    neu = TLorentzVector()
    pi1.SetPtEtaPhiM(df['pi1_pt'][i], df['pi1_eta'][i], math.atan2(df.iloc[i]['pi1_sinphi'], df.iloc[i]['pi1_cosphi']), pion_mass)
    pi2.SetPtEtaPhiM(df['pi2_pt'][i], df['pi2_eta'][i], math.atan2(df.iloc[i]['pi2_sinphi'], df.iloc[i]['pi2_cosphi']), pion_mass)
    pi3.SetPtEtaPhiM(df['pi3_pt'][i], df['pi3_eta'][i], math.atan2(df.iloc[i]['pi3_sinphi'], df.iloc[i]['pi3_cosphi']), pion_mass)
    neu.SetPtEtaPhiM(df['neu_pt'][i], df['neu_eta'][i], math.atan2(df.iloc[i]['neu_sinphi'], df.iloc[i]['neu_cosphi']), 0.0)

    while df_og['upsilon?'][j] != 'y':
        j += 1

    og_pi1 = TLorentzVector()
    og_pi2 = TLorentzVector()
    og_pi3 = TLorentzVector()
    og_pi1.SetPtEtaPhiM(df_og['pi1_pt'][j], df_og['pi1_eta'][j], df_og['pi1_phi'][j], pion_mass)
    og_pi2.SetPtEtaPhiM(df_og['pi2_pt'][j], df_og['pi2_eta'][j], df_og['pi2_phi'][j], pion_mass)
    og_pi3.SetPtEtaPhiM(df_og['pi3_pt'][j], df_og['pi3_eta'][j], df_og['pi3_phi'][j], pion_mass)

    j += 1

    og_total_momentum = og_pi1 + og_pi2 + og_pi3
    beta = og_total_momentum.BoostVector()
    og_pi1.Boost(-beta)
    og_pi2.Boost(-beta)
    og_pi3.Boost(-beta)


    #picking the highest pT pion
    og_pi_list = [og_pi2,og_pi3]
    og_best_pi = og_pi1
    for pi in og_pi_list:
        if pi.Pt() > og_best_pi.Pt():
            og_best_pi = pi
    
    og_leading_pt_pi = np.array([og_best_pi.Px(),og_best_pi.Py(),0])

    # Rotate back around x-axis (inverse rotation)
    pi1_vec = np.array([pi1.Px(), pi1.Py(), pi1.Pz()])
    pi2_vec = np.array([pi2.Px(), pi2.Py(), pi2.Pz()])
    pi3_vec = np.array([pi3.Px(), pi3.Py(), pi3.Pz()])
    neu_vec = np.array([neu.Px(), neu.Py(), neu.Pz()])

    x_axis = np.array([1,0,0])
    x_pre_norm = np.cross(x_axis, og_leading_pt_pi)
    x_axis_of_rotation = x_pre_norm/np.linalg.norm(x_pre_norm)
    # print(x_axis_of_rotation)
    x_rotation_angle = np.arccos(np.dot(x_axis, og_leading_pt_pi)/np.linalg.norm(og_leading_pt_pi))

    pi1_vec = rotation_from_axis_and_angle(x_axis_of_rotation, -x_rotation_angle, pi1_vec)
    pi2_vec = rotation_from_axis_and_angle(x_axis_of_rotation, -x_rotation_angle, pi2_vec)
    pi3_vec = rotation_from_axis_and_angle(x_axis_of_rotation, -x_rotation_angle, pi3_vec)
    neu_vec = rotation_from_axis_and_angle(x_axis_of_rotation, -x_rotation_angle, neu_vec)

    z_axis = np.array([0,0,1])
    z_pre_norm = np.cross(z_axis, -beta)
    if np.linalg.norm(z_pre_norm) == 0 or np.linalg.norm(-beta) == 0:
        final_pi1 = TLorentzVector()
        final_pi2 = TLorentzVector()
        final_pi3 = TLorentzVector()
        final_neu = TLorentzVector()

        final_pi1.SetPxPyPzE(pi1_vec[0], pi1_vec[1], pi1_vec[2], np.sqrt(pi1_vec[0]**2 + pi1_vec[1]**2 + pi1_vec[2]**2 + pi1.M()**2))
        final_pi2.SetPxPyPzE(pi2_vec[0], pi2_vec[1], pi2_vec[2], np.sqrt(pi2_vec[0]**2 + pi2_vec[1]**2 + pi2_vec[2]**2 + pi2.M()**2))
        final_pi3.SetPxPyPzE(pi3_vec[0], pi3_vec[1], pi3_vec[2], np.sqrt(pi3_vec[0]**2 + pi3_vec[1]**2 + pi3_vec[2]**2 + pi3.M()**2))
        final_neu.SetPxPyPzE(neu_vec[0], neu_vec[1], neu_vec[2], np.sqrt(neu_vec[0]**2 + neu_vec[1]**2 + neu_vec[2]**2 + neu.M()**2))

        # Apply the inverse boost
        final_pi1.Boost(beta)
        final_pi2.Boost(beta)
        final_pi3.Boost(beta)
        final_neu.Boost(beta)
        continue
    z_axis_of_rotation = z_pre_norm/np.linalg.norm(z_pre_norm)
    z_rotation_angle = np.arccos(np.dot(z_axis, -beta)/np.linalg.norm(-beta))

    # Rotate back around z-axis (inverse rotation)
    pi1_vec = rotation_from_axis_and_angle(z_axis_of_rotation, -z_rotation_angle, pi1_vec)
    pi2_vec = rotation_from_axis_and_angle(z_axis_of_rotation, -z_rotation_angle, pi2_vec)
    pi3_vec = rotation_from_axis_and_angle(z_axis_of_rotation, -z_rotation_angle, pi3_vec)
    neu_vec = rotation_from_axis_and_angle(z_axis_of_rotation, -z_rotation_angle, neu_vec)

    # Create Lorentz Vectors after inverse rotations

    final_pi1 = TLorentzVector()
    final_pi2 = TLorentzVector()
    final_pi3 = TLorentzVector()
    final_neu = TLorentzVector()

    final_pi1.SetPxPyPzE(pi1_vec[0], pi1_vec[1], pi1_vec[2], np.sqrt(pi1_vec[0]**2 + pi1_vec[1]**2 + pi1_vec[2]**2 + pi1.M()**2))
    final_pi2.SetPxPyPzE(pi2_vec[0], pi2_vec[1], pi2_vec[2], np.sqrt(pi2_vec[0]**2 + pi2_vec[1]**2 + pi2_vec[2]**2 + pi2.M()**2))
    final_pi3.SetPxPyPzE(pi3_vec[0], pi3_vec[1], pi3_vec[2], np.sqrt(pi3_vec[0]**2 + pi3_vec[1]**2 + pi3_vec[2]**2 + pi3.M()**2))
    final_neu.SetPxPyPzE(neu_vec[0], neu_vec[1], neu_vec[2], np.sqrt(neu_vec[0]**2 + neu_vec[1]**2 + neu_vec[2]**2 + neu.M()**2))

    # Apply the inverse boost
    final_pi1.Boost(beta)
    final_pi2.Boost(beta)
    final_pi3.Boost(beta)
    final_neu.Boost(beta)

    # Save the original coordinates back to the DataFrame
    df.loc[i, 'pi1_pt'] = final_pi1.Pt()
    df.loc[i, 'pi1_eta'] = final_pi1.Eta()
    df.loc[i, 'pi1_phi'] = final_pi1.Phi()
    df.loc[i, 'pi2_pt'] = final_pi2.Pt()
    df.loc[i, 'pi2_eta'] = final_pi2.Eta()
    df.loc[i, 'pi2_phi'] = final_pi2.Phi()
    df.loc[i, 'pi3_pt'] = final_pi3.Pt()
    df.loc[i, 'pi3_eta'] = final_pi3.Eta()
    df.loc[i, 'pi3_phi'] = final_pi3.Phi()
    df.loc[i, 'neu_pt'] = final_neu.Pt()
    df.loc[i, 'neu_eta'] = final_neu.Eta()
    df.loc[i, 'neu_phi'] = final_neu.Phi()

# Save the DataFrame with the original coordinates
df.to_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/unboosted_upsilons.csv')

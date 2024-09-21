import pandas as pd
import ROOT
from ROOT import TLorentzVector, TVector3
import numpy as np

# import data frame (reco pions)
df = pd.read_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/200k_varied_Y.csv')

#this is the rodrigues formula (k is the axis of rotation, theta is the angle). k must be normalized
def rotation_from_axis_and_angle(k,theta,v):
    return (v*np.cos(theta) + np.cross(k,v)*np.sin(theta) + k*np.dot(k,v)*(1-np.cos(theta)))

pion_mass = 0.13957

# move to COM frame
for i in range(len(df)):
    # create Lorentz Vectors
    pi1 = TLorentzVector()
    pi2 = TLorentzVector()
    pi3 = TLorentzVector()
    neu = TLorentzVector()
    neu.SetPtEtaPhiM(df['neu_pt'][i], df['neu_eta'][i], df['neu_phi'][i],0.0)
    pi1.SetPtEtaPhiM(df['pi1_pt'][i], df['pi1_eta'][i], df['pi1_phi'][i],pion_mass)
    pi2.SetPtEtaPhiM(df['pi2_pt'][i], df['pi2_eta'][i], df['pi2_phi'][i],pion_mass)
    pi3.SetPtEtaPhiM(df['pi3_pt'][i], df['pi3_eta'][i], df['pi3_phi'][i],pion_mass)
    # total_energy = pi1.E()+ pi2.E() + pi3.E()
    # create total momentum vector (and thus boost vector)
    total_momentum = pi1 + pi2 + pi3
    beta = total_momentum.BoostVector()
    pi1.Boost(-beta)
    pi2.Boost(-beta)
    pi3.Boost(-beta)
    neu.Boost(-beta)

    # print('these should be 0 if in COM frame:')
    # print(pi1.Px() + pi2.Px() + pi3.Px())
    # print(pi1.Py() + pi2.Py() + pi3.Py())
    # print(pi1.Pz() + pi2.Pz() + pi3.Pz())

    # df['pi1_pt'][i] = pi1.Pt()
    # df['pi1_eta'][i] = pi1.Eta()
    # df['pi1_phi'][i] = pi1.Phi()
    # df['pi2_pt'][i] = pi2.Pt()
    # df['pi2_eta'][i] = pi2.Eta()
    # df['pi2_phi'][i] = pi2.Phi()
    # df['pi3_pt'][i] = pi3.Pt()
    # df['pi3_eta'][i] = pi3.Eta()
    # df['pi3_phi'][i] = pi3.Phi()
    # df['neu_pt'][i] = neu.Pt()
    # df['neu_eta'][i] = neu.Eta()
    # df['neu_phi'][i] = neu.Phi()

    # boost_for_rotation = np.array([-beta.Px(), -beta.Py(), -beta.Pz()])

    #picking the highest pT pion
    pi_list = [pi2,pi3]
    best_pi = pi1
    for pi in pi_list:
        if pi.Pt() > best_pi.Pt():
            best_pi = pi

    # here is np.array details of leading pT pion for use in rotation
    leading_pt_pi = np.array([best_pi.Px(),best_pi.Py(),0]) #/best_pi_norm

    # here are the np.array info for the pions and neutrinos AFTER the boost
    good_pi1 = np.array([pi1.Px(),pi1.Py(),pi1.Pz()])
    good_pi2 = np.array([pi2.Px(),pi2.Py(),pi2.Pz()])
    good_pi3 = np.array([pi3.Px(),pi3.Py(),pi3.Pz()])
    good_neu = np.array([neu.Px(),neu.Py(),neu.Pz()])
    
    # # print(are_points_coplanar(good_pi1,good_pi2,good_pi3,boost_for_rotation))

    z_axis = np.array([0,0,1])
    z_pre_norm = np.cross(z_axis, -beta)
    z_axis_of_rotation = z_pre_norm/np.linalg.norm(z_pre_norm)
    z_rotation_angle = np.arccos(np.dot(z_axis, -beta)/np.linalg.norm(-beta))

    z_good_pi1 = rotation_from_axis_and_angle(z_axis_of_rotation, z_rotation_angle, good_pi1)
    z_good_pi2 = rotation_from_axis_and_angle(z_axis_of_rotation, z_rotation_angle, good_pi2)
    z_good_pi3 = rotation_from_axis_and_angle(z_axis_of_rotation, z_rotation_angle, good_pi3)
    z_good_neu = rotation_from_axis_and_angle(z_axis_of_rotation, z_rotation_angle, good_neu)

    x_axis = np.array([1,0,0])
    x_pre_norm = np.cross(x_axis, leading_pt_pi)
    x_axis_of_rotation = x_pre_norm/np.linalg.norm(x_pre_norm)
    # print(x_axis_of_rotation)
    x_rotation_angle = np.arccos(np.dot(x_axis, leading_pt_pi)/np.linalg.norm(leading_pt_pi))

    pi1 = rotation_from_axis_and_angle(x_axis_of_rotation, x_rotation_angle, z_good_pi1) # CHANGE BACK!
    pi2 = rotation_from_axis_and_angle(x_axis_of_rotation, x_rotation_angle, z_good_pi2)
    pi3 = rotation_from_axis_and_angle(x_axis_of_rotation, x_rotation_angle, z_good_pi3)
    neu = rotation_from_axis_and_angle(x_axis_of_rotation, x_rotation_angle, z_good_neu)

    final_pi1 = TLorentzVector()
    final_pi2 = TLorentzVector()
    final_pi3 = TLorentzVector()
    final_neu = TLorentzVector()

    final_pi1.SetPxPyPzE(pi1[0], pi1[1], pi1[2], 100.0)
    final_pi2.SetPxPyPzE(pi2[0], pi2[1], pi2[2], 100.0)
    final_pi3.SetPxPyPzE(pi3[0], pi3[1], pi3[2], 100.0)
    final_neu.SetPxPyPzE(neu[0], neu[1], neu[2], 100.0)
    
    df.loc[i,'pi1_pt'] = final_pi1.Pt()
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

# create CSV from our data
df.to_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/200k_varied_Y_boosted.csv')
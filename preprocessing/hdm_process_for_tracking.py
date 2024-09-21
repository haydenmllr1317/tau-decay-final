# Script to process MiniAODs and convert them into flat ntuples

# Import section

import ROOT # for 4-vector builds
from DataFormats.FWLite import Events, Handle # to open MiniAODs
import pandas as pd
# from fast_histogram import histogram1d
import matplotlib.pyplot as plt
import numpy as np

import os

# parser for arguments

import argparse

# Mapping for pdgID

pdgId_map = {
    553: 'upsilon_id',
    15: 'tau_plus' ,
    -15: 'tau_minus',
    211: 'pion_plus' ,
    -211: 'pion_minus',
    16: 'tau_neu_plus' ,
    -16: 'tau_neu_minus',
    111: 'neutral_pion_id' ,
    22: 'photon_id' ,
}
# Function definitions 

df = pd.DataFrame(columns=['y'])
#df_unmatched = pd.DataFrame(columns=['pt'])
rap = []
for i in range(1):
    # if i != 2 and i!=6:
        print(f'file: {i} being read in')

        if __name__ == '__main__':

            # Argument parser and fixing the CMSSW version via the options container
            parser = argparse.ArgumentParser(description='Args')
            parser.add_argument('--f_in', default=f'UpsilonToTauTau_3prong_miniaod_part{i}') 
            #parser.add_argument('--maxEvents', default = 100)
            args = parser.parse_args()

            # file path
            path = '/isilon/export/home/mhhadley/preprocessingNomMassMiniaodWithGenSimLowPtTauMatcher_noCheckForNeutralPiInDecayChainButPiPtCutIncl_piPtCut0p35/CMSSW_10_2_15/src/low_pt_tau_reco/preprocessing'
            filename = path + '/' + args.f_in + '.root'

            from FWCore.ParameterSet.VarParsing import VarParsing # Needed to input the file
            options = VarParsing ('python')  
            options.inputFiles = [filename]
            options.maxEvents =  -1 # run on 10 events first, -1 for all the events

            options.parseArguments()
            #print(options)

            # Labels and handles

            handleGen  = Handle ("std::vector<reco::GenParticle>") # CMSSW list of reco::GenParticles
            labelGen = ("prunedGenParticles")

            handleReco = Handle ("std::vector<pat::PackedCandidate>") # CMSSW list of reconstructed candidates
            recoLabel = ("packedPFCandidates")

            lostLabel = ("lostTracks")

            handleMET = Handle ("std::vector<pat::MET>")
            labelMET = ("slimmedMETs")


            # Open the events in MiniAOD

            events = Events(options)

           

            for event in events: # Loops over all the events sepcified with maxEvents
                #print("Event number",eventNumber)
                event.getByLabel(labelGen, handleGen)
                gen_particles = handleGen.product()

                event.getByLabel(recoLabel, handleReco)
                pf_particles = handleReco.product()

                event.getByLabel(lostLabel, handleReco)
                lost_particles = handleReco.product()
            
                event.getByLabel(labelMET, handleMET)
                met = handleMET.product().front()


                pi_plus_list = []
                pi_minus_list = []
                pi_neutral_list = []
                neu_plus_list = []
                neu_minus_list = []
                tau_plus_list = []
                tau_minus_list = []
                upsilon_list = []
                # photon_list = []


                for gen_particle in gen_particles:
                    if gen_particle.pdgId() == 553:
                        upsilon_list.append(gen_particle)
                    if gen_particle.pdgId() == 15:
                        tau_plus_list.append(gen_particle)
                    if gen_particle.pdgId() == -15:
                        tau_minus_list.append(gen_particle)
                    if gen_particle.pdgId() == 211:
                        pi_plus_list.append(gen_particle)
                    if gen_particle.pdgId() == -211:
                        pi_minus_list.append(gen_particle)
                    if gen_particle.pdgId() == 16:
                        neu_plus_list.append(gen_particle)
                    if gen_particle.pdgId() == -16:
                        neu_minus_list.append(gen_particle)
                    if gen_particle.pdgId() == 111:
                        pi_neutral_list.append(gen_particle)

                upsilon = upsilon_list[0]
                rap.append(upsilon.y())

column = ['y']
df = pd.DataFrame(rap, columns=column)
df.to_csv('upsilon_rapidity.csv', index=False)
                    

                # if len(upsilon_list) == 1:
                #     # print('We have this many upsilons:')
                #     # print(len(upsilon_list))
                #     for i in range(len(upsilon_list)):
                #         # print('This upsilons has this many mothers:')
                #         # print(upsilon_list[i].numberOfMothers())
                #         for j in range(0, upsilon_list[i].numberOfMothers()):
                #             # print(upsilon_list[i].mother(j).pdgId())
                #             # print('This particle has this many mothers')
                #             print(upsilon_list[i].mother(j).numberOfMothers())
                #             # for k in range(0, upsilon_list[i].mother(j).numberOfMothers()):
                #             #     #print(upsilon_list[i].mother(j).mother(k).pdgId())
                #             #     for l in range(0, upsilon_list[i].mother(j).mother(k).numberOfMothers()):
                #             #         print(upsilon_list[i].mother(j).mother(k).mother(l).pdgId())


                # tau_plus_counter = 0
                # tau_minus_counter = 0

                # def isAncestor(a,p):
                #     if not p: 
                #         return False
            
                #     if a == p: 
                #         return True
            
                #     for i in range(0, p.numberOfMothers()):
                #         if isAncestor(a,p.mother(i)): 
                #             return True
                
                #Loops through each upsilon in the event, tagging upsilons that decay into a tau and an antitau and keep information on these decays
                # Does not check to see if there is only one tau daughter, because I assumed Upsilon decays in pairs? Check with Marco.
                # Also, should we consider taus that decay from particles that aren't upsilons?


                # tau_plus_daughters = []
                # tau_plus_neutrino = []
                # tau_minus_daughters = []
                # tau_minus_neutrino = []
                # tau_plus_neutral_check = 0
                # tau_minus_neutral_check = 0
                # good_taup = False
                # good_taum = False
                # taup_daughter = False
                # taum_daughter = False
                
                # upsilon = upsilon_list[0]
                #upsilon_counter += 1

                # if upsilon.mass() < 12:
                #     # massive_upsilon_count += 1

                #     for tau_plus in tau_plus_list:
                #         #Should only happen once, since there can't be a double tau or double antitau decay (I beleive, again check with Marco)
                #         if isAncestor(upsilon, tau_plus.mother(0)):
                #             taup_daughter = True
                #             for pi_plus in pi_plus_list:
                #                 if isAncestor(tau_plus, pi_plus.mother(0)):
                #                     tau_plus_daughters.append(pi_plus)
                #             for pi_minus in pi_minus_list:
                #                 if isAncestor(tau_plus, pi_minus.mother(0)):
                #                     tau_plus_daughters.append(pi_minus)
                #             for pi_neutral in pi_neutral_list:
                #                 if isAncestor(tau_plus, pi_neutral.mother(0)):
                #                     tau_plus_daughters.append(pi_neutral)
                #                     tau_plus_neutral_check = tau_plus_neutral_check + 1
                #             for neutrino_plus in neu_plus_list:
                #                 if isAncestor(tau_plus, neutrino_plus.mother(0)):
                #                     tau_plus_neutrino.append(neutrino_plus)
                #             for neutrino_minus in neu_minus_list:
                #                 if isAncestor(tau_plus, neutrino_minus.mother(0)):
                #                     tau_plus_neutrino.append(neutrino_minus)
                #             if len(tau_plus_daughters) == 3 and tau_plus_neutral_check == 0:
                #                 tau_plus_keep = tau_plus
                #                 good_taup = True
                #                 # print(tau_plus_keep.pt())
                #                 # print(tau_plus_keep.eta())
                #                 # print(tau_plus_keep.phi())
                #             break
                
                    

                #     for tau_minus in tau_minus_list:
                #         #Should only happen once, since there can't be a double tau or double antitau decay (I beleive, again check with Marco)
                #         if isAncestor(upsilon, tau_minus.mother(0)):
                #             taum_daughter = True
                #             for pi_plus in pi_plus_list:
                #                 if isAncestor(tau_minus, pi_plus.mother(0)):
                #                     tau_minus_daughters.append(pi_plus)
                #             for pi_minus in pi_minus_list:
                #                 if isAncestor(tau_minus, pi_minus.mother(0)):
                #                     tau_minus_daughters.append(pi_minus)
                #             for pi_neutral in pi_neutral_list:
                #                 if isAncestor(tau_minus, pi_neutral.mother(0)):
                #                     tau_minus_daughters.append(pi_neutral)
                #                     tau_minus_neutral_check = tau_minus_neutral_check + 1
                #             for neutrino_plus in neu_plus_list:
                #                 if isAncestor(tau_minus, neutrino_plus.mother(0)):
                #                     tau_minus_neutrino.append(neutrino_plus)
                #             for neutrino_minus in neu_minus_list:
                #                 if isAncestor(tau_minus, neutrino_minus.mother(0)):
                #                     tau_minus_neutrino.append(neutrino_minus)
                #             if len(tau_minus_daughters) == 3 and tau_minus_neutral_check == 0:
                #                 tau_minus_keep = tau_minus
                #                 good_taum = True
                #                 # print(tau_minus_keep.pt())
                #                 # print(tau_minus_keep.eta())
                #                 # print(tau_minus_keep.phi())
                #             break
                            
                # #print(upsilon_counter)
                        
                #     #Enters this if statement for a given Upsilon if it has both a tau and antitau daughter particle.
                #     if taup_daughter and taum_daughter:

                #         #Case in which tau decays without neutral pions, but antitau does not
                #         if good_taup and not good_taum:

                #             ###MATCHING CODE###
                #             tau_from_upsilon = tau_plus_keep
                #             matched_pion_plus = []
                #             # nonmatched_pion_plus = []
                #             dR = []

                #             for gen_pion_plus in tau_plus_daughters:
                #                 min_deltaR_plus = 999
                #                 match = False
                #                 for reco_particle in pf_particles:
                #                     if reco_particle.pdgId() == gen_pion_plus.pdgId():

                #                         reco_lv = ROOT.TLorentzVector() 
                #                         reco_lv.SetPtEtaPhiM(reco_particle.pt(), reco_particle.eta(), reco_particle.phi(), reco_particle.mass())

                #                         gen_lv_plus = ROOT.TLorentzVector()
                #                         gen_lv_plus.SetPtEtaPhiM(gen_pion_plus.pt(), gen_pion_plus.eta(), gen_pion_plus.phi(), gen_pion_plus.mass())

                #                         deltaR_plus = gen_lv_plus.DeltaR(reco_lv)
                #                         deltaPT_plus = (reco_lv.Pt() - gen_lv_plus.Pt()) / gen_lv_plus.Pt()
                                
                #                         if abs(deltaR_plus) < 0.1 and abs(deltaPT_plus) < 0.3 and abs(deltaR_plus) < min_deltaR_plus and abs(reco_particle.eta()) < 2.5 and reco_particle not in matched_pion_plus:
                #                             min_deltaR_plus = deltaR_plus
                #                             matched_pion_p = reco_particle
                #                             match = True
                #                 if match:
                #                     matched_pion_plus.append(matched_pion_p)
                #                     dR.append(min_deltaR_plus)
                #                 else:
                #                     continue
                #             # for reco_particle in pf_particles:
                #             #     if reco_particle.pdgId() == 211 or reco_particle.pdgId() == -211:
                #             #         if reco_particle not in matched_pion_plus:
                #             #             nonmatched_pion_plus.append(reco_particle.pt())

                #             # columns = ['unmatched_pion_pt']
                #             #add_row = [nonmatched_pion_plus]
                #             # df_unmatched_pions = pd.DataFrame(nonmatched_pion_plus, columns=columns)
                #             # df_unmatched = pd.concat([df_unmatched,df_unmatched_pions], ignore_index=True)

                #             #Continues if and only if each daughter pion has found a match
                #             if len(matched_pion_plus) == len(tau_plus_daughters):
                #                 # pi1_plus_lv = ROOT.TLorentzVector()
                #                 # pi2_plus_lv = ROOT.TLorentzVector()
                #                 # pi3_plus_lv = ROOT.TLorentzVector()
                #                 # neutrino_plus_lv = ROOT.TLorentzVector()

                #                 # pi1_plus_lv.SetPtEtaPhiM(matched_pion_plus[0].pt(), matched_pion_plus[0].eta(), matched_pion_plus[0].phi(), 0.139)
                #                 # pi2_plus_lv.SetPtEtaPhiM(matched_pion_plus[1].pt(), matched_pion_plus[1].eta(), matched_pion_plus[1].phi(), 0.139)
                #                 # pi3_plus_lv.SetPtEtaPhiM(matched_pion_plus[2].pt(), matched_pion_plus[2].eta(), matched_pion_plus[2].phi(), 0.139)
                #                 # neutrino_plus_lv.SetPtEtaPhiM(tau_plus_neutrino[0].pt(), tau_plus_neutrino[0].eta(), tau_plus_neutrino[0].phi(), 0)

                #                 # tau_plus_no_neutrino_lv = pi1_plus_lv + pi2_plus_lv + pi3_plus_lv
                #                 # tau_plus_with_neutrino_lv = pi1_plus_lv + pi2_plus_lv + pi3_plus_lv + neutrino_plus_lv

                #                 # tau_no_neutrino_mass = tau_plus_no_neutrino_lv.M()
                #                 # tau_with_neutrino_mass = tau_plus_with_neutrino_lv.M()

                #                 #switching nomenclature here to tau and antitau

                #                 # pi1_from_tau_pt = pi1_plus_lv.Pt()
                #                 # pi2_from_tau_pt = pi2_plus_lv.Pt()
                #                 # pi3_from_tau_pt = pi3_plus_lv.Pt()

                #                 # max_pt_pion = max(pi1_from_tau_pt, pi2_from_tau_pt, pi3_from_tau_pt)

                #                 # columns = ['pion_max_pt']
                #                 # add_row = [max_pt_pion]
                #                 # df_maxpt_pion_matched = pd.DataFrame([add_row], columns=columns)
                #                 # df = pd.concat([df,df_maxpt_pion_matched], ignore_index=True)
                #                 # print(len(dR))
                #                 column = ['deltaR']
                #                 # add_row = [dR]
                #                 df_dR = pd.DataFrame(dR, columns=column)
                #                 df = pd.concat([df,df_dR], ignore_index=True)


                #         #Case in which antitau decays properly, but tau does not    
                #         if good_taum and not good_taup:

                #             #Start matching gen pions to reco pions for antitau
                                            
                #             antitau_from_upsilon = tau_minus_keep
                #             matched_pion_minus = []
                #             # unmatched_pion_minus = []
                #             dR = []


                #             for gen_pion_minus in tau_minus_daughters:
                #                 min_deltaR_minus = 999
                #                 match = False
                #                 for reco_particle in pf_particles:
                #                     if reco_particle.pdgId() == gen_pion_minus.pdgId():

                #                         reco_lv = ROOT.TLorentzVector() 
                #                         reco_lv.SetPtEtaPhiM(reco_particle.pt(), reco_particle.eta(), reco_particle.phi(), reco_particle.mass())

                #                         gen_lv_minus = ROOT.TLorentzVector()
                #                         gen_lv_minus.SetPtEtaPhiM(gen_pion_minus.pt(), gen_pion_minus.eta(), gen_pion_minus.phi(), gen_pion_minus.mass())

                #                         deltaR_minus = gen_lv_minus.DeltaR(reco_lv)
                #                         deltaPT_minus = (reco_lv.Pt() - gen_lv_minus.Pt()) / gen_lv_minus.Pt()

                #                         if abs(deltaR_minus) < 0.1 and abs(deltaPT_minus) < 0.3 and abs(deltaR_minus) < min_deltaR_minus and abs(reco_particle.eta()) < 2.5 and reco_particle not in matched_pion_minus and reco_particle not in matched_pion_plus:
                #                             min_deltaR_minus = deltaR_minus
                #                             matched_pion_m = reco_particle
                #                             match = True
                #                 if match:
                #                     matched_pion_minus.append(matched_pion_m)
                #                     dR.append(min_deltaR_minus)
                #                 else: 
                #                     continue

                #             # for reco_particle in pf_particles:
                #             #     if reco_particle.pdgId() == 211 or reco_particle.pdgId() == -211:
                #             #         if reco_particle not in matched_pion_plus:
                #             #             unmatched_pion_minus.append(reco_particle.pt())

                #             # columns = ['unmatched_pion_pt']
                #             # #add_row = [unmatched_pion_minus]
                #             # df_unmatched_pions = pd.DataFrame(unmatched_pion_minus, columns=columns)
                #             # df_unmatched = pd.concat([df_unmatched,df_unmatched_pions], ignore_index=True)

                #             #Continues if and only if each daughter pion has found a match
                #             if len(matched_pion_minus) == len(tau_minus_daughters):

                #                 # pi1_minus_lv = ROOT.TLorentzVector()
                #                 # pi2_minus_lv = ROOT.TLorentzVector()
                #                 # pi3_minus_lv = ROOT.TLorentzVector()

                #                 # pi1_minus_lv.SetPtEtaPhiM(matched_pion_minus[0].pt(), matched_pion_minus[0].eta(), matched_pion_minus[0].phi(), 0.139)
                #                 # pi2_minus_lv.SetPtEtaPhiM(matched_pion_minus[1].pt(), matched_pion_minus[1].eta(), matched_pion_minus[1].phi(), 0.139)
                #                 # pi3_minus_lv.SetPtEtaPhiM(matched_pion_minus[2].pt(), matched_pion_minus[2].eta(), matched_pion_minus[2].phi(), 0.139)

                #                 # pi1_from_antitau_pt = pi1_minus_lv.Pt()
                #                 # pi2_from_antitau_pt = pi2_minus_lv.Pt()
                #                 # pi3_from_antitau_pt = pi3_minus_lv.Pt()

                #                 # max_pt_pion = max(pi1_from_antitau_pt, pi2_from_antitau_pt, pi3_from_antitau_pt)

                #                 # columns = ['pion_max_pt']
                #                 # add_row = [max_pt_pion]
                #                 # df_maxpt_pion_matched = pd.DataFrame([add_row], columns=columns)
                #                 # df = pd.concat([df, df_maxpt_pion_matched], ignore_index=True)

                #                 column = ['deltaR']
                #                 #add_row = [dR]
                #                 df_dR = pd.DataFrame(dR, columns=column)
                #                 df = pd.concat([df,df_dR], ignore_index=True)

                                    

                #         if good_taum and good_taup:
                #             #Matching gen to reco now
                #             antitau_from_upsilon = tau_minus_keep
                #             tau_from_upsilon = tau_plus_keep
                #             matched_pion_plus = []
                #             # nonmatched_pions = []
                #             matched_pion_minus = []
                #             dRp = []
                #             dRm = []
                #             # unmatched_pion_minus = []
                #             for gen_pion_plus in tau_plus_daughters:
                #                 min_deltaR_plus = 999
                #                 match = False
                #                 for reco_particle in pf_particles:
                #                     if reco_particle.pdgId() == gen_pion_plus.pdgId():
                #                         reco_lv = ROOT.TLorentzVector() 
                #                         reco_lv.SetPtEtaPhiM(reco_particle.pt(), reco_particle.eta(), reco_particle.phi(), reco_particle.mass())

                #                         gen_lv_plus = ROOT.TLorentzVector()
                #                         gen_lv_plus.SetPtEtaPhiM(gen_pion_plus.pt(), gen_pion_plus.eta(), gen_pion_plus.phi(), gen_pion_plus.mass())

                #                         deltaR_plus = gen_lv_plus.DeltaR(reco_lv)
                #                         deltaPT_plus = (reco_lv.Pt() - gen_lv_plus.Pt()) / gen_lv_plus.Pt()
                                
                #                         if abs(deltaR_plus) < 0.1 and abs(deltaPT_plus) < 0.3 and abs(deltaR_plus) < min_deltaR_plus and abs(reco_particle.eta()) < 2.5 and reco_particle not in matched_pion_plus and reco_particle not in matched_pion_minus:
                #                             min_deltaR_plus = deltaR_plus
                #                             matched_pion_p = reco_particle
                #                             match = True
                #                 if match:
                #                     matched_pion_plus.append(matched_pion_p)
                #                     dRp.append(min_deltaR_plus)
                #                 else:
                #                     continue
                            

                #             for gen_pion_minus in tau_minus_daughters:
                #                 min_deltaR_minus = 999
                #                 match = False
                #                 for reco_particle in pf_particles:
                #                     if reco_particle.pdgId() == gen_pion_minus.pdgId():
                #                         #reco_pions += 1

                #                         reco_lv = ROOT.TLorentzVector() 
                #                         reco_lv.SetPtEtaPhiM(reco_particle.pt(), reco_particle.eta(), reco_particle.phi(), reco_particle.mass())

                #                         gen_lv_minus = ROOT.TLorentzVector()
                #                         gen_lv_minus.SetPtEtaPhiM(gen_pion_minus.pt(), gen_pion_minus.eta(), gen_pion_minus.phi(), gen_pion_minus.mass())

                #                         deltaR_minus = gen_lv_minus.DeltaR(reco_lv)
                #                         deltaPT_minus = (reco_lv.Pt() - gen_lv_minus.Pt()) / gen_lv_minus.Pt()

                #                         if abs(deltaR_minus) < 0.1 and abs(deltaPT_minus) < 0.3 and abs(deltaR_minus) < min_deltaR_minus and abs(reco_particle.eta()) < 2.5 and reco_particle not in matched_pion_minus and reco_particle not in matched_pion_plus:
                #                             min_deltaR_minus = deltaR_minus
                #                             matched_pion_m = reco_particle
                #                             match = True
                #                 if match:
                #                     matched_pion_minus.append(matched_pion_m)
                #                     dRm.append(min_deltaR_minus)
                #                 else: 
                #                     continue
                            

                #             if len(matched_pion_plus) == len(tau_plus_daughters):

                #                 # max_pt_pion = max(matched_pion_plus[0].pt(), matched_pion_plus[1].pt(), matched_pion_plus[2].pt())

                #                 # columns = ['pion_max_pt']
                #                 # add_row = [max_pt_pion]
                #                 # df_maxpt_pion_matched = pd.DataFrame([add_row], columns=columns)
                #                 # df = pd.concat([df, df_maxpt_pion_matched], ignore_index=True)

                #                 column = ['deltaR']
                #                 #add_row = [dRp]
                #                 df_dR = pd.DataFrame(dRp, columns=column)
                #                 df = pd.concat([df,df_dR], ignore_index=True)


                #             if len(matched_pion_minus) == len(tau_minus_daughters):
                #                 # pi1_from_antitau_pt = matched_pion_minus[0].pt()
                #                 # pi2_from_antitau_pt = matched_pion_minus[1].pt()
                #                 # pi3_from_antitau_pt = matched_pion_minus[2].pt()

                #                 # max_pt_pion = max(pi1_from_antitau_pt, pi2_from_antitau_pt, pi3_from_antitau_pt)

                #                 # columns = ['pion_max_pt']
                #                 # add_row = [max_pt_pion]
                #                 # df_maxpt_pion_matched = pd.DataFrame([add_row], columns=columns)
                #                 # df = pd.concat([df, df_maxpt_pion_matched], ignore_index=True)
                #                 column = ['deltaR']
                #                 add_row = [dRm]
                #                 df_dR = pd.DataFrame(dRm, columns=column)
                #                 df = pd.concat([df,df_dR], ignore_index=True)


                            # for reco_particle in pf_particles:
                            #     if reco_particle.pdgId() == 211 or reco_particle.pdgId() == -211:
                            #         if reco_particle not in matched_pion_plus:
                            #             if reco_particle not in matched_pion_minus:
                            #                 nonmatched_pions.append(reco_particle.pt())

                            # columns = ['unmatched_pion_pt']
                            # #add_row = [nonmatched_pions]
                            # df_unmatched_pions = pd.DataFrame(nonmatched_pions, columns=columns)
                            # df_unmatched = pd.concat([df_unmatched,df_unmatched_pions], ignore_index=True)

# df.to_csv('deltaR_matched_pions.csv', index=False)

# df.to_csv('maxpt_matched_pion_data.csv', index=False)
# df_unmatched.to_csv('pt_nonmatched_pion_data.csv', index=False)

# plt.hist(df['pion_max_pt'], bins=100, alpha=0.5, label='Max pT of Matched RECO Pions')
# plt.hist(df_unmatched['unmatched_pion_pt'], bins=100, alpha=0.5, label='All pT of Nonmatched RECO Pions')

# plt.xlabel('pT')
# plt.ylabel('Frequency')
# plt.title('pT of Matched vs Nonmatched RECO Pions')

# plt.savefig('matching_pt_graph.png')
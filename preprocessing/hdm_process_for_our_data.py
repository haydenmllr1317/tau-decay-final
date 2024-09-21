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

massive_upsilon_count = 0
proper_decay_number = 0
tagged_num = 0

reco_pions = 0

# for i in range(10):
    # if i != 2 and i!=6:
# Main 

if __name__ == '__main__':
    # I would like to keep a dataframe with the following information:
    # information regarding the four vectors of ALL pions that decay from taus WITHOUT neutral pions
    # Information regarding the number of upsilons that decay into taus that decay properly and improperly (number of each)
    # Information for reconstruction of taus that decay properly
    # Information for reconstruction of anti taues that decay properly

    # Argument parser and fixing the CMSSW version via the options container
    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('--f_in', default=f'40k_8GeV_Y_MINIAOD') 
    #parser.add_argument('--maxEvents', default = 100)
    args = parser.parse_args()

    # file path
    path = '/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data'
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

    eventNumber = 0

    # column_names_both = ['pi1_from_tau_pt', 'pi1_from_tau_eta', 'pi1_from_tau_phi','pi2_from_tau_pt', 'pi2_from_tau_eta', 'pi2_from_tau_phi', 'pi3_from_tau_pt', 'pi3_from_tau_eta', 'pi3_from_tau_phi',
    #                             'pi1_from_antitau_pt', 'pi1_from_antitau_eta', 'pi1_from_antitau_phi', 'pi2_from_antitau_pt', 'pi2_from_antitau_eta', 'pi2_from_antitau_phi','pi3_from_antitau_pt', 'pi3_from_antitau_eta', 'pi3_from_antitau_phi', 
    #                             'neutrino_from_tau_pt','neutrino_from_tau_eta','neutrino_from_tau_phi','neutrino_from_antitau_pt','neutrino_from_antitau_eta','neutrino_from_antitau_phi', 'tau_no_neutrino_mass', 'tau_with_neutrino_mass', 
    #                             'antitau_no_neutrino_mass', 'antitau_with_neutrino_mass', 'upsilon_no_neutrino_mass', 'upsilon_with_neutrino_mass']

    # gen_column_names_both = ['gen_pi1_from_tau_pt', 'gen_pi1_from_tau_eta', 'gen_pi1_from_tau_phi', 'gen_pi2_from_tau_pt', 'gen_pi2_from_tau_eta', 'gen_pi2_from_tau_phi', 'gen_pi3_from_tau_pt', 'gen_pi3_from_tau_eta', 'gen_pi3_from_tau_phi',
    #                         'gen_pi1_from_antitau_pt', 'gen_pi1_from_antitau_eta', 'gen_pi1_from_antitau_phi', 'gen_pi2_from_antitau_pt', 'gen_pi2_from_antitau_eta', 'gen_pi2_from_antitau_phi', 'gen_pi3_from_antitau_pt', 'gen_pi3_from_antitau_eta', 'gen_pi3_from_antitau_phi',
    #                         'gen_tau_pt', 'gen_tau_eta', 'gen_tau_phi', 'gen_antitau_pt', 'gen_antitau_eta', 'gen_antitau_phi', 'gen_upsilon_pt', 'gen_upsilon_eta', 'gen_upsilon_phi', 'gen_upsilon_mass']

    # df_toUse_both = pd.DataFrame(columns = column_names_both)

    # df_matched_gen_info_both = pd.DataFrame(columns = gen_column_names_both)

    #Generated level information regarding all pions from taus that decay properly, not just matched information
    gen_tau_only_column_names = ['gen_pi1_from_tau_pt', 'gen_pi1_from_tau_eta', 'gen_pi1_from_tau_phi', 'gen_pi2_from_tau_pt', 'gen_pi2_from_tau_eta', 'gen_pi2_from_tau_phi', 'gen_pi3_from_tau_pt', 'gen_pi3_from_tau_eta', 'gen_pi3_from_tau_phi']

    # gen_antitau_only_column_names = ['gen_pi1_from_antitau_pt', 'gen_pi1_from_antitau_eta', 'gen_pi1_from_antitau_phi', 'gen_pi2_from_antitau_pt', 'gen_pi2_from_antitau_eta', 'gen_pi2_from_antitau_phi', 'gen_pi3_from_antitau_pt', 'gen_pi3_from_antitau_eta', 'gen_pi3_from_antitau_phi']

    df_unmatched_gen_info_tau_only = pd.DataFrame(columns = gen_tau_only_column_names)

    # df_unmatched_gen_info_antitau_only = pd.DataFrame(columns = gen_antitau_only_column_names)


    #Information from taus and pions that have been matched to reco level info, separate for taus and antitaus
    column_names = ['pi1_pt', 'pi1_eta', 'pi1_phi','pi2_pt', 'pi2_eta', 'pi2_phi', 'pi3_pt', 'pi3_eta', 'pi3_phi',
                                'neu_pt','neu_eta','neu_phi', 'sign', 'upsilon?']

    # gen_column_names_only_plus = ['gen_pi1_from_tau_pt', 'gen_pi1_from_tau_eta', 'gen_pi1_from_tau_phi', 'gen_pi2_from_tau_pt', 'gen_pi2_from_tau_eta', 'gen_pi2_from_tau_phi', 'gen_pi3_from_tau_pt', 'gen_pi3_from_tau_eta', 'gen_pi3_from_tau_phi',
    #                         'gen_tau_pt', 'gen_tau_eta', 'gen_tau_phi']

    df_toUse = pd.DataFrame(columns = column_names)

    # df_matched_gen_info_tau_only = pd.DataFrame(columns = gen_column_names_only_plus)

    # gen_column_names_only_minus = ['gen_pi1_from_antitau_pt', 'gen_pi1_from_antitau_eta', 'gen_pi1_from_antitau_phi', 'gen_pi2_from_antitau_pt', 'gen_pi2_from_antitau_eta', 'gen_pi2_from_antitau_phi', 'gen_pi3_from_antitau_pt', 'gen_pi3_from_antitau_eta', 'gen_pi3_from_antitau_phi',
    #                         'gen_antitau_pt', 'gen_antitau_eta', 'gen_antitau_phi']

    # column_names_only_minus = ['pi1_from_antitau_pt', 'pi1_from_antitau_eta', 'pi1_from_antitau_phi','pi2_from_antitau_pt', 'pi2_from_antitau_eta', 'pi2_from_antitau_phi', 'pi3_from_antitau_pt', 'pi3_from_antitau_eta', 'pi3_from_antitau_phi',
    #                             'neutrino_from_antitau_pt','neutrino_from_antitau_eta','neutrino_from_antitau_phi', 'tau_no_neutrino_mass', 'tau_with_neutrino_mass']

    # df_toUse_antitau_only = pd.DataFrame(columns = column_names_only_minus)

    # df_matched_gen_info_antitau_only = pd.DataFrame(columns = gen_column_names_only_minus)

    # tracking_info_columns = ['both_proper_decay', 'proper_tau_decay_only', 'proper_antitau_decy_only', 'no_proper_decays', 'num_taus_fully_reconstructed', 'num_taus_not_reconstructed', 'num_antitaus_fully_reconstructed', 'num_antitaus_not_reconstructed', 'num_both_fully_reconstructed', 'num_both_not_reconstructed']
    # df_tracking_frequency = pd.DataFrame(columns = tracking_info_columns)

    both_proper_decay = 0
    proper_tau_decay_only = 0
    proper_antitau_decay_only = 0
    no_proper_decays = 0

    num_taus_fully_reconstructed = 0
    num_taus_not_reconstructed = 0
    num_antitaus_fully_reconstructed = 0
    num_antitaus_not_reconstructed = 0
    num_both_fully_reconstructed = 0
    num_both_not_reconstructed = 0
    #upsilon_counter = 0

    event_count = 0
    multiple_taus_counter = 0

    # events_with_photon_counter = 0
    # no_pdgid = 0
    # num_photons = {}
    # num_parts = {}

    df_photons = pd.DataFrame(columns=['gamma_pt'])
    # print(events)
    for event in events: # Loops over all the events sepcified with maxEvents
        # print('lajwknd')

        upsilon_ready = False
        event_count += 1
        has_photon = False
        # ph_count = 0
        # part_count = 0

        

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
        photon_list = []


        for gen_particle in gen_particles:
            if gen_particle.pdgId() == 553:
                # print('upsilon m: ' + str(gen_particle.mass()))
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
            if gen_particle.pdgId() == 22:
                has_photon = True
                photon_list.append(gen_particle)
            if gen_particle.pdgId() == 111:
                pi_neutral_list.append(gen_particle)
            # else: 
            #     try: print(gen_particle.pdgID())
            #     except: no_pdgid += 1



        if len(tau_minus_list) > 1 or len(tau_plus_list) > 1:
            if has_photon == False: print('doesnt have photon!??')
            print(len(tau_plus_list))
            print(len(tau_minus_list))
            # print('adjwnoainwd')
            multiple_taus_counter += 1
        # print(len(upsilon_list))
        # print(len(tau_minus_list))
        # print(len(tau_plus_list))
        # print(len(pi_neutral_list))
        # print(len(neu_minus_list))
        # print(len(neu_plus_list))

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
        
        # #Loops through each upsilon in the event, tagging upsilons that decay into a tau and an antitau and keep information on these decays
        # # Does not check to see if there is only one tau daughter, because I assumed Upsilon decays in pairs? Check with Marco.
        # # Also, should we consider taus that decay from particles that aren't upsilons?


        # tau_plus_daughters = []
        # tau_plus_neutrino = []
        # tau_minus_daughters = []
        # tau_minus_neutrino = []
        # tau_plus_neutral_check = 0
        # tau_minus_neutral_check = 0
        # good_taup = False
        # good_taum = False
        # # taup_daughter = False
        # # taum_daughter = False

        # upsilon = upsilon_list[0]
        # #upsilon_counter += 1

        # # if upsilon.mass() < 12:
        # # massive_upsilon_count += 1
        # # for upsilon in upsilon_list:
        # for tau_plus in tau_plus_list:
        #     #Should only happen once, since there can't be a double tau or double antitau decay (I beleive, again check with Marco)
        #     if isAncestor(upsilon, tau_plus.mother(0)):
        #         taup_daughter = True
        #         for pi_plus in pi_plus_list:
        #             if isAncestor(tau_plus, pi_plus.mother(0)):
        #                 tau_plus_daughters.append(pi_plus)
        #         for pi_minus in pi_minus_list:
        #             if isAncestor(tau_plus, pi_minus.mother(0)):
        #                 tau_plus_daughters.append(pi_minus)
        #         for pi_neutral in pi_neutral_list:
        #             if isAncestor(tau_plus, pi_neutral.mother(0)):
        #                 tau_plus_daughters.append(pi_neutral)
        #                 tau_plus_neutral_check = tau_plus_neutral_check + 1
        #         for neutrino_plus in neu_plus_list:
        #             if isAncestor(tau_plus, neutrino_plus.mother(0)):
        #                 tau_plus_neutrino.append(neutrino_plus)
        #         for neutrino_minus in neu_minus_list:
        #             if isAncestor(tau_plus, neutrino_minus.mother(0)):
        #                 tau_plus_neutrino.append(neutrino_minus)
        #         if len(tau_plus_daughters) == 3 and tau_plus_neutral_check == 0:
        #             # tau_plus_counter = tau_plus_counter + 1
        #             tau_plus_keep = tau_plus
        #             good_taup = True
        #         break



        # for tau_minus in tau_minus_list:
        #     #Should only happen once, since there can't be a double tau or double antitau decay (I beleive, again check with Marco)
        #     if isAncestor(upsilon, tau_minus.mother(0)):
        #         taum_daughter = True
        #         for pi_plus in pi_plus_list:
        #             if isAncestor(tau_minus, pi_plus.mother(0)):
        #                 tau_minus_daughters.append(pi_plus)
        #         for pi_minus in pi_minus_list:
        #             if isAncestor(tau_minus, pi_minus.mother(0)):
        #                 tau_minus_daughters.append(pi_minus)
        #         for pi_neutral in pi_neutral_list:
        #             if isAncestor(tau_minus, pi_neutral.mother(0)):
        #                 tau_minus_daughters.append(pi_neutral)
        #                 tau_minus_neutral_check = tau_minus_neutral_check + 1
        #         for neutrino_plus in neu_plus_list:
        #             if isAncestor(tau_minus, neutrino_plus.mother(0)):
        #                 tau_minus_neutrino.append(neutrino_plus)
        #         for neutrino_minus in neu_minus_list:
        #             if isAncestor(tau_minus, neutrino_minus.mother(0)):
        #                 tau_minus_neutrino.append(neutrino_minus)
        #         if len(tau_minus_daughters) == 3 and tau_minus_neutral_check == 0:
        #             # tau_minus_counter = tau_minus_counter + 1
        #             tau_minus_keep = tau_minus
        #             good_taum = True
        #         break

        # print(len(pi_plus_list))
        # print(len(pi_minus_list))
        # print('pi neutral!!' + str(len(pi_neutral_list)))
        # print('neup' + str(len(neu_plus_list)))
        # print('neum' + str(len(neu_minus_list)))
        # print(len(tau_plus_list))
        # print(len(tau_minus_list))
        # print(len(upsilon_list))
        # print(len(photon_list))
        # print(tau_plus_counter)
        # print(tau_minus_counter)
        # print('done')
        # try: num_photons[ph_count] += 1
        # except: num_photons[ph_count] = 1
        # try: num_parts[part_count] += 1
        # except: num_parts[part_count] = 1


        # if has_photon:
        #     events_with_photon_counter += 1

    # df_photons.to_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/photons.csv')        
             
print('There are this many events:' + str(event_count))
print('This many with multiple taus: ' + str(multiple_taus_counter))

    # print(events_with_photon_counter)
    # print('no pdgid amoun:' + str(no_pdgid))
    # print(num_photons)
    # print(num_parts)
    # COMMENTING OUT EVERYTHING BELOW
    #print(upsilon_counter)

'''

        #Enters this if statement for a given Upsilon if it has both a tau and antitau daughter particle.
        if taup_daughter and taum_daughter:

            #Case in which both tau daughters decays with neutral pions
            if not good_taup and not good_taum:
                print('something very confusing occuring')
                no_proper_decays = no_proper_decays + 1 
            
            #Case in which tau decays without neutral pions, but antitau does not
            if good_taup and not good_taum:
                print('were wrong about understanding')
                proper_decay_number += 1

                proper_tau_decay_only = proper_tau_decay_only + 1

                #Save gen pion information regardless of whether it is matched to a reco level pion
                # gen_pi1_plus_lv_unmatched = ROOT.TLorentzVector()
                # gen_pi2_plus_lv_unmatched = ROOT.TLorentzVector()
                # gen_pi3_plus_lv_unmatched = ROOT.TLorentzVector()

                # gen_pi1_plus_lv_unmatched.SetPtEtaPhiM(tau_plus_daughters[0].pt(), tau_plus_daughters[0].eta(), tau_plus_daughters[0].phi(), 0.139)
                # gen_pi2_plus_lv_unmatched.SetPtEtaPhiM(tau_plus_daughters[1].pt(), tau_plus_daughters[1].eta(), tau_plus_daughters[1].phi(), 0.139)
                # gen_pi3_plus_lv_unmatched.SetPtEtaPhiM(tau_plus_daughters[2].pt(), tau_plus_daughters[2].eta(), tau_plus_daughters[2].phi(), 0.139)

                # gen_pi1_from_tau_pt = gen_pi1_plus_lv_unmatched.Pt()
                # gen_pi1_from_tau_eta = gen_pi1_plus_lv_unmatched.Eta()
                # gen_pi1_from_tau_phi = gen_pi1_plus_lv_unmatched.Phi()

                # gen_pi2_from_tau_pt = gen_pi2_plus_lv_unmatched.Pt()
                # gen_pi2_from_tau_eta = gen_pi2_plus_lv_unmatched.Eta()
                # gen_pi2_from_tau_phi = gen_pi2_plus_lv_unmatched.Phi()

                # gen_pi3_from_tau_pt = gen_pi3_plus_lv_unmatched.Pt()
                # gen_pi3_from_tau_eta = gen_pi3_plus_lv_unmatched.Eta()                        
                # gen_pi3_from_tau_phi = gen_pi3_plus_lv_unmatched.Phi()


                # add_gen_row_tau_plus = [gen_pi1_from_tau_pt, gen_pi1_from_tau_eta, gen_pi1_from_tau_phi, gen_pi2_from_tau_pt, gen_pi2_from_tau_eta, gen_pi2_from_tau_phi, gen_pi3_from_tau_pt, gen_pi3_from_tau_eta, gen_pi3_from_tau_phi]
                # add_gen_row_df_tau_plus = pd.DataFrame([add_gen_row_tau_plus], columns = df_unmatched_gen_info_tau_only.columns)
                # df_unmatched_gen_info_tau_only = pd.concat([df_unmatched_gen_info_tau_only, add_gen_row_df_tau_plus], ignore_index=True)

                ###MATCHING CODE###
                tau_from_upsilon = tau_plus_keep
                matched_pion_plus = []

                for gen_pion_plus in tau_plus_daughters:
                    min_deltaR_plus = 999
                    match = False
                    for reco_particle in pf_particles:
                        if reco_particle.pdgId() == gen_pion_plus.pdgId():
                            reco_pions += 1

                            reco_lv = ROOT.TLorentzVector() 
                            reco_lv.SetPtEtaPhiM(reco_particle.pt(), reco_particle.eta(), reco_particle.phi(), reco_particle.mass())

                            gen_lv_plus = ROOT.TLorentzVector()
                            gen_lv_plus.SetPtEtaPhiM(gen_pion_plus.pt(), gen_pion_plus.eta(), gen_pion_plus.phi(), gen_pion_plus.mass())

                            deltaR_plus = gen_lv_plus.DeltaR(reco_lv)
                            deltaPT_plus = (reco_lv.Pt() - gen_lv_plus.Pt()) / gen_lv_plus.Pt()
        
                            if abs(deltaR_plus) < 0.1 and abs(deltaPT_plus) < 0.3 and abs(deltaR_plus) < min_deltaR_plus and abs(reco_particle.eta()) < 2.5 and reco_particle not in matched_pion_plus:
                                min_deltaR_plus = deltaR_plus
                                matched_pion_p = reco_particle
                                match = True
                    if match:
                        matched_pion_plus.append(matched_pion_p)
                    else:
                        continue   

                #Continues if and only if each daughter pion has found a match
                if len(matched_pion_plus) == len(tau_plus_daughters):
                    tagged_num += 1

            #         num_taus_fully_reconstructed = num_taus_fully_reconstructed + 1

                    pi1_plus_lv = ROOT.TLorentzVector()
                    pi2_plus_lv = ROOT.TLorentzVector()
                    pi3_plus_lv = ROOT.TLorentzVector()
                    neutrino_plus_lv = ROOT.TLorentzVector()

                    pi1_plus_lv.SetPtEtaPhiM(matched_pion_plus[0].pt(), matched_pion_plus[0].eta(), matched_pion_plus[0].phi(), 0.139)
                    pi2_plus_lv.SetPtEtaPhiM(matched_pion_plus[1].pt(), matched_pion_plus[1].eta(), matched_pion_plus[1].phi(), 0.139)
                    pi3_plus_lv.SetPtEtaPhiM(matched_pion_plus[2].pt(), matched_pion_plus[2].eta(), matched_pion_plus[2].phi(), 0.139)
                    neutrino_plus_lv.SetPtEtaPhiM(tau_plus_neutrino[0].pt(), tau_plus_neutrino[0].eta(), tau_plus_neutrino[0].phi(), 0)

            #         tau_plus_no_neutrino_lv = pi1_plus_lv + pi2_plus_lv + pi3_plus_lv
            #         tau_plus_with_neutrino_lv = pi1_plus_lv + pi2_plus_lv + pi3_plus_lv + neutrino_plus_lv

            #         tau_no_neutrino_mass = tau_plus_no_neutrino_lv.M()
            #         tau_with_neutrino_mass = tau_plus_with_neutrino_lv.M()

            #         #switching nomenclature here to tau and antitau

                    pi1_pt = pi1_plus_lv.Pt()
                    pi1_eta = pi1_plus_lv.Eta()
                    pi1_phi = pi1_plus_lv.Phi()

                    pi2_pt = pi2_plus_lv.Pt()
                    pi2_eta = pi2_plus_lv.Eta()
                    pi2_phi = pi2_plus_lv.Phi()

                    pi3_pt = pi3_plus_lv.Pt()
                    pi3_eta = pi3_plus_lv.Eta()
                    pi3_phi = pi3_plus_lv.Phi()

                    neu_pt = neutrino_plus_lv.Pt()
                    neu_eta = neutrino_plus_lv.Eta()
                    neu_phi = neutrino_plus_lv.Phi()

            # ### Gen Level Info Saving ###

            #         # gen_tau_lv = ROOT.TLorentzVector()
            #         # gen_pi1_from_tau_lv = ROOT.TLorentzVector()
            #         # gen_pi2_from_tau_lv = ROOT.TLorentzVector()
            #         # gen_pi3_from_tau_lv = ROOT.TLorentzVector()

            #         # gen_tau_lv.SetPtEtaPhiM(tau_from_upsilon.pt(), tau_from_upsilon.eta(), tau_from_upsilon.phi(), tau_from_upsilon.mass())
            #         # gen_pi1_from_tau_lv.SetPtEtaPhiM(tau_plus_daughters[0].pt(), tau_plus_daughters[0].eta(), tau_plus_daughters[0].phi(), tau_plus_daughters[0].mass())
            #         # gen_pi2_from_tau_lv.SetPtEtaPhiM(tau_plus_daughters[1].pt(), tau_plus_daughters[1].eta(), tau_plus_daughters[1].phi(), tau_plus_daughters[1].mass())
            #         # gen_pi3_from_tau_lv.SetPtEtaPhiM(tau_plus_daughters[2].pt(), tau_plus_daughters[2].eta(), tau_plus_daughters[2].phi(), tau_plus_daughters[2].mass())

            #         # gen_pi1_from_tau_pt = gen_pi1_from_tau_lv.Pt()
            #         # gen_pi1_from_tau_eta = gen_pi1_from_tau_lv.Eta()
            #         # gen_pi1_from_tau_phi = gen_pi1_from_tau_lv.Phi()

            #         # gen_pi2_from_tau_pt = gen_pi2_from_tau_lv.Pt()
            #         # gen_pi2_from_tau_eta = gen_pi2_from_tau_lv.Eta()
            #         # gen_pi2_from_tau_phi = gen_pi2_from_tau_lv.Phi()

            #         # gen_pi3_from_tau_pt = gen_pi3_from_tau_lv.Pt()
            #         # gen_pi3_from_tau_eta = gen_pi3_from_tau_lv.Eta()
            #         # gen_pi3_from_tau_phi = gen_pi3_from_tau_lv.Phi()

            #         # gen_tau_pt = gen_tau_lv.Pt()
            #         # gen_tau_eta = gen_tau_lv.Eta()
            #         # gen_tau_phi = gen_tau_lv.Phi()


            #         # column_names = ['pi1_pt', 'pi1_eta', 'pi1_phi','pi2_pt', 'pi2_eta', 'pi2_phi', 'pi3_pt', 'pi3_eta', 'pi3_phi',
            #         #             'neu_pt','neu_eta','neu_phi', 'sign']

                    add_row = [pi1_pt, pi1_eta, pi1_phi, pi2_pt, pi2_eta, pi2_phi, pi3_pt, pi3_eta, pi3_phi, 
                                    neu_pt, neu_eta, neu_phi, 'p', 'n']

            #         # gen_column_names_only_plus = ['gen_pi1_from_tau_pt', 'gen_pi1_from_tau_eta', 'gen_pi1_from_tau_phi', 'gen_pi2_from_tau_pt', 'gen_pi2_from_tau_eta', 'gen_pi2_from_tau_phi', 'gen_pi3_from_tau_pt', 'gen_pi3_from_tau_eta', 'gen_pi3_from_tau_phi',
            #         #         'gen_tau_pt', 'gen_tau_eta', 'gen_tau_phi']

            #         # add_gen_row = [gen_pi1_from_tau_pt, gen_pi1_from_tau_eta, gen_pi1_from_tau_phi, gen_pi2_from_tau_pt, gen_pi2_from_tau_eta, gen_pi2_from_tau_phi, gen_pi3_from_tau_pt, gen_pi3_from_tau_eta, gen_pi3_from_tau_phi,
            #         #         gen_tau_pt, gen_tau_eta, gen_tau_phi]

                    add_row_df = pd.DataFrame([add_row], columns = df_toUse.columns)
                    df_toUse = pd.concat([df_toUse, add_row_df], ignore_index=True)

            #         # add_gen_row_df = pd.DataFrame([add_gen_row], columns = df_matched_gen_info_tau_only.columns)
            #         # df_matched_gen_info_tau_only = pd.concat([df_matched_gen_info_tau_only, add_gen_row_df], ignore_index=True)    

                # else:
                #     num_taus_not_reconstructed = num_taus_not_reconstructed + 1

            #Case in which antitau decays properly, but tau does not    
            if good_taum and not good_taup:
                print('were wrong')
                proper_decay_number += 1

                proper_antitau_decay_only = proper_antitau_decay_only + 1

                #Save gen pion information regardless of whether it is matched to a reco level pion
                # gen_pi1_minus_lv_unmatched = ROOT.TLorentzVector()
                # gen_pi2_minus_lv_unmatched = ROOT.TLorentzVector()
                # gen_pi3_minus_lv_unmatched = ROOT.TLorentzVector()

                # gen_pi1_minus_lv_unmatched.SetPtEtaPhiM(tau_minus_daughters[0].pt(), tau_minus_daughters[0].eta(), tau_minus_daughters[0].phi(), 0.139)
                # gen_pi2_minus_lv_unmatched.SetPtEtaPhiM(tau_minus_daughters[1].pt(), tau_minus_daughters[1].eta(), tau_minus_daughters[1].phi(), 0.139)
                # gen_pi3_minus_lv_unmatched.SetPtEtaPhiM(tau_minus_daughters[2].pt(), tau_minus_daughters[2].eta(), tau_minus_daughters[2].phi(), 0.139)

                # gen_pi1_from_antitau_pt = gen_pi1_minus_lv_unmatched.Pt()
                # gen_pi1_from_antitau_eta = gen_pi1_minus_lv_unmatched.Eta()
                # gen_pi1_from_antitau_phi = gen_pi1_minus_lv_unmatched.Phi()

                # gen_pi2_from_antitau_pt = gen_pi2_minus_lv_unmatched.Pt()
                # gen_pi2_from_antitau_eta = gen_pi2_minus_lv_unmatched.Eta()
                # gen_pi2_from_antitau_phi = gen_pi2_minus_lv_unmatched.Phi()

                # gen_pi3_from_antitau_pt = gen_pi3_minus_lv_unmatched.Pt()
                # gen_pi3_from_antitau_eta = gen_pi3_minus_lv_unmatched.Eta()                        
                # gen_pi3_from_antitau_phi = gen_pi3_minus_lv_unmatched.Phi()


                # add_gen_row_tau_minus = [gen_pi1_from_antitau_pt, gen_pi1_from_antitau_eta, gen_pi1_from_antitau_phi, gen_pi2_from_antitau_pt, gen_pi2_from_antitau_eta, gen_pi2_from_antitau_phi, gen_pi3_from_antitau_pt, gen_pi3_from_antitau_eta, gen_pi3_from_antitau_phi]
                # add_gen_row_df_tau_minus = pd.DataFrame([add_gen_row_tau_minus], columns = df_unmatched_gen_info_antitau_only.columns)


                # df_unmatched_gen_info_antitau_only = pd.concat([df_unmatched_gen_info_antitau_only, add_gen_row_df_tau_minus], ignore_index=True)   


                #Start matching gen pions to reco pions for antitau
                    
                antitau_from_upsilon = tau_minus_keep
                matched_pion_minus = []

                for gen_pion_minus in tau_minus_daughters:
                    min_deltaR_minus = 999
                    match = False
                    for reco_particle in pf_particles:
                        if reco_particle.pdgId() == gen_pion_minus.pdgId():
                            reco_pions += 1

                            reco_lv = ROOT.TLorentzVector() 
                            reco_lv.SetPtEtaPhiM(reco_particle.pt(), reco_particle.eta(), reco_particle.phi(), reco_particle.mass())

                            gen_lv_minus = ROOT.TLorentzVector()
                            gen_lv_minus.SetPtEtaPhiM(gen_pion_minus.pt(), gen_pion_minus.eta(), gen_pion_minus.phi(), gen_pion_minus.mass())

                            deltaR_minus = gen_lv_minus.DeltaR(reco_lv)
                            deltaPT_minus = (reco_lv.Pt() - gen_lv_minus.Pt()) / gen_lv_minus.Pt()

                            if abs(deltaR_minus) < 0.1 and abs(deltaPT_minus) < 0.3 and abs(deltaR_minus) < min_deltaR_minus and abs(reco_particle.eta()) < 2.5 and reco_particle not in matched_pion_minus and reco_particle not in matched_pion_plus:
                                min_deltaR_minus = deltaR_minus
                                matched_pion_m = reco_particle
                                match = True
                    if match:
                        matched_pion_minus.append(matched_pion_m)
                    else: 
                        continue

                #Continues if and only if each daughter pion has found a match
                if len(matched_pion_minus) == len(tau_minus_daughters):
                    tagged_num += 1

            #         num_antitaus_fully_reconstructed = num_antitaus_fully_reconstructed + 1

                    pi1_minus_lv = ROOT.TLorentzVector()
                    pi2_minus_lv = ROOT.TLorentzVector()
                    pi3_minus_lv = ROOT.TLorentzVector()
                    neutrino_minus_lv = ROOT.TLorentzVector()

                    pi1_minus_lv.SetPtEtaPhiM(matched_pion_minus[0].pt(), matched_pion_minus[0].eta(), matched_pion_minus[0].phi(), 0.139)
                    pi2_minus_lv.SetPtEtaPhiM(matched_pion_minus[1].pt(), matched_pion_minus[1].eta(), matched_pion_minus[1].phi(), 0.139)
                    pi3_minus_lv.SetPtEtaPhiM(matched_pion_minus[2].pt(), matched_pion_minus[2].eta(), matched_pion_minus[2].phi(), 0.139)
                    neutrino_minus_lv.SetPtEtaPhiM(tau_minus_neutrino[0].pt(), tau_minus_neutrino[0].eta(), tau_minus_neutrino[0].phi(), 0)

            #         tau_minus_no_neutrino_lv = pi1_minus_lv + pi2_minus_lv + pi3_minus_lv
            #         tau_minus_with_neutrino_lv = pi1_minus_lv + pi2_minus_lv + pi3_minus_lv + neutrino_minus_lv

            #         antitau_no_neutrino_mass = tau_minus_no_neutrino_lv.M()
            #         antitau_with_neutrino_mass = tau_minus_with_neutrino_lv.M()

            #         #switching nomenclature here to tau and antitau

                    pi1_from_antitau_pt = pi1_minus_lv.Pt()
                    pi1_from_antitau_eta = pi1_minus_lv.Eta()
                    pi1_from_antitau_phi = pi1_minus_lv.Phi()

                    pi2_from_antitau_pt = pi2_minus_lv.Pt()
                    pi2_from_antitau_eta = pi2_minus_lv.Eta()
                    pi2_from_antitau_phi = pi2_minus_lv.Phi()

                    pi3_from_antitau_pt = pi3_minus_lv.Pt()
                    pi3_from_antitau_eta = pi3_minus_lv.Eta()
                    pi3_from_antitau_phi = pi3_minus_lv.Phi()

                    neutrino_from_antitau_pt = neutrino_minus_lv.Pt()
                    neutrino_from_antitau_eta = neutrino_minus_lv.Eta()
                    neutrino_from_antitau_phi = neutrino_minus_lv.Phi()

            # ### Gen Level Info Saving ###

            #         # gen_antitau_lv = ROOT.TLorentzVector()
            #         # gen_pi1_from_antitau_lv = ROOT.TLorentzVector()
            #         # gen_pi2_from_antitau_lv = ROOT.TLorentzVector()
            #         # gen_pi3_from_antitau_lv = ROOT.TLorentzVector()

            #         # gen_antitau_lv.SetPtEtaPhiM(antitau_from_upsilon.pt(), antitau_from_upsilon.eta(), antitau_from_upsilon.phi(), antitau_from_upsilon.mass())
            #         # gen_pi1_from_antitau_lv.SetPtEtaPhiM(tau_minus_daughters[0].pt(), tau_minus_daughters[0].eta(), tau_minus_daughters[0].phi(), tau_minus_daughters[0].mass())
            #         # gen_pi2_from_antitau_lv.SetPtEtaPhiM(tau_minus_daughters[1].pt(), tau_minus_daughters[1].eta(), tau_minus_daughters[1].phi(), tau_minus_daughters[1].mass())
            #         # gen_pi3_from_antitau_lv.SetPtEtaPhiM(tau_minus_daughters[2].pt(), tau_minus_daughters[2].eta(), tau_minus_daughters[2].phi(), tau_minus_daughters[2].mass())

            #         # gen_pi1_from_antitau_pt = gen_pi1_from_antitau_lv.Pt()
            #         # gen_pi1_from_antitau_eta = gen_pi1_from_antitau_lv.Eta()
            #         # gen_pi1_from_antitau_phi = gen_pi1_from_antitau_lv.Phi()

            #         # gen_pi2_from_antitau_pt = gen_pi2_from_antitau_lv.Pt()
            #         # gen_pi2_from_antitau_eta = gen_pi2_from_antitau_lv.Eta()
            #         # gen_pi2_from_antitau_phi = gen_pi2_from_antitau_lv.Phi()

            #         # gen_pi3_from_antitau_pt = gen_pi3_from_antitau_lv.Pt()
            #         # gen_pi3_from_antitau_eta = gen_pi3_from_antitau_lv.Eta()
            #         # gen_pi3_from_antitau_phi = gen_pi3_from_antitau_lv.Phi()

            #         # gen_antitau_pt = gen_antitau_lv.Pt()
            #         # gen_antitau_eta = gen_antitau_lv.Eta()
            #         # gen_antitau_phi = gen_antitau_lv.Phi()


                    # column_names_only_minus = ['pi1_pt', 'pi1_eta', 'pi1_phi','pi2_pt', 'pi2_eta', 'pi2_phi', 'pi3_pt', 'pi3_eta', 'pi3_phi',
                    #             'neu_pt','neu_eta','neu_phi', 'sign']

                    add_row = [pi1_from_antitau_pt, pi1_from_antitau_eta, pi1_from_antitau_phi, pi2_from_antitau_pt, pi2_from_antitau_eta, pi2_from_antitau_phi, pi3_from_antitau_pt, pi3_from_antitau_eta, pi3_from_antitau_phi,
                                neutrino_from_antitau_pt, neutrino_from_antitau_eta, neutrino_from_antitau_phi, 'm', 'n']

                    # gen_column_names_only_minus = ['gen_pi1_from_antitau_pt', 'gen_pi1_from_antitau_eta', 'gen_pi1_from_antitau_phi', 'gen_pi2_from_antitau_pt', 'gen_pi2_from_antitau_eta', 'gen_pi2_from_antitau_phi', 'gen_pi3_from_antitau_pt', 'gen_pi3_from_antitau_eta', 'gen_pi3_from_antitau_phi',
                    #         'gen_antitau_pt', 'gen_antitau_eta', 'gen_antitau_phi']

                    # add_gen_row = [gen_pi1_from_antitau_pt, gen_pi1_from_antitau_eta, gen_pi1_from_antitau_phi, gen_pi2_from_antitau_pt, gen_pi2_from_antitau_eta, gen_pi2_from_antitau_phi, gen_pi3_from_antitau_pt, gen_pi3_from_antitau_eta, gen_pi3_from_antitau_phi,
                    #         gen_antitau_pt, gen_antitau_eta, gen_antitau_phi]

                    add_row_df = pd.DataFrame([add_row], columns = df_toUse.columns)
                    df_toUse = pd.concat([df_toUse, add_row_df], ignore_index=True)

                    # add_gen_row_df = pd.DataFrame([add_gen_row], columns = df_matched_gen_info_antitau_only.columns)
                    # df_matched_gen_info_antitau_only = pd.concat([df_matched_gen_info_antitau_only, add_gen_row_df], ignore_index=True) 


                # else:
                #     num_antitaus_not_reconstructed = num_antitaus_not_reconstructed + 1
            
            
            if good_taum and good_taup:
                proper_decay_number = proper_decay_number + 2

                #Entered if both daughters of the Upsilon particle decay properly
                both_proper_decay = both_proper_decay + 1

                #Add Generated info for tau pions
                gen_pi1_plus_lv_unmatched = ROOT.TLorentzVector()
                gen_pi2_plus_lv_unmatched = ROOT.TLorentzVector()
                gen_pi3_plus_lv_unmatched = ROOT.TLorentzVector()

                gen_pi1_plus_lv_unmatched.SetPtEtaPhiM(tau_plus_daughters[0].pt(), tau_plus_daughters[0].eta(), tau_plus_daughters[0].phi(), 0.139)
                gen_pi2_plus_lv_unmatched.SetPtEtaPhiM(tau_plus_daughters[1].pt(), tau_plus_daughters[1].eta(), tau_plus_daughters[1].phi(), 0.139)
                gen_pi3_plus_lv_unmatched.SetPtEtaPhiM(tau_plus_daughters[2].pt(), tau_plus_daughters[2].eta(), tau_plus_daughters[2].phi(), 0.139)

                gen_pi1_from_tau_pt = gen_pi1_plus_lv_unmatched.Pt()
                gen_pi1_from_tau_eta = gen_pi1_plus_lv_unmatched.Eta()
                gen_pi1_from_tau_phi = gen_pi1_plus_lv_unmatched.Phi()

                gen_pi2_from_tau_pt = gen_pi2_plus_lv_unmatched.Pt()
                gen_pi2_from_tau_eta = gen_pi2_plus_lv_unmatched.Eta()
                gen_pi2_from_tau_phi = gen_pi2_plus_lv_unmatched.Phi()

                gen_pi3_from_tau_pt = gen_pi3_plus_lv_unmatched.Pt()
                gen_pi3_from_tau_eta = gen_pi3_plus_lv_unmatched.Eta()                        
                gen_pi3_from_tau_phi = gen_pi3_plus_lv_unmatched.Phi()

                add_gen_row_taup = [gen_pi1_from_tau_pt, gen_pi1_from_tau_eta, gen_pi1_from_tau_phi, gen_pi2_from_tau_pt, gen_pi2_from_tau_eta, gen_pi2_from_tau_phi, gen_pi3_from_tau_pt, gen_pi3_from_tau_eta, gen_pi3_from_tau_phi]
                add_gen_row_df_taup = pd.DataFrame([add_gen_row_taup], columns = df_unmatched_gen_info_tau_only.columns)
                df_unmatched_gen_info_tau_only = pd.concat([df_unmatched_gen_info_tau_only, add_gen_row_df_taup], ignore_index=True) 

                #Add Generated info for the antitau pions
                gen_pi1_minus_lv_unmatched = ROOT.TLorentzVector()
                gen_pi2_minus_lv_unmatched = ROOT.TLorentzVector()
                gen_pi3_minus_lv_unmatched = ROOT.TLorentzVector()

                gen_pi1_minus_lv_unmatched.SetPtEtaPhiM(tau_minus_daughters[0].pt(), tau_minus_daughters[0].eta(), tau_minus_daughters[0].phi(), 0.139)
                gen_pi2_minus_lv_unmatched.SetPtEtaPhiM(tau_minus_daughters[1].pt(), tau_minus_daughters[1].eta(), tau_minus_daughters[1].phi(), 0.139)
                gen_pi3_minus_lv_unmatched.SetPtEtaPhiM(tau_minus_daughters[2].pt(), tau_minus_daughters[2].eta(), tau_minus_daughters[2].phi(), 0.139)

                gen_pi1_from_antitau_pt = gen_pi1_minus_lv_unmatched.Pt()
                gen_pi1_from_antitau_eta = gen_pi1_minus_lv_unmatched.Eta()
                gen_pi1_from_antitau_phi = gen_pi1_minus_lv_unmatched.Phi()

                gen_pi2_from_antitau_pt = gen_pi2_minus_lv_unmatched.Pt()
                gen_pi2_from_antitau_eta = gen_pi2_minus_lv_unmatched.Eta()
                gen_pi2_from_antitau_phi = gen_pi2_minus_lv_unmatched.Phi()

                gen_pi3_from_antitau_pt = gen_pi3_minus_lv_unmatched.Pt()
                gen_pi3_from_antitau_eta = gen_pi3_minus_lv_unmatched.Eta()                        
                gen_pi3_from_antitau_phi = gen_pi3_minus_lv_unmatched.Phi()

                add_gen_row_taum = [gen_pi1_from_antitau_pt, gen_pi1_from_antitau_eta, gen_pi1_from_antitau_phi, gen_pi2_from_antitau_pt, gen_pi2_from_antitau_eta, gen_pi2_from_antitau_phi, gen_pi3_from_antitau_pt, gen_pi3_from_antitau_eta, gen_pi3_from_antitau_phi]
                add_gen_row_df_taum = pd.DataFrame([add_gen_row_taum], columns = df_unmatched_gen_info_tau_only.columns)
                df_unmatched_gen_info_tau_only = pd.concat([df_unmatched_gen_info_tau_only, add_gen_row_df_taum], ignore_index=True)                     
    
    df_unmatched_gen_info_tau_only.to_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/weird_eta_problems_15GeV.csv')
                #Matching gen to reco now
                # antitau_from_upsilon = tau_minus_keep
                # tau_from_upsilon = tau_plus_keep
                # matched_pion_plus = []
                # matched_pion_minus = []
                # match = False
                # for gen_pion_plus in tau_plus_daughters:
                #     min_deltaR_plus = 999
                #     match = False
                #     for reco_particle in pf_particles:
                #         if reco_particle.pdgId() == gen_pion_plus.pdgId():
                #             reco_pions += 1

                #             reco_lv = ROOT.TLorentzVector() 
                #             reco_lv.SetPtEtaPhiM(reco_particle.pt(), reco_particle.eta(), reco_particle.phi(), reco_particle.mass())

                #             gen_lv_plus = ROOT.TLorentzVector()
                #             gen_lv_plus.SetPtEtaPhiM(gen_pion_plus.pt(), gen_pion_plus.eta(), gen_pion_plus.phi(), gen_pion_plus.mass())

                #             deltaR_plus = gen_lv_plus.DeltaR(reco_lv)
                #             deltaPT_plus = (reco_lv.Pt() - gen_lv_plus.Pt()) / gen_lv_plus.Pt()
        
                #             if abs(deltaR_plus) < 0.1 and abs(deltaPT_plus) < 0.3 and abs(deltaR_plus) < min_deltaR_plus and abs(reco_particle.eta()) < 2.5 and reco_particle not in matched_pion_plus and reco_particle not in matched_pion_minus:
                #                 min_deltaR_plus = deltaR_plus
                #                 matched_pion_p = reco_particle
                #                 match = True
                #     if match:
                #         matched_pion_plus.append(matched_pion_p)
                #     else:
                #         continue
                
                # if match: tagged_num += 1

                # for gen_pion_minus in tau_minus_daughters:
                #     min_deltaR_minus = 999
                #     match = False
                #     for reco_particle in pf_particles:
                #         if reco_particle.pdgId() == gen_pion_minus.pdgId():
                #             reco_pions += 1

                #             reco_lv = ROOT.TLorentzVector() 
                #             reco_lv.SetPtEtaPhiM(reco_particle.pt(), reco_particle.eta(), reco_particle.phi(), reco_particle.mass())

                #             gen_lv_minus = ROOT.TLorentzVector()
                #             gen_lv_minus.SetPtEtaPhiM(gen_pion_minus.pt(), gen_pion_minus.eta(), gen_pion_minus.phi(), gen_pion_minus.mass())

                #             deltaR_minus = gen_lv_minus.DeltaR(reco_lv)
                #             deltaPT_minus = (reco_lv.Pt() - gen_lv_minus.Pt()) / gen_lv_minus.Pt()

                #             if abs(deltaR_minus) < 0.1 and abs(deltaPT_minus) < 0.3 and abs(deltaR_minus) < min_deltaR_minus and abs(reco_particle.eta()) < 2.5 and reco_particle not in matched_pion_minus and reco_particle not in matched_pion_plus:
                #                 min_deltaR_minus = deltaR_minus
                #                 matched_pion_m = reco_particle
                #                 match = True
                #     if match:
                #         matched_pion_minus.append(matched_pion_m)
                #     else: 
                #         continue

                # if match: tagged_num += 1
# print(tagged_num)

                if len(matched_pion_plus) == len(tau_plus_daughters) == len(matched_pion_minus) == len(tau_minus_daughters):
                    upsilon_ready = True


                if len(matched_pion_plus) == len(tau_plus_daughters):
                    # tagged_num += 1

            #         num_taus_fully_reconstructed = num_taus_fully_reconstructed + 1

                    pi1_plus_lv = ROOT.TLorentzVector()
                    pi2_plus_lv = ROOT.TLorentzVector()
                    pi3_plus_lv = ROOT.TLorentzVector()
                    neutrino_plus_lv = ROOT.TLorentzVector()

                    pi1_plus_lv.SetPtEtaPhiM(matched_pion_plus[0].pt(), matched_pion_plus[0].eta(), matched_pion_plus[0].phi(), 0.139)
                    pi2_plus_lv.SetPtEtaPhiM(matched_pion_plus[1].pt(), matched_pion_plus[1].eta(), matched_pion_plus[1].phi(), 0.139)
                    pi3_plus_lv.SetPtEtaPhiM(matched_pion_plus[2].pt(), matched_pion_plus[2].eta(), matched_pion_plus[2].phi(), 0.139)
                    neutrino_plus_lv.SetPtEtaPhiM(tau_plus_neutrino[0].pt(), tau_plus_neutrino[0].eta(), tau_plus_neutrino[0].phi(), 0)

                    # pi1_minus_lv = ROOT.TLorentzVector()
                    # pi2_minus_lv = ROOT.TLorentzVector()
                    # pi3_minus_lv = ROOT.TLorentzVector()
                    # neutrino_minus_lv = ROOT.TLorentzVector()

                    # pi1_minus_lv.SetPtEtaPhiM(matched_pion_minus[0].pt(), matched_pion_minus[0].eta(), matched_pion_minus[0].phi(), 0.139)
                    # pi2_minus_lv.SetPtEtaPhiM(matched_pion_minus[1].pt(), matched_pion_minus[1].eta(), matched_pion_minus[1].phi(), 0.139)
                    # pi3_minus_lv.SetPtEtaPhiM(matched_pion_minus[2].pt(), matched_pion_minus[2].eta(), matched_pion_minus[2].phi(), 0.139)
                    # neutrino_minus_lv.SetPtEtaPhiM(tau_minus_neutrino[0].pt(), tau_minus_neutrino[0].eta(), tau_minus_neutrino[0].phi(), 0)

            #         tau_plus_no_neutrino_lv = pi1_plus_lv + pi2_plus_lv + pi3_plus_lv
            #         tau_plus_with_neutrino_lv = pi1_plus_lv + pi2_plus_lv + pi3_plus_lv + neutrino_plus_lv

            #         # tau_minus_no_neutrino_lv = pi1_minus_lv + pi2_minus_lv + pi3_minus_lv
            #         # tau_minus_with_neutrino_lv = pi1_minus_lv + pi2_minus_lv + pi3_minus_lv + neutrino_minus_lv

            #         # upsilon_no_neutrino_lv = tau_plus_no_neutrino_lv + tau_minus_no_neutrino_lv
            #         # upsilon_with_neutrino_lv = tau_plus_with_neutrino_lv + tau_minus_with_neutrino_lv

            #         tau_no_neutrino_mass = tau_plus_no_neutrino_lv.M()
            #         tau_with_neutrino_mass = tau_plus_with_neutrino_lv.M()

            #         # antitau_no_neutrino_mass = tau_plus_no_neutrino_lv.M()
            #         # antitau_with_neutrino_mass = tau_plus_with_neutrino_lv.M()

            #         # upsilon_no_neutrino_mass = upsilon_no_neutrino_lv.M()
            #         # upsilon_with_neutrino_mass = upsilon_with_neutrino_lv.M()

            #         #switching nomenclature here to tau and antitau

                    pi1_from_tau_pt = pi1_plus_lv.Pt()
                    pi1_from_tau_eta = pi1_plus_lv.Eta()
                    pi1_from_tau_phi = pi1_plus_lv.Phi()

                    pi2_from_tau_pt = pi2_plus_lv.Pt()
                    pi2_from_tau_eta = pi2_plus_lv.Eta()
                    pi2_from_tau_phi = pi2_plus_lv.Phi()

                    pi3_from_tau_pt = pi3_plus_lv.Pt()
                    pi3_from_tau_eta = pi3_plus_lv.Eta()
                    pi3_from_tau_phi = pi3_plus_lv.Phi()

                    neutrino_from_tau_pt = neutrino_plus_lv.Pt()
                    neutrino_from_tau_eta = neutrino_plus_lv.Eta()
                    neutrino_from_tau_phi = neutrino_plus_lv.Phi()

                    # pi1_from_antitau_pt = pi1_minus_lv.Pt()
                    # pi1_from_antitau_eta = pi1_minus_lv.Eta()
                    # pi1_from_antitau_phi = pi1_minus_lv.Phi()

                    # pi2_from_antitau_pt = pi2_minus_lv.Pt()
                    # pi2_from_antitau_eta = pi2_minus_lv.Eta()
                    # pi2_from_antitau_phi = pi2_minus_lv.Phi()

                    # pi3_from_antitau_pt = pi3_minus_lv.Pt()
                    # pi3_from_antitau_eta = pi3_minus_lv.Eta()
                    # pi3_from_antitau_phi = pi3_minus_lv.Phi()

                    # neutrino_from_antitau_pt = neutrino_minus_lv.Pt()
                    # neutrino_from_antitau_eta = neutrino_minus_lv.Eta()
                    # neutrino_from_antitau_phi = neutrino_minus_lv.Phi()

            # ### Gen Level Info Saving ###

            #         # gen_tau_lv = ROOT.TLorentzVector()
            #         # gen_pi1_from_tau_lv = ROOT.TLorentzVector()
            #         # gen_pi2_from_tau_lv = ROOT.TLorentzVector()
            #         # gen_pi3_from_tau_lv = ROOT.TLorentzVector()

            #         # gen_tau_lv.SetPtEtaPhiM(tau_from_upsilon.pt(), tau_from_upsilon.eta(), tau_from_upsilon.phi(), tau_from_upsilon.mass())
            #         # gen_pi1_from_tau_lv.SetPtEtaPhiM(tau_plus_daughters[0].pt(), tau_plus_daughters[0].eta(), tau_plus_daughters[0].phi(), tau_plus_daughters[0].mass())
            #         # gen_pi2_from_tau_lv.SetPtEtaPhiM(tau_plus_daughters[1].pt(), tau_plus_daughters[1].eta(), tau_plus_daughters[1].phi(), tau_plus_daughters[1].mass())
            #         # gen_pi3_from_tau_lv.SetPtEtaPhiM(tau_plus_daughters[2].pt(), tau_plus_daughters[2].eta(), tau_plus_daughters[2].phi(), tau_plus_daughters[2].mass())

            #         # gen_pi1_from_tau_pt = gen_pi1_from_tau_lv.Pt()
            #         # gen_pi1_from_tau_eta = gen_pi1_from_tau_lv.Eta()
            #         # gen_pi1_from_tau_phi = gen_pi1_from_tau_lv.Phi()

            #         # gen_pi2_from_tau_pt = gen_pi2_from_tau_lv.Pt()
            #         # gen_pi2_from_tau_eta = gen_pi2_from_tau_lv.Eta()
            #         # gen_pi2_from_tau_phi = gen_pi2_from_tau_lv.Phi()

            #         # gen_pi3_from_tau_pt = gen_pi3_from_tau_lv.Pt()
            #         # gen_pi3_from_tau_eta = gen_pi3_from_tau_lv.Eta()
            #         # gen_pi3_from_tau_phi = gen_pi3_from_tau_lv.Phi()

            #         # gen_tau_pt = gen_tau_lv.Pt()
            #         # gen_tau_eta = gen_tau_lv.Eta()
            #         # gen_tau_phi = gen_tau_lv.Phi()

            #         # gen_antitau_lv = ROOT.TLorentzVector()
            #         # gen_pi1_from_antitau_lv = ROOT.TLorentzVector()
            #         # gen_pi2_from_antitau_lv = ROOT.TLorentzVector()
            #         # gen_pi3_from_antitau_lv = ROOT.TLorentzVector()

            #         # gen_antitau_lv.SetPtEtaPhiM(antitau_from_upsilon.pt(), antitau_from_upsilon.eta(), antitau_from_upsilon.phi(), antitau_from_upsilon.mass())
            #         # gen_pi1_from_antitau_lv.SetPtEtaPhiM(tau_minus_daughters[0].pt(), tau_minus_daughters[0].eta(), tau_minus_daughters[0].phi(), tau_minus_daughters[0].mass())
            #         # gen_pi2_from_antitau_lv.SetPtEtaPhiM(tau_minus_daughters[1].pt(), tau_minus_daughters[1].eta(), tau_minus_daughters[1].phi(), tau_minus_daughters[1].mass())
            #         # gen_pi3_from_antitau_lv.SetPtEtaPhiM(tau_minus_daughters[2].pt(), tau_minus_daughters[2].eta(), tau_minus_daughters[2].phi(), tau_minus_daughters[2].mass())

            #         # gen_pi1_from_antitau_pt = gen_pi1_from_antitau_lv.Pt()
            #         # gen_pi1_from_antitau_eta = gen_pi1_from_antitau_lv.Eta()
            #         # gen_pi1_from_antitau_phi = gen_pi1_from_antitau_lv.Phi()

            #         # gen_pi2_from_antitau_pt = gen_pi2_from_antitau_lv.Pt()
            #         # gen_pi2_from_antitau_eta = gen_pi2_from_antitau_lv.Eta()
            #         # gen_pi2_from_antitau_phi = gen_pi2_from_antitau_lv.Phi()

            #         # gen_pi3_from_antitau_pt = gen_pi3_from_antitau_lv.Pt()
            #         # gen_pi3_from_antitau_eta = gen_pi3_from_antitau_lv.Eta()
            #         # gen_pi3_from_antitau_phi = gen_pi3_from_antitau_lv.Phi()

            #         # gen_antitau_pt = gen_antitau_lv.Pt()
            #         # gen_antitau_eta = gen_antitau_lv.Eta()
            #         # gen_antitau_phi = gen_antitau_lv.Phi()

            #         # gen_upsilon = upsilon_list[0]
            #         # gen_upsilon_lv = ROOT.TLorentzVector()
            #         # gen_upsilon_lv.SetPtEtaPhiM(gen_upsilon.pt(), gen_upsilon.eta(), gen_upsilon.phi(), gen_upsilon.mass())

            #         # gen_upsilon_pt = gen_upsilon_lv.Pt()
            #         # gen_upsilon_eta = gen_upsilon_lv.Eta()
            #         # gen_upsilon_phi = gen_upsilon_lv.Phi()
            #         # gen_upsilon_mass = gen_upsilon_lv.M()

                    # column_names_only_plus = ['pi1_pt', 'pi1_eta', 'pi1_phi','pi2_pt', 'pi2_eta', 'pi2_phi', 'pi3_pt', 'pi3_eta', 'pi3_phi',
                    #             'neu_pt','neu_eta','neu_phi', 'sign']
                    if upsilon_ready:
                        add_row = [pi1_from_tau_pt, pi1_from_tau_eta, pi1_from_tau_phi, pi2_from_tau_pt, pi2_from_tau_eta, pi2_from_tau_phi, pi3_from_tau_pt, pi3_from_tau_eta, pi3_from_tau_phi,
                                neutrino_from_tau_pt, neutrino_from_tau_eta, neutrino_from_tau_phi, 'p', 'y']
                    else:
                        add_row = [pi1_from_tau_pt, pi1_from_tau_eta, pi1_from_tau_phi, pi2_from_tau_pt, pi2_from_tau_eta, pi2_from_tau_phi, pi3_from_tau_pt, pi3_from_tau_eta, pi3_from_tau_phi,
                                neutrino_from_tau_pt, neutrino_from_tau_eta, neutrino_from_tau_phi, 'p', 'n']

                    # gen_column_names_only_plus = ['gen_pi1_from_tau_pt', 'gen_pi1_from_tau_eta', 'gen_pi1_from_tau_phi', 'gen_pi2_from_tau_pt', 'gen_pi2_from_tau_eta', 'gen_pi2_from_tau_phi', 'gen_pi3_from_tau_pt', 'gen_pi3_from_tau_eta', 'gen_pi3_from_tau_phi',
                    #         'gen_tau_pt', 'gen_tau_eta', 'gen_tau_phi']

                    # add_gen_row = [gen_pi1_from_tau_pt, gen_pi1_from_tau_eta, gen_pi1_from_tau_phi, gen_pi2_from_tau_pt, gen_pi2_from_tau_eta, gen_pi2_from_tau_phi, gen_pi3_from_tau_pt, gen_pi3_from_tau_eta, gen_pi3_from_tau_phi,
                    #         gen_tau_pt, gen_tau_eta, gen_tau_phi]

                    add_row_df = pd.DataFrame([add_row], columns = df_toUse.columns)
                    df_toUse = pd.concat([df_toUse, add_row_df], ignore_index=True)

                    # add_gen_row_df = pd.DataFrame([add_gen_row], columns = df_matched_gen_info_tau_only.columns)
                    # df_matched_gen_info_tau_only = pd.concat([df_matched_gen_info_tau_only, add_gen_row_df], ignore_index=True)    

            #     # else:
            #     #     num_taus_not_reconstructed = num_taus_not_reconstructed + 1

                if len(matched_pion_minus) == len(tau_minus_daughters):
                    tagged_num += 1

            #         num_antitaus_fully_reconstructed = num_antitaus_fully_reconstructed + 1

                    pi1_minus_lv = ROOT.TLorentzVector()
                    pi2_minus_lv = ROOT.TLorentzVector()
                    pi3_minus_lv = ROOT.TLorentzVector()
                    neutrino_minus_lv = ROOT.TLorentzVector()

                    pi1_minus_lv.SetPtEtaPhiM(matched_pion_minus[0].pt(), matched_pion_minus[0].eta(), matched_pion_minus[0].phi(), 0.139)
                    pi2_minus_lv.SetPtEtaPhiM(matched_pion_minus[1].pt(), matched_pion_minus[1].eta(), matched_pion_minus[1].phi(), 0.139)
                    pi3_minus_lv.SetPtEtaPhiM(matched_pion_minus[2].pt(), matched_pion_minus[2].eta(), matched_pion_minus[2].phi(), 0.139)
                    neutrino_minus_lv.SetPtEtaPhiM(tau_minus_neutrino[0].pt(), tau_minus_neutrino[0].eta(), tau_minus_neutrino[0].phi(), 0)


            #         tau_minus_no_neutrino_lv = pi1_minus_lv + pi2_minus_lv + pi3_minus_lv
            #         tau_minus_with_neutrino_lv = pi1_minus_lv + pi2_minus_lv + pi3_minus_lv + neutrino_minus_lv

            #         # upsilon_no_neutrino_lv = tau_plus_no_neutrino_lv + tau_minus_no_neutrino_lv
            #         # upsilon_with_neutrino_lv = tau_plus_with_neutrino_lv + tau_minus_with_neutrino_lv

            #         # tau_no_neutrino_mass = tau_plus_no_neutrino_lv.M()
            #         # tau_with_neutrino_mass = tau_plus_with_neutrino_lv.M()

            #         antitau_no_neutrino_mass = tau_minus_no_neutrino_lv.M()
            #         antitau_with_neutrino_mass = tau_minus_with_neutrino_lv.M()

            #         # upsilon_no_neutrino_mass = upsilon_no_neutrino_lv.M()
            #         # upsilon_with_neutrino_mass = upsilon_with_neutrino_lv.M()

            #         #switching nomenclature here to tau and antitau

            #         # pi1_from_tau_pt = pi1_plus_lv.Pt()
            #         # pi1_from_tau_eta = pi1_plus_lv.Eta()
            #         # pi1_from_tau_phi = pi1_plus_lv.Phi()

            #         # pi2_from_tau_pt = pi2_plus_lv.Pt()
            #         # pi2_from_tau_eta = pi2_plus_lv.Eta()
            #         # pi2_from_tau_phi = pi2_plus_lv.Phi()

            #         # pi3_from_tau_pt = pi3_plus_lv.Pt()
            #         # pi3_from_tau_eta = pi3_plus_lv.Eta()
            #         # pi3_from_tau_phi = pi3_plus_lv.Phi()

            #         # neutrino_from_tau_pt = neutrino_plus_lv.Pt()
            #         # neutrino_from_tau_eta = neutrino_plus_lv.Eta()
            #         # neutrino_from_tau_phi = neutrino_plus_lv.Phi()

                    pi1_from_antitau_pt = pi1_minus_lv.Pt()
                    pi1_from_antitau_eta = pi1_minus_lv.Eta()
                    pi1_from_antitau_phi = pi1_minus_lv.Phi()

                    pi2_from_antitau_pt = pi2_minus_lv.Pt()
                    pi2_from_antitau_eta = pi2_minus_lv.Eta()
                    pi2_from_antitau_phi = pi2_minus_lv.Phi()

                    pi3_from_antitau_pt = pi3_minus_lv.Pt()
                    pi3_from_antitau_eta = pi3_minus_lv.Eta()
                    pi3_from_antitau_phi = pi3_minus_lv.Phi()

                    neutrino_from_antitau_pt = neutrino_minus_lv.Pt()
                    neutrino_from_antitau_eta = neutrino_minus_lv.Eta()
                    neutrino_from_antitau_phi = neutrino_minus_lv.Phi()

            # ### Gen Level Info Saving ###

            #         # gen_tau_lv = ROOT.TLorentzVector()
            #         # gen_pi1_from_tau_lv = ROOT.TLorentzVector()
            #         # gen_pi2_from_tau_lv = ROOT.TLorentzVector()
            #         # gen_pi3_from_tau_lv = ROOT.TLorentzVector()

            #         # gen_tau_lv.SetPtEtaPhiM(tau_from_upsilon.pt(), tau_from_upsilon.eta(), tau_from_upsilon.phi(), tau_from_upsilon.mass())
            #         # gen_pi1_from_tau_lv.SetPtEtaPhiM(tau_plus_daughters[0].pt(), tau_plus_daughters[0].eta(), tau_plus_daughters[0].phi(), tau_plus_daughters[0].mass())
            #         # gen_pi2_from_tau_lv.SetPtEtaPhiM(tau_plus_daughters[1].pt(), tau_plus_daughters[1].eta(), tau_plus_daughters[1].phi(), tau_plus_daughters[1].mass())
            #         # gen_pi3_from_tau_lv.SetPtEtaPhiM(tau_plus_daughters[2].pt(), tau_plus_daughters[2].eta(), tau_plus_daughters[2].phi(), tau_plus_daughters[2].mass())

            #         # gen_pi1_from_tau_pt = gen_pi1_from_tau_lv.Pt()
            #         # gen_pi1_from_tau_eta = gen_pi1_from_tau_lv.Eta()
            #         # gen_pi1_from_tau_phi = gen_pi1_from_tau_lv.Phi()

            #         # gen_pi2_from_tau_pt = gen_pi2_from_tau_lv.Pt()
            #         # gen_pi2_from_tau_eta = gen_pi2_from_tau_lv.Eta()
            #         # gen_pi2_from_tau_phi = gen_pi2_from_tau_lv.Phi()

            #         # gen_pi3_from_tau_pt = gen_pi3_from_tau_lv.Pt()
            #         # gen_pi3_from_tau_eta = gen_pi3_from_tau_lv.Eta()
            #         # gen_pi3_from_tau_phi = gen_pi3_from_tau_lv.Phi()

            #         # gen_tau_pt = gen_tau_lv.Pt()
            #         # gen_tau_eta = gen_tau_lv.Eta()
            #         # gen_tau_phi = gen_tau_lv.Phi()

            #         # gen_antitau_lv = ROOT.TLorentzVector()
            #         # gen_pi1_from_antitau_lv = ROOT.TLorentzVector()
            #         # gen_pi2_from_antitau_lv = ROOT.TLorentzVector()
            #         # gen_pi3_from_antitau_lv = ROOT.TLorentzVector()

            #         # gen_antitau_lv.SetPtEtaPhiM(antitau_from_upsilon.pt(), antitau_from_upsilon.eta(), antitau_from_upsilon.phi(), antitau_from_upsilon.mass())
            #         # gen_pi1_from_antitau_lv.SetPtEtaPhiM(tau_minus_daughters[0].pt(), tau_minus_daughters[0].eta(), tau_minus_daughters[0].phi(), tau_minus_daughters[0].mass())
            #         # gen_pi2_from_antitau_lv.SetPtEtaPhiM(tau_minus_daughters[1].pt(), tau_minus_daughters[1].eta(), tau_minus_daughters[1].phi(), tau_minus_daughters[1].mass())
            #         # gen_pi3_from_antitau_lv.SetPtEtaPhiM(tau_minus_daughters[2].pt(), tau_minus_daughters[2].eta(), tau_minus_daughters[2].phi(), tau_minus_daughters[2].mass())

            #         # gen_pi1_from_antitau_pt = gen_pi1_from_antitau_lv.Pt()
            #         # gen_pi1_from_antitau_eta = gen_pi1_from_antitau_lv.Eta()
            #         # gen_pi1_from_antitau_phi = gen_pi1_from_antitau_lv.Phi()

            #         # gen_pi2_from_antitau_pt = gen_pi2_from_antitau_lv.Pt()
            #         # gen_pi2_from_antitau_eta = gen_pi2_from_antitau_lv.Eta()
            #         # gen_pi2_from_antitau_phi = gen_pi2_from_antitau_lv.Phi()

            #         # gen_pi3_from_antitau_pt = gen_pi3_from_antitau_lv.Pt()
            #         # gen_pi3_from_antitau_eta = gen_pi3_from_antitau_lv.Eta()
            #         # gen_pi3_from_antitau_phi = gen_pi3_from_antitau_lv.Phi()

            #         # gen_antitau_pt = gen_antitau_lv.Pt()
            #         # gen_antitau_eta = gen_antitau_lv.Eta()
            #         # gen_antitau_phi = gen_antitau_lv.Phi()

            #         # gen_upsilon = upsilon_list[0]
            #         # gen_upsilon_lv = ROOT.TLorentzVector()
            #         # gen_upsilon_lv.SetPtEtaPhiM(gen_upsilon.pt(), gen_upsilon.eta(), gen_upsilon.phi(), gen_upsilon.mass())

            #         # gen_upsilon_pt = gen_upsilon_lv.Pt()
            #         # gen_upsilon_eta = gen_upsilon_lv.Eta()
            #         # gen_upsilon_phi = gen_upsilon_lv.Phi()
            #         # gen_upsilon_mass = gen_upsilon_lv.M()


            #         # column_names_only_minus = ['pi1_from_antitau_pt', 'pi1_from_antitau_eta', 'pi1_from_antitau_phi','pi2_from_antitau_pt', 'pi2_from_antitau_eta', 'pi2_from_antitau_phi', 'pi3_from_antitau_pt', 'pi3_from_antitau_eta', 'pi3_from_antitau_phi',
            #         #             'neutrino_from_antitau_pt','neutrino_from_antitau_eta','neutrino_from_antitau_phi', 'tau_no_neutrino_mass', 'tau_with_neutrino_mass']

                    if upsilon_ready:
                        add_row = [pi1_from_antitau_pt, pi1_from_antitau_eta, pi1_from_antitau_phi, pi2_from_antitau_pt, pi2_from_antitau_eta, pi2_from_antitau_phi, pi3_from_antitau_pt, pi3_from_antitau_eta, pi3_from_antitau_phi,
                                neutrino_from_antitau_pt, neutrino_from_antitau_eta, neutrino_from_antitau_phi, 'm', 'y']
                    else:
                        add_row = [pi1_from_antitau_pt, pi1_from_antitau_eta, pi1_from_antitau_phi, pi2_from_antitau_pt, pi2_from_antitau_eta, pi2_from_antitau_phi, pi3_from_antitau_pt, pi3_from_antitau_eta, pi3_from_antitau_phi,
                                neutrino_from_antitau_pt, neutrino_from_antitau_eta, neutrino_from_antitau_phi, 'm', 'n']

                    # gen_column_names_only_minus = ['gen_pi1_from_antitau_pt', 'gen_pi1_from_antitau_eta', 'gen_pi1_from_antitau_phi', 'gen_pi2_from_antitau_pt', 'gen_pi2_from_antitau_eta', 'gen_pi2_from_antitau_phi', 'gen_pi3_from_antitau_pt', 'gen_pi3_from_antitau_eta', 'gen_pi3_from_antitau_phi',
                    #         'gen_antitau_pt', 'gen_antitau_eta', 'gen_antitau_phi', 'sign']

                    # add_gen_row = [gen_pi1_from_antitau_pt, gen_pi1_from_antitau_eta, gen_pi1_from_antitau_phi, gen_pi2_from_antitau_pt, gen_pi2_from_antitau_eta, gen_pi2_from_antitau_phi, gen_pi3_from_antitau_pt, gen_pi3_from_antitau_eta, gen_pi3_from_antitau_phi,
                    #         gen_antitau_pt, gen_antitau_eta, gen_antitau_phi, 'm']

                    add_row_df = pd.DataFrame([add_row], columns = df_toUse.columns)
                    df_toUse = pd.concat([df_toUse, add_row_df], ignore_index=True)

                    # add_gen_row_df = pd.DataFrame([add_gen_row], columns = df_matched_gen_info_antitau_only.columns)
                    # df_matched_gen_info_antitau_only = pd.concat([df_matched_gen_info_antitau_only, add_gen_row_df], ignore_index=True) 
            #     # else:
            #     #     num_both_not_reconstructed = num_both_not_reconstructed + 1


print(proper_decay_number)
print(tagged_num)

df_toUse.to_csv(f'/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing/15GeV_for_tracking_purposes.csv')
    # # df_genInfo.to_csv(f'/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing/gen_information_{i}_miniaod.csv')
    # print(f'csv {i+20} saved')
        

    # tracking_info = [both_proper_decay, proper_tau_decay_only, proper_antitau_decay_only, no_proper_decays, num_taus_fully_reconstructed, num_taus_not_reconstructed, num_antitaus_fully_reconstructed, num_antitaus_not_reconstructed, num_both_fully_reconstructed, num_both_not_reconstructed]
    # add_tracking_row_df = pd.DataFrame([tracking_info], columns = df_tracking_frequency.columns)
    # df_tracking_frequency = pd.concat([df_tracking_frequency, add_tracking_row_df], ignore_index=True)

    # #Information that contains the matched gen info and the info to use by the model in the case that the upsilon decays into two proper taus
    # df_toUse_both.to_csv(f'/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing/individual_datasets_15GeV/both_proper_decay_info{i+20}.csv')
    # df_matched_gen_info_both.to_csv(f'/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing/individual_datasets_15GeV/gen_info_both_proper_decay{i+20}.csv')

    # #Information that contains the unmatched geenrated data for the pions for each of the tau and antitau
    # df_unmatched_gen_info_tau_only.to_csv(f'/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing/individual_datasets_15GeV/unmatched_gen_info_tau_only{i+20}.csv')
    # df_unmatched_gen_info_antitau_only.to_csv(f'/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing/individual_datasets_15GeV/unmatched_gen_info_antitau_only{i+20}.csv')

    # #Information that contains the matched gen info and the info to use by the model in the case that the upsilon decays into one proper tau
    # df_toUse_tau_only.to_csv(f'/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing/individual_datasets_15GeV/tau_proper_decay_info{i+20}.csv')
    # df_matched_gen_info_tau_only.to_csv(f'/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing/individual_datasets_15GeV/gen_info_tau_proper_decay{i+20}.csv')

    # #Information that contains the matched gen info and the info to use by the model in the case that the upsilon decays into one proper antitau
    # df_toUse_antitau_only.to_csv(f'/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing/individual_datasets_15GeV/anti_proper_decay_info{i+20}.csv')
    # df_matched_gen_info_antitau_only.to_csv(f'/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing/individual_datasets_15GeV/gen_info_antitau_proper_decay{i+20}.csv')

    # #Contains counts for each of the cases - see column names
    # df_tracking_frequency.to_csv(f'/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing/individual_datasets_15GeV/tracking_frequency{i+20}.csv')

    # print("We should begin with 100,000 taus from 50,000 upsilons.")
    # print("Yet only:")
    # print(massive_upsilon_count * 2)
    # print("taus come from a nominal mass upsilon.")
    # print("And only:")
    # print(proper_decay_number)
    # print("taus decay properly.")
    # print("Further, we where only able to tag:")
    # print(tagged_num)
    # print("of the taus. This is a much smaller number than 100,000.")

    # print("we have this many reco tracks (pions only)")
    # print(reco_pions)
    # print("we should have:")
    # print(proper_decay_number*3)
    # adnoio = proper_decay_number*3
    # print('^ that many good gen pions')
    # print('thus, we have:')
    # print(reco_pions/adnoio)
    # print('reco tracks (pions) per good gen pion for these 10 files')
    # print('and also we should have:')
    # print(reco_pions/50000)
    # print('^ that many reco tracks (pions) per total events (50000) for the 10 files')

'''
# Script to import MINIAODS, convert to pandas dataframes, and export as CSV's
# 
# IMPORTS
import ROOT # for 4-vector builds
from DataFormats.FWLite import Events, Handle # to open MiniAODs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse # parser for arguments

# MAIN
if __name__ == '__main__':
    # Argument parser and fixing the CMSSW version via the options container
    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('--f_in', default=f'40k_15GeV_Y_MINIAOD')
    #parser.add_argument('--maxEvents', default = 100)
    args = parser.parse_args()

    # file path
    path = '/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/40k_Varied_Y_Mass_Files'
    filename = path + '/' + args.f_in + '.root'

    from FWCore.ParameterSet.VarParsing import VarParsing # Needed to input the file
    options = VarParsing ('python')  
    options.inputFiles = [filename]
    options.maxEvents =  -1 # run on 10 events first, -1 for all the events
    options.parseArguments()

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

    # SET UP DFS
    gen_tau_only_column_names = ['gen_pi1_from_tau_pt', 'gen_pi1_from_tau_eta', 'gen_pi1_from_tau_phi',
                                 'gen_pi2_from_tau_pt', 'gen_pi2_from_tau_eta', 'gen_pi2_from_tau_phi',
                                 'gen_pi3_from_tau_pt', 'gen_pi3_from_tau_eta', 'gen_pi3_from_tau_phi']
    df_gen_pions = pd.DataFrame(columns = gen_tau_only_column_names)
    column_names = ['pi1_pt', 'pi1_eta', 'pi1_phi',
                    'pi2_pt', 'pi2_eta', 'pi2_phi',
                    'pi3_pt', 'pi3_eta', 'pi3_phi',
                    'neu_pt','neu_eta','neu_phi', 'sign', 'upsilon?']
    df_toUse = pd.DataFrame(columns = column_names)

    for event in events: # Loops over all the events sepcified with maxEvents

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

        def isAncestor(a,p):
            if not p: 
                return False

            if a == p: 
                return True

            for i in range(0, p.numberOfMothers()):
                if isAncestor(a,p.mother(i)): 
                    return True

        tau_plus_daughters = []
        tau_plus_neutrino = []
        tau_minus_daughters = []
        tau_minus_neutrino = []
        tau_plus_neutral_check = 0
        tau_minus_neutral_check = 0
        good_taup = False
        good_taum = False

        if len(upsilon_list) > 1: raise ValueError('should only be one upsilon')
        else: upsilon = upsilon_list[0]

        for tau_plus in tau_plus_list:
            if isAncestor(upsilon, tau_plus.mother(0)):
                taup_daughter = True
                for pi_plus in pi_plus_list:
                    if isAncestor(tau_plus, pi_plus.mother(0)):
                        tau_plus_daughters.append(pi_plus)
                for pi_minus in pi_minus_list:
                    if isAncestor(tau_plus, pi_minus.mother(0)):
                        tau_plus_daughters.append(pi_minus)
                for pi_neutral in pi_neutral_list:
                    if isAncestor(tau_plus, pi_neutral.mother(0)):
                        print('issues!!!!!')
                        tau_plus_daughters.append(pi_neutral)
                        tau_plus_neutral_check = tau_plus_neutral_check + 1
                for neutrino_plus in neu_plus_list:
                    if isAncestor(tau_plus, neutrino_plus.mother(0)):
                        tau_plus_neutrino.append(neutrino_plus)
                for neutrino_minus in neu_minus_list:
                    if isAncestor(tau_plus, neutrino_minus.mother(0)):
                        tau_plus_neutrino.append(neutrino_minus)
                if len(tau_plus_daughters) == 3 and tau_plus_neutral_check == 0:
                    tau_plus_keep = tau_plus
                    good_taup = True
                break

        for tau_minus in tau_minus_list:
            if isAncestor(upsilon, tau_minus.mother(0)):
                taum_daughter = True
                for pi_plus in pi_plus_list:
                    if isAncestor(tau_minus, pi_plus.mother(0)):
                        tau_minus_daughters.append(pi_plus)
                for pi_minus in pi_minus_list:
                    if isAncestor(tau_minus, pi_minus.mother(0)):
                        tau_minus_daughters.append(pi_minus)
                for pi_neutral in pi_neutral_list:
                    if isAncestor(tau_minus, pi_neutral.mother(0)):
                        print('issues!!!!!')
                        tau_minus_daughters.append(pi_neutral)
                        tau_minus_neutral_check = tau_minus_neutral_check + 1
                for neutrino_plus in neu_plus_list:
                    if isAncestor(tau_minus, neutrino_plus.mother(0)):
                        tau_minus_neutrino.append(neutrino_plus)
                for neutrino_minus in neu_minus_list:
                    if isAncestor(tau_minus, neutrino_minus.mother(0)):
                        tau_minus_neutrino.append(neutrino_minus)
                if len(tau_minus_daughters) == 3 and tau_minus_neutral_check == 0:
                    tau_minus_keep = tau_minus
                    good_taum = True
                break

        if not good_taum or not good_taup: raise ValueError('all taus should be good, but some arent')
          
        if good_taum and good_taup:

            #Add Generated info for tau pions
            add_gen_row_taup = [tau_plus_daughters[0].pt(), tau_plus_daughters[0].eta(), tau_plus_daughters[0].phi(),
                                tau_plus_daughters[1].pt(), tau_plus_daughters[1].eta(), tau_plus_daughters[1].phi(),
                                tau_plus_daughters[2].pt(), tau_plus_daughters[2].eta(), tau_plus_daughters[2].phi()]
            add_gen_row_df_taup = pd.DataFrame([add_gen_row_taup], columns = df_gen_pions.columns)
            df_gen_pions = pd.concat([df_gen_pions, add_gen_row_df_taup], ignore_index=True) 

            #Add Generated info for the antitau pions                     
            add_gen_row_taum = [tau_minus_daughters[0].pt(), tau_minus_daughters[0].eta(), tau_minus_daughters[0].phi(),
                                tau_minus_daughters[1].pt(), tau_minus_daughters[1].eta(), tau_minus_daughters[1].phi(),
                                tau_minus_daughters[2].pt(), tau_minus_daughters[2].eta(), tau_minus_daughters[2].phi()]
            add_gen_row_df_taum = pd.DataFrame([add_gen_row_taum], columns = df_gen_pions.columns)
            df_gen_pions = pd.concat([df_gen_pions, add_gen_row_df_taum], ignore_index=True)                     
        
            #  Matching gen to reco now
            upsilon_ready = False
            matched_pion_plus = []
            matched_pion_minus = []
            for gen_pion_plus in tau_plus_daughters:
                min_deltaR_plus = 999
                match = False
                for reco_particle in pf_particles:
                    if reco_particle.pdgId() == gen_pion_plus.pdgId():
                        
                        reco_lv = ROOT.TLorentzVector() 
                        reco_lv.SetPtEtaPhiM(reco_particle.pt(), reco_particle.eta(), reco_particle.phi(), reco_particle.mass())

                        gen_lv_plus = ROOT.TLorentzVector()
                        gen_lv_plus.SetPtEtaPhiM(gen_pion_plus.pt(), gen_pion_plus.eta(), gen_pion_plus.phi(), gen_pion_plus.mass())

                        deltaR_plus = gen_lv_plus.DeltaR(reco_lv)
                        deltaPT_plus = (reco_lv.Pt() - gen_lv_plus.Pt()) / gen_lv_plus.Pt()

                        if abs(deltaR_plus) < 0.1 and abs(deltaPT_plus) < 0.3 and abs(deltaR_plus) < min_deltaR_plus and abs(reco_particle.eta()) < 2.5 and reco_particle not in matched_pion_plus and reco_particle not in matched_pion_minus:
                            min_deltaR_plus = deltaR_plus
                            matched_pion_p = reco_particle
                            match = True
                if match:
                    matched_pion_plus.append(matched_pion_p)
                else:
                    continue

            for gen_pion_minus in tau_minus_daughters:
                min_deltaR_minus = 999
                match = False
                for reco_particle in pf_particles:
                    if reco_particle.pdgId() == gen_pion_minus.pdgId():

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

            if len(matched_pion_plus) == len(tau_plus_daughters) == len(matched_pion_minus) == len(tau_minus_daughters):
                upsilon_ready = True

            if len(matched_pion_plus) == len(tau_plus_daughters):
                if upsilon_ready:
                    add_row = [matched_pion_plus[0].pt(), matched_pion_plus[0].eta(), matched_pion_plus[0].phi(),
                            matched_pion_plus[1].pt(), matched_pion_plus[1].eta(), matched_pion_plus[1].phi(),
                            matched_pion_plus[2].pt(), matched_pion_plus[2].eta(), matched_pion_plus[2].phi(),
                            tau_plus_neutrino[0].pt(), tau_plus_neutrino[0].eta(), tau_plus_neutrino[0].phi(), 'p', 'y']
                else:
                    add_row = [matched_pion_plus[0].pt(), matched_pion_plus[0].eta(), matched_pion_plus[0].phi(),
                            matched_pion_plus[1].pt(), matched_pion_plus[1].eta(), matched_pion_plus[1].phi(),
                            matched_pion_plus[2].pt(), matched_pion_plus[2].eta(), matched_pion_plus[2].phi(),
                            tau_plus_neutrino[0].pt(), tau_plus_neutrino[0].eta(), tau_plus_neutrino[0].phi(), 'p', 'n']

                add_row_df = pd.DataFrame([add_row], columns = df_toUse.columns)
                df_toUse = pd.concat([df_toUse, add_row_df], ignore_index=True)

            if len(matched_pion_minus) == len(tau_minus_daughters):
                if upsilon_ready:
                    add_row = [matched_pion_minus[0].pt(), matched_pion_minus[0].eta(), matched_pion_minus[0].phi(),
                            matched_pion_minus[1].pt(), matched_pion_minus[1].eta(), matched_pion_minus[1].phi(),
                            matched_pion_minus[2].pt(), matched_pion_minus[2].eta(), matched_pion_minus[2].phi(),
                            tau_minus_neutrino[0].pt(), tau_minus_neutrino[0].eta(), tau_minus_neutrino[0].phi(), 'm', 'y']
                else:
                    add_row = [matched_pion_minus[0].pt(), matched_pion_minus[0].eta(), matched_pion_minus[0].phi(),
                            matched_pion_minus[1].pt(), matched_pion_minus[1].eta(), matched_pion_minus[1].phi(),
                            matched_pion_minus[2].pt(), matched_pion_minus[2].eta(), matched_pion_minus[2].phi(),
                            tau_minus_neutrino[0].pt(), tau_minus_neutrino[0].eta(), tau_minus_neutrino[0].phi(), 'm', 'n']

                add_row_df = pd.DataFrame([add_row], columns = df_toUse.columns)
                df_toUse = pd.concat([df_toUse, add_row_df], ignore_index=True)

print('Total num of events: ' + str(num_events))
df_gen_pions.to_csv('/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/40k_Varied_Y_Mass_Files/40k_15GeV_Y_GEN.csv')
df_toUse.to_csv(f'/isilon/export/home/hdmiller/cms_work/tau-decay-ml/data/40k_Varied_Y_Mass_Files/40k_15GeV_Y.csv')
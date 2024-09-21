# Script to process MiniAODs and convert them into flat ntuples

# Import section

import ROOT # for 4-vector builds
from DataFormats.FWLite import Events, Handle # to open MiniAODs
import pandas as pd

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


# Main 
for i in range(20):
    #if i != 2 and i!=6:
        if __name__ == '__main__':
            column_names = ['pi1_from_tau_pt', 'pi1_from_tau_eta', 'pi1_from_tau_phi','pi2_from_tau_pt', 'pi2_from_tau_eta', 'pi2_from_tau_phi', 'pi3_from_tau_pt', 'pi3_from_tau_eta', 'pi3_from_tau_phi',
                                        'pi1_from_antitau_pt', 'pi1_from_antitau_eta', 'pi1_from_antitau_phi', 'pi2_from_antitau_pt', 'pi2_from_antitau_eta', 'pi2_from_antitau_phi','pi3_from_antitau_pt', 'pi3_from_antitau_eta', 'pi3_from_antitau_phi', 
                                        'neutrino_from_tau_pt','neutrino_from_tau_eta','neutrino_from_tau_phi','neutrino_from_antitau_pt','neutrino_from_antitau_eta','neutrino_from_antitau_phi', 'tau_no_neutrino_mass', 'tau_with_neutrino_mass', 
                                        'antitau_no_neutrino_mass', 'antitau_with_neutrino_mass', 'upsilon_no_neutrino_mass', 'upsilon_with_neutrino_mass']
            
            df_toUse = pd.DataFrame(columns = column_names)

            gen_column_names = ['gen_pi1_from_tau_pt', 'gen_pi1_from_tau_eta', 'gen_pi1_from_tau_phi', 'gen_pi2_from_tau_pt', 'gen_pi2_from_tau_eta', 'gen_pi2_from_tau_phi', 'gen_pi3_from_tau_pt', 'gen_pi3_from_tau_eta', 'gen_pi3_from_tau_phi',
                                    'gen_pi1_from_antitau_pt', 'gen_pi1_from_antitau_eta', 'gen_pi1_from_antitau_phi', 'gen_pi2_from_antitau_pt', 'gen_pi2_from_antitau_eta', 'gen_pi2_from_antitau_phi', 'gen_pi3_from_antitau_pt', 'gen_pi3_from_antitau_eta', 'gen_pi3_from_antitau_phi',
                                    'gen_tau_pt', 'gen_tau_eta', 'gen_tau_phi', 'gen_antitau_pt', 'gen_antitau_eta', 'gen_antitau_phi', 'gen_upsilon_pt', 'gen_upsilon_eta', 'gen_upsilon_phi', 'gen_upsilon_mass']
            
            df_genInfo = pd.DataFrame(columns = gen_column_names)


            # Argument parser and fixing the CMSSW version via the options container
            parser = argparse.ArgumentParser(description='Args')
            parser.add_argument('--f_in', default=f'UpsilonToTauTau_3prong_miniaod_part{i}') 
            #parser.add_argument('--maxEvents', default = 100)
            args = parser.parse_args()

            # file path
            path = '/isilon/data/users/mstamenk/tau-project-summer24/samples/preprocess15GeVMiniaodSampleWITHOUTpTCutIncl'
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

                gen_dict = {}
                num_pi_plus = 0

                #print(len(gen_particles))

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
                        photon_list.append(gen_particle)
                    if gen_particle.pdgId() == 111:
                        pi_neutral_list.append(gen_particle)
                    
                    """
                    if gen_particle.pdgId() in pdgId_map.keys():
                        vec = ROOT.TLorentzVector()
                        vec.SetPtEtaPhiM(gen_particle.pt(), gen_particle.eta(), gen_particle.phi(), gen_particle.mass())
                        if pdgId_map[int(gen_particle.pdgId())] in gen_dict.keys():
                            gen_dict[pdgId_map[int(gen_particle.pdgId())]].append(vec)
                        else:
                            gen_dict[pdgId_map[int(gen_particle.pdgId())]] = [vec]
                            #print(gen_dict[pdgId_map[int(gen_particle.pdgId())]])
                    #else: 
                        #print(gen_particle.pdgId())
                    """

                # Retrieve 4 vectors
                """
                tau_plus = gen_dict['tau_plus']  
                tau_minus = gen_dict['tau_minus']
                upsilon = gen_dict['upsilon_id']
                pi_plus = gen_dict['pion_plus']
                pi_minus = gen_dict['pion_minus']
                tau_neu_plus = gen_dict['tau_neu_plus']
                tau_neu_minus = gen_dict['tau_neu_plus']
                """

                tau_counter = 0

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
                for tau_plus in tau_plus_list:
                    if isAncestor(upsilon_list[0], tau_plus.mother(0)):
                        tau_from_upsilon = tau_plus
                        for pi_plus in pi_plus_list:
                            if isAncestor(tau_plus, pi_plus.mother(0)):
                                tau_plus_daughters.append(pi_plus)
                        for pi_minus in pi_minus_list:
                            if isAncestor(tau_plus, pi_minus.mother(0)):
                                tau_plus_daughters.append(pi_minus)
                        for pi_neutral in pi_neutral_list:
                            if isAncestor(tau_plus, pi_neutral.mother(0)):
                                tau_plus_daughters.append(pi_neutral)
                        for neutrino_plus in neu_plus_list:
                            if isAncestor(tau_plus, neutrino_plus.mother(0)):
                                tau_plus_neutrino.append(neutrino_plus)
                        for neutrino_minus in neu_minus_list:
                            if isAncestor(tau_plus, neutrino_minus.mother(0)):
                                tau_plus_neutrino.append(neutrino_minus)
                        if len(tau_plus_daughters) == 3:
                            tau_counter = tau_counter + 1
                            tau_plus_keep = tau_plus
                            break
                    
                    

                tau_minus_daughters = []
                tau_minus_neutrino = []
                for tau_minus in tau_minus_list:
                    antitau_from_upsilon = tau_minus
                    if isAncestor(upsilon_list[0], tau_minus.mother(0)):
                        for pi_plus in pi_plus_list:
                            if isAncestor(tau_minus, pi_plus.mother(0)):
                                tau_minus_daughters.append(pi_plus)
                        for pi_minus in pi_minus_list:
                            if isAncestor(tau_minus, pi_minus.mother(0)):
                                tau_minus_daughters.append(pi_minus)
                        for pi_neutral in pi_neutral_list:
                            if isAncestor(tau_minus, pi_neutral.mother(0)):
                                tau_minus_daughters.append(pi_neutral)
                        for neutrino_plus in neu_plus_list:
                            if isAncestor(tau_minus, neutrino_plus.mother(0)):
                                tau_minus_neutrino.append(neutrino_plus)
                        for neutrino_minus in neu_minus_list:
                            if isAncestor(tau_minus, neutrino_minus.mother(0)):
                                tau_minus_neutrino.append(neutrino_minus)
                        if len(tau_minus_daughters) == 3:
                            tau_counter = tau_counter + 1
                            tau_minus_keep = tau_minus
                            break


            ###
                if tau_counter != 2:
                    #print('No good taus found!')
                    continue

                else:
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
                                #print(deltaPT_plus, deltaR_plus)
                                
                                if abs(deltaR_plus) < 0.1 and abs(deltaPT_plus) < 0.3 and abs(deltaR_plus) < min_deltaR_plus and abs(reco_particle.eta()) < 2.5 and reco_particle not in matched_pion_plus and reco_particle not in matched_pion_minus:
                                    #print('found candidate')
                                    min_deltaR_plus = deltaR_plus
                                    matched_pion_p = reco_particle
                                    match = True
                        if match:
                            matched_pion_plus.append(matched_pion_p)
                            #print(len(matched_pion_plus))
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

                                if abs(deltaR_minus) < 0.1 and deltaPT_minus < 0.3 and deltaR_minus < min_deltaR_minus and abs(reco_particle.eta()) < 2.5 and reco_particle not in matched_pion_minus and reco_particle not in matched_pion_plus:
                                    #print('found candidate')
                                    min_deltaR_minus = deltaR_minus
                                    matched_pion_m = reco_particle
                                    match = True
                        if match:
                            matched_pion_minus.append(matched_pion_m)
                            #print(len(matched_pion_minus))
                        else: 
                            #print('no match')
                            continue
                            

                    if len(matched_pion_plus) == len(tau_plus_daughters) and len(matched_pion_minus) == len(tau_minus_daughters):
                        pi1_plus_lv = ROOT.TLorentzVector()
                        pi2_plus_lv = ROOT.TLorentzVector()
                        pi3_plus_lv = ROOT.TLorentzVector()
                        neutrino_plus_lv = ROOT.TLorentzVector()

                        pi1_plus_lv.SetPtEtaPhiM(matched_pion_plus[0].pt(), matched_pion_plus[0].eta(), matched_pion_plus[0].phi(), 0.139)
                        pi2_plus_lv.SetPtEtaPhiM(matched_pion_plus[1].pt(), matched_pion_plus[1].eta(), matched_pion_plus[1].phi(), 0.139)
                        pi3_plus_lv.SetPtEtaPhiM(matched_pion_plus[2].pt(), matched_pion_plus[2].eta(), matched_pion_plus[2].phi(), 0.139)
                        neutrino_plus_lv.SetPtEtaPhiM(tau_plus_neutrino[0].pt(), tau_plus_neutrino[0].eta(), tau_plus_neutrino[0].phi(), 0)

                        pi1_minus_lv = ROOT.TLorentzVector()
                        pi2_minus_lv = ROOT.TLorentzVector()
                        pi3_minus_lv = ROOT.TLorentzVector()
                        neutrino_minus_lv = ROOT.TLorentzVector()

                        pi1_minus_lv.SetPtEtaPhiM(matched_pion_minus[0].pt(), matched_pion_minus[0].eta(), matched_pion_minus[0].phi(), 0.139)
                        pi2_minus_lv.SetPtEtaPhiM(matched_pion_minus[1].pt(), matched_pion_minus[1].eta(), matched_pion_minus[1].phi(), 0.139)
                        pi3_minus_lv.SetPtEtaPhiM(matched_pion_minus[2].pt(), matched_pion_minus[2].eta(), matched_pion_minus[2].phi(), 0.139)
                        neutrino_minus_lv.SetPtEtaPhiM(tau_minus_neutrino[0].pt(), tau_minus_neutrino[0].eta(), tau_minus_neutrino[0].phi(), 0)

                        tau_plus_no_neutrino_lv = pi1_plus_lv + pi2_plus_lv + pi3_plus_lv
                        tau_plus_with_neutrino_lv = pi1_plus_lv + pi2_plus_lv + pi3_plus_lv + neutrino_plus_lv

                        tau_minus_no_neutrino_lv = pi1_minus_lv + pi2_minus_lv + pi3_minus_lv
                        tau_minus_with_neutrino_lv = pi1_minus_lv + pi2_minus_lv + pi3_minus_lv + neutrino_minus_lv

                        upsilon_no_neutrino_lv = tau_plus_no_neutrino_lv + tau_minus_no_neutrino_lv
                        upsilon_with_neutrino_lv = tau_plus_with_neutrino_lv + tau_minus_with_neutrino_lv

                        tau_no_neutrino_mass = tau_plus_no_neutrino_lv.M()
                        tau_with_neutrino_mass = tau_plus_with_neutrino_lv.M()

                        antitau_no_neutrino_mass = tau_plus_no_neutrino_lv.M()
                        antitau_with_neutrino_mass = tau_plus_with_neutrino_lv.M()

                        upsilon_no_neutrino_mass = upsilon_no_neutrino_lv.M()
                        upsilon_with_neutrino_mass = upsilon_with_neutrino_lv.M()





                    
                        #switching nomenclature here to tau and antitau

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

                        ### Gen Level Info Saving ###

                        gen_tau_lv = ROOT.TLorentzVector()
                        gen_pi1_from_tau_lv = ROOT.TLorentzVector()
                        gen_pi2_from_tau_lv = ROOT.TLorentzVector()
                        gen_pi3_from_tau_lv = ROOT.TLorentzVector()

                        gen_tau_lv.SetPtEtaPhiM(tau_from_upsilon.pt(), tau_from_upsilon.eta(), tau_from_upsilon.phi(), tau_from_upsilon.mass())
                        gen_pi1_from_tau_lv.SetPtEtaPhiM(tau_plus_daughters[0].pt(), tau_plus_daughters[0].eta(), tau_plus_daughters[0].phi(), tau_plus_daughters[0].mass())
                        gen_pi2_from_tau_lv.SetPtEtaPhiM(tau_plus_daughters[1].pt(), tau_plus_daughters[1].eta(), tau_plus_daughters[1].phi(), tau_plus_daughters[1].mass())
                        gen_pi3_from_tau_lv.SetPtEtaPhiM(tau_plus_daughters[2].pt(), tau_plus_daughters[2].eta(), tau_plus_daughters[2].phi(), tau_plus_daughters[2].mass())

                        gen_pi1_from_tau_pt = gen_pi1_from_tau_lv.Pt()
                        gen_pi1_from_tau_eta = gen_pi1_from_tau_lv.Eta()
                        gen_pi1_from_tau_phi = gen_pi1_from_tau_lv.Phi()

                        gen_pi2_from_tau_pt = gen_pi2_from_tau_lv.Pt()
                        gen_pi2_from_tau_eta = gen_pi2_from_tau_lv.Eta()
                        gen_pi2_from_tau_phi = gen_pi2_from_tau_lv.Phi()

                        gen_pi3_from_tau_pt = gen_pi3_from_tau_lv.Pt()
                        gen_pi3_from_tau_eta = gen_pi3_from_tau_lv.Eta()
                        gen_pi3_from_tau_phi = gen_pi3_from_tau_lv.Phi()

                        gen_tau_pt = gen_tau_lv.Pt()
                        gen_tau_eta = gen_tau_lv.Eta()
                        gen_tau_phi = gen_tau_lv.Phi()

                        gen_antitau_lv = ROOT.TLorentzVector()
                        gen_pi1_from_antitau_lv = ROOT.TLorentzVector()
                        gen_pi2_from_antitau_lv = ROOT.TLorentzVector()
                        gen_pi3_from_antitau_lv = ROOT.TLorentzVector()

                        gen_antitau_lv.SetPtEtaPhiM(antitau_from_upsilon.pt(), antitau_from_upsilon.eta(), antitau_from_upsilon.phi(), antitau_from_upsilon.mass())
                        gen_pi1_from_antitau_lv.SetPtEtaPhiM(tau_minus_daughters[0].pt(), tau_minus_daughters[0].eta(), tau_minus_daughters[0].phi(), tau_minus_daughters[0].mass())
                        gen_pi2_from_antitau_lv.SetPtEtaPhiM(tau_minus_daughters[1].pt(), tau_minus_daughters[1].eta(), tau_minus_daughters[1].phi(), tau_minus_daughters[1].mass())
                        gen_pi3_from_antitau_lv.SetPtEtaPhiM(tau_minus_daughters[2].pt(), tau_minus_daughters[2].eta(), tau_minus_daughters[2].phi(), tau_minus_daughters[2].mass())

                        gen_pi1_from_antitau_pt = gen_pi1_from_antitau_lv.Pt()
                        gen_pi1_from_antitau_eta = gen_pi1_from_antitau_lv.Eta()
                        gen_pi1_from_antitau_phi = gen_pi1_from_antitau_lv.Phi()

                        gen_pi2_from_antitau_pt = gen_pi2_from_antitau_lv.Pt()
                        gen_pi2_from_antitau_eta = gen_pi2_from_antitau_lv.Eta()
                        gen_pi2_from_antitau_phi = gen_pi2_from_antitau_lv.Phi()

                        gen_pi3_from_antitau_pt = gen_pi3_from_antitau_lv.Pt()
                        gen_pi3_from_antitau_eta = gen_pi3_from_antitau_lv.Eta()
                        gen_pi3_from_antitau_phi = gen_pi3_from_antitau_lv.Phi()

                        gen_antitau_pt = gen_antitau_lv.Pt()
                        gen_antitau_eta = gen_antitau_lv.Eta()
                        gen_antitau_phi = gen_antitau_lv.Phi()

                        gen_upsilon = upsilon_list[0]
                        gen_upsilon_lv = ROOT.TLorentzVector()
                        gen_upsilon_lv.SetPtEtaPhiM(gen_upsilon.pt(), gen_upsilon.eta(), gen_upsilon.phi(), gen_upsilon.mass())

                        gen_upsilon_pt = gen_upsilon_lv.Pt()
                        gen_upsilon_eta = gen_upsilon_lv.Eta()
                        gen_upsilon_phi = gen_upsilon_lv.Phi()
                        gen_upsilon_mass = gen_upsilon_lv.M()

                        columns_names = ['pi1_from_tau_pt', 'pi1_from_tau_eta', 'pi1_from_tau_phi','pi2_from_tau_pt', 'pi2_from_tau_eta', 'pi2_from_tau_phi', 'pi3_from_tau_pt', 'pi3_from_tau_eta', 'pi3_from_tau_phi',
                                        'pi1_from_antitau_pt', 'pi1_from_antitau_eta', 'pi1_from_antitau_phi', 'pi2_from_antitau_pt', 'pi2_from_antitau_eta', 'pi2_from_antitau_phi','pi3_from_antitau_pt', 'pi3_from_antitau_eta', 'pi3_from_antitau_phi' 
                                        'neutrino_from_tau_pt','neutrino_from_tau_eta','neutrino_from_tau_phi','neutrino_from_antitau_pt','neutrino_from_antitau_eta','neutrino_from_antitau_phi', 'tau_no_neutrino_mass', 'tau_with_neutrino_mass', 
                                        'antitau_no_neutrino_mass', 'antitau_with_neutrino_mass', 'upsilon_no_neutrino_mass', 'upsilon_with_neutrino_mass']
                    
                        add_row = [pi1_from_tau_pt, pi1_from_tau_eta, pi1_from_tau_phi, pi2_from_tau_pt, pi2_from_tau_eta, pi2_from_tau_phi, pi3_from_tau_pt, pi3_from_tau_eta, pi3_from_tau_phi,
                                        pi1_from_antitau_pt, pi1_from_antitau_eta, pi1_from_antitau_phi, pi2_from_antitau_pt, pi2_from_antitau_eta, pi2_from_antitau_phi,pi3_from_antitau_pt, pi3_from_antitau_eta, pi3_from_antitau_phi,
                                        neutrino_from_tau_pt, neutrino_from_tau_eta, neutrino_from_tau_phi,neutrino_from_antitau_pt,neutrino_from_antitau_eta,neutrino_from_antitau_phi, tau_no_neutrino_mass, tau_with_neutrino_mass, 
                                        antitau_no_neutrino_mass, antitau_with_neutrino_mass, upsilon_no_neutrino_mass, upsilon_with_neutrino_mass]
                        
                        gen_column_names = ['gen_pi1_from_tau_pt', 'gen_pi1_from_tau_eta', 'gen_pi1_from_tau_phi', 'gen_pi2_from_tau_pt', 'gen_pi2_from_tau_eta', 'gen_pi2_from_tau_phi', 'gen_pi3_from_tau_pt', 'gen_pi3_from_tau_eta', 'gen_pi3_from_tau_phi',
                                    'gen_pi1_from_antitau_pt', 'gen_pi1_from_antitau_eta', 'gen_pi1_from_antitau_phi', 'gen_pi2_from_antitau_pt', 'gen_pi2_from_antitau_eta', 'gen_pi2_from_antitau_phi', 'gen_pi3_from_antitau_pt', 'gen_pi3_from_antitau_eta', 'gen_pi3_from_antitau_phi',
                                    'gen_tau_pt', 'gen_tau_eta', 'gen_tau_phi', 'gen_antitau_pt', 'gen_antitau_eta', 'gen_antitau_phi', 'gen_upsilon_pt', 'gen_upsilon_eta', 'gen_upsilon_phi', 'gen_upsilon_mass']

                        add_gen_row = [gen_pi1_from_tau_pt, gen_pi1_from_tau_eta, gen_pi1_from_tau_phi, gen_pi2_from_tau_pt, gen_pi2_from_tau_eta, gen_pi2_from_tau_phi, gen_pi3_from_tau_pt, gen_pi3_from_tau_eta, gen_pi3_from_tau_phi,
                                    gen_pi1_from_antitau_pt, gen_pi1_from_antitau_eta, gen_pi1_from_antitau_phi, gen_pi2_from_antitau_pt, gen_pi2_from_antitau_eta, gen_pi2_from_antitau_phi, gen_pi3_from_antitau_pt, gen_pi3_from_antitau_eta, gen_pi3_from_antitau_phi,
                                    gen_tau_pt, gen_tau_eta, gen_tau_phi, gen_antitau_pt, gen_antitau_eta, gen_antitau_phi, gen_upsilon_pt, gen_upsilon_eta, gen_upsilon_phi, gen_upsilon_mass]
                        
                        add_row_df = pd.DataFrame([add_row], columns = df_toUse.columns)
                        df_toUse = pd.concat([df_toUse, add_row_df], ignore_index=True)

                        add_gen_row_df = pd.DataFrame([add_gen_row], columns = df_genInfo.columns)
                        df_genInfo = pd.concat([df_genInfo, add_gen_row_df], ignore_index=True)

            df_toUse.to_csv(f'/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing/reco_information_{i}_miniaod.csv')
            df_genInfo.to_csv(f'/isilon/export/home/hdmiller/cms_work/tau-decay-ml/preprocessing/gen_information_{i}_miniaod.csv')
            print(f'csv {i} saved')

import ROOT 

import os

# f_sim = ROOT.TFile.Open('ship.conical.MuonBack-TGeant4.root')
f_sim = ROOT.TFile.Open('muons.root')
# f = ROOT.TFile.Open('/eos/experiment/ship/user/amarshal/RPV_output/check/ship.conical.Pythia8-TGeant4_rec_46.root')

tree = f_sim.Get("pythia8-Geant4")

N = tree.GetEntries()

print(N)
# python FairShip/muonShieldOptimization/run_fixedTarget.py -n 1100000
os.system("python FairShip/muonShieldOptimization/run_fixedTarget.py -n %d"%(N-1))








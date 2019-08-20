
import ROOT 
import numpy as np
import argparse
import shipunit as u
from ShipGeoConfig import ConfigRegistry
from decorators import *
import shipRoot_conf
shipRoot_conf.configure()
from array import array
import math

import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# f_sim = ROOT.TFile.Open('ship.conical.MuonBack-TGeant4.root')
f_sim = ROOT.TFile.Open('muons.root')
# f = ROOT.TFile.Open('/eos/experiment/ship/user/amarshal/RPV_output/check/ship.conical.Pythia8-TGeant4_rec_46.root')

tree = f_sim.Get("pythia8-Geant4")

N = tree.GetEntries()

print(N)









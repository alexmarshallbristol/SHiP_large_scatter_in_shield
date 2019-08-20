
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
f_sim = ROOT.TFile.Open('FIXED_TARGET_OUTPUT.root')
# f = ROOT.TFile.Open('/eos/experiment/ship/user/amarshal/RPV_output/check/ship.conical.Pythia8-TGeant4_rec_46.root')

tree = f_sim.Get("cbmsim")

N = tree.GetEntries()

data = np.empty((0,3))
i = 0
for event in tree:
	i += 1

	# for e in event.MCTrack:

		# print(e.GetPdgCode())
		# MCTrack_buffer = [e.GetPdgCode(),e.GetStartX(),e.GetStartY(),e.GetStartZ(),e.GetPx(),e.GetPy(),e.GetPz()]
		# buffer

	for e in event.vetoPoint:
		# help(e)

		if i == 1:
			print(e.GetZ())
			# data = np.append(data, [[e.GetX(), e.GetY(), e.GetZ()]], axis=0)
		data = np.append(data,[[e.GetX(),e.GetY(),e.GetZ()]],axis=0)

		# muon_history = [MCTrack_buffer,[e.PdgCode(), e.GetX(), e.GetY(), e.GetZ(), e.GetPx(), e.GetPy(), e.GetPz()]]


		# print(muon_history)
	# quit()
	# print(' ')
	
	# print(np.shape(muon_history))

print(np.shape(data))

plt.figure(figsize=(12,6))

materials_iron_bool = np.load('materials_iron_bool.npy')

materials_iron_bool2 = materials_iron_bool*100000 + 1

plt.subplot(1,2,1)
plt.hist2d(data[:,0], data[:,1],bins=100,norm=LogNorm(),weights=materials_iron_bool2,range=[[-400,400],[-400,400]])

# materials_iron_bool = np.load('materials_iron_bool.npy')

data = data[np.where(materials_iron_bool==1)]

plt.subplot(1,2,2)
plt.hist2d(data[:,0], data[:,1],bins=100,norm=LogNorm(),range=[[-400,400],[-400,400]])

# plt.colorbar()
plt.savefig('dis',bbox_inches='tight')
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')



# ax.scatter(data[:,0], data[:,1], data[:,2], c='r', marker='o')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()
# quit()
# plt.savefig('dis')








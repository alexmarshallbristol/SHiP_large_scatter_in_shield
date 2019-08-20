
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



# parser = argparse.ArgumentParser()
# parser.add_argument('-jobid', action='store', dest='jobid', type=int,
# 					help='jobid')
# results = parser.parse_args()
# jobid = int(results.jobid)

# # Pass in jobid - index of current point in 50x50 grid.
# parser = argparse.ArgumentParser()
# parser.add_argument('-jobid', action='store', dest='jobid', type=int,
# 					help='jobid')
# parser.add_argument('-benchmark', action='store', dest='benchmark', type=int,
# 					help='benchmark')
# parser.add_argument('-plot', action='store', dest='plot', type=int,
# 					help='plot')
# parser.add_argument('-plot2mass', action='store', dest='plot2mass', type=float,
# 					help='plot2mass GeV',default=0.6)
# parser.add_argument('-plot1massmax', action='store', dest='plot1massmax', type=float,
# 					help='plot1massmax GeV',default=2.0)
# parser.add_argument('-leptongeneration', action='store', dest='leptongeneration', type=int,
# 					help='leptongeneration',default=2.0)
# results = parser.parse_args()


file_ouput = 'muons_scattered.root'
f_save = ROOT.TFile(file_ouput, "recreate")
t = ROOT.TTree("pythia8-Geant4", "pythia8-Geant4")

float64 = 'd'
event_id = np.zeros(1, dtype=float64)
id_ = np.zeros(1, dtype=float64)
parentid = np.zeros(1, dtype=float64)
pythiaid = np.zeros(1, dtype=float64)
ecut = np.zeros(1, dtype=float64)
w = np.zeros(1, dtype=float64)
vx = np.zeros(1, dtype=float64)
vy = np.zeros(1, dtype=float64)
vz = np.zeros(1, dtype=float64)
px = np.zeros(1, dtype=float64)
py = np.zeros(1, dtype=float64)
pz = np.zeros(1, dtype=float64)
release_time = np.zeros(1, dtype=float64)
mother_id = np.zeros(1, dtype=float64)
process_id = np.zeros(1, dtype=float64)

t.Branch('event_id', event_id, 'event_id/D')
t.Branch('id', id_, 'id/D')
t.Branch('parentid', parentid, 'parentid/D')
t.Branch('pythiaid', pythiaid, 'pythiaid/D')
t.Branch('ecut', ecut, 'ecut/D')
t.Branch('w', w, 'w/D')#relative weight
t.Branch('x', vx, 'vx/D')#pos
t.Branch('y', vy, 'vy/D')
t.Branch('z', vz, 'vz/D')
t.Branch('px', px, 'px/D')
t.Branch('py', py, 'py/D')
t.Branch('pz', pz, 'pz/D')
t.Branch('release_time', release_time, 'release_time/D')
t.Branch('mother_id', mother_id, 'mother_id/D')
t.Branch('process_id', process_id, 'process_id/D')


def save(event_id_in, pdg_in, x_in, y_in, z_in, px_in, py_in, pz_in, mother_id_in, process_id_in):
			event_id[0] = event_id_in
			id_[0] = 0
			parentid[0] = 0
			pythiaid[0] = pdg_in
			ecut[0] = 0.00001
			w[0] = 1
			vx[0] = x_in
			vy[0] = y_in
			vz[0] = z_in
			# vz[0] = 0
			px[0] = px_in
			py[0] = py_in
			pz[0] = pz_in
			release_time[0] = 0
			mother_id[0] = mother_id_in
			process_id[0] = process_id_in
			t.Fill()




# sensitive_plane_details = np.load("Sensitive_plane_position.npy")
# z_position = sensitive_plane_details[1]
# z_position = -7075.0

# quit()


f_sim = ROOT.TFile.Open('/afs/cern.ch/user/a/amarshal/GEANT_fairship_geo/lxplus602.cern.ch_run_fixedTarget_1/pythia8_evtgen_Geant4_1_0.5.root')
# 
# f_sim = ROOT.TFile.Open('/afs/cern.ch/user/a/amarshal/GEANT_fairship_geo/ship.conical.MuonBack-TGeant4.root')

# f = ROOT.TFile.Open('/eos/experiment/ship/user/amarshal/RPV_output/check/ship.conical.Pythia8-TGeant4_rec_46.root')

tree = f_sim.Get("cbmsim")

N = tree.GetEntries()

# muon_kinematics_at_plane = np.empty((0,8))
# [inputmomentum ,charge, x, y, z, px, py, pz]

i = -1

print('starting event loop')


# muon_scattering_history_pre_scatter = np.empty((0,2,7))
# muon_scattering_history_scatter = np.empty((0,1,7))

for event in tree:


	# for e in event.MCTrack:

		# print(e.GetStartZ())
		# MCTrack_buffer = [e.GetPdgCode(),e.GetStartX(),e.GetStartY(),e.GetStartZ(),e.GetPx(),e.GetPy(),e.GetPz()]
		# buffer

	for e in event.vetoPoint:



		#check if iron


		# muon_kinematics_at_plane = np.append(muon_kinematics_at_plane, [[np.sqrt(np.add(e.GetPx()**2,np.add(e.GetPy()**2,e.GetPz()**2))),e.PdgCode(), e.GetX(), e.GetY(), e.GetZ(), e.GetPx(), e.GetPy(), e.GetPz()]], axis=0)

		# def save(event_id_in, pdg_in, x_in, y_in, z_in, px_in, py_in, pz_in, mother_id_in, process_id_in):
		save(1,e.PdgCode(),e.GetX(),e.GetY(),e.GetZ(),e.GetPx(),e.GetPy(),e.GetPz(),99,99)


# muon_scattering_history = np.append(muon_scattering_history_pre_scatter,muon_scattering_history_scatter, axis=1)

# print(muon_scattering_history, np.shape(muon_scattering_history))


# np.save('muon_scattering_history_%d'%jobid,muon_scattering_history)


f_save.Write()
f_save.Close()

# python FairShip/macro/run_simScript.py -f muons_scattered.root --MuonBack --FastMuon -n 100000






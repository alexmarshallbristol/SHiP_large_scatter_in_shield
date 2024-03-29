import numpy as np
import time
import ROOT

# generated_training = np.load('muons.npy')
generated_training = np.load('/eos/experiment/ship/user/amarshal/HUGE_GAN_random_id/huge_generation_april_999958023.npy')

generated_training = np.take(generated_training,np.random.permutation(generated_training.shape[0]),axis=0,out=generated_training)


file_ouput = 'muons.root'
f = ROOT.TFile(file_ouput, "recreate")
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


# for i in range(0, np.shape(generated_training)[0]):
for i in range(0, 10000):
	save(1,generated_training[i][0],generated_training[i][1],generated_training[i][2],generated_training[i][3],generated_training[i][4],generated_training[i][5],generated_training[i][6],99,99)
	# save(1,13,0,0,-8000,0,0,1250,99,99)

print(np.shape(generated_training))

f.Write()
f.Close()












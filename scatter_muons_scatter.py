
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

# import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm




parser = argparse.ArgumentParser()
parser.add_argument('-jobid', action='store', dest='jobid', type=int,
					help='jobid')
results = parser.parse_args()
jobid = int(results.jobid)

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
# # z_position = -7075.0

# # quit()


f_sim = ROOT.TFile.Open('FIXED_TARGET_OUTPUT.root')
# f = ROOT.TFile.Open('/eos/experiment/ship/user/amarshal/RPV_output/check/ship.conical.Pythia8-TGeant4_rec_46.root')

tree = f_sim.Get("cbmsim")

N = tree.GetEntries()

muon_kinematics_at_plane = np.empty((0,8))
# [inputmomentum ,charge, x, y, z, px, py, pz]

i = -1

print('starting event loop')

import checkMagFields



muon_scattering_history_pre_scatter = np.empty((0,2,7))
muon_scattering_history_scatter = np.empty((0,1,7))


materials_iron_bool = np.load('materials_iron_bool.npy')

i = -1
for event in tree:
	

	for e in event.MCTrack:

		# print(e.GetPdgCode())
		MCTrack_buffer = [e.GetPdgCode(),e.GetStartX(),e.GetStartY(),e.GetStartZ(),e.GetPx(),e.GetPy(),e.GetPz()]
		# buffer

	for e in event.vetoPoint:
		i += 1
			#check if iron

		# material = checkMagFields.check_material(e.GetX(), e.GetY(), e.GetZ())
		# print(material)

		if materials_iron_bool[i] == 1:
			#iron



			muon_kinematics_at_plane = np.append(muon_kinematics_at_plane, [[np.sqrt(np.add(e.GetPx()**2,np.add(e.GetPy()**2,e.GetPz()**2))),e.PdgCode(), e.GetX(), e.GetY(), e.GetZ(), e.GetPx(), e.GetPy(), e.GetPz()]], axis=0)

			muon_history = [MCTrack_buffer,[e.PdgCode(), e.GetX(), e.GetY(), e.GetZ(), e.GetPx(), e.GetPy(), e.GetPz()]]

			# print(np.shape(muon_history))

			muon_scattering_history_pre_scatter = np.append(muon_scattering_history_pre_scatter,[muon_history],axis=0)


print(np.shape(muon_scattering_history_pre_scatter), np.shape(muon_kinematics_at_plane))
# muon_scattering_history_pre_scatter = np.load('muon_scattering_history_pre_scatter.npy')

print(np.shape(materials_iron_bool))
#   # if material == 'iron':
#   #   materials[i] = 1
#   # else:
#   #   materials[i] = 0

# # plt.hist2d(muon_scattering_history_pre_scatter[:,1,1],muon_scattering_history_pre_scatter[:,1,2],bins=50,norm=LogNorm())
# # plt.savefig('test.png')

# quit()


# quit()

# print(np.shape(muon_scattering_history_pre_scatter))

# quit()
		# print(e.PdgCode(), e.GetZ())
# muon_kinematics_at_plane[0][0] = 235
# print(muon_kinematics_at_plane)
# print(np.shape(muon_kinematics_at_plane))



# use input mom [:,0] to get scattering angle and momentum loss


KDE_values_array = np.load('KDE_values_array.npy')
# print(np.shape(scattering_data))
# [initial_mom,fraction_of_mom_lost,scatter_angle]

# input_mom_scattering_data = scattering_data[0]



# range_values =[[0,400],[-10.1,-0.2],[-10,-0.4]]
range_values =[[-2.2,6],[-10.1,-0.2],[-10,-0.4]]
mom_step = range_values[0][1]/np.shape(KDE_values_array)[0]

mom_bins = np.empty(0)

for i in range(0, np.shape(KDE_values_array)[0]):
    mom = i * mom_step + mom_step/2
    mom_bins = np.append(mom_bins, mom)



log_frac_bins = np.empty(0)
log_angle_bins = np.empty(0)

log_frac_step = (range_values[1][1]-range_values[1][0])/np.shape(KDE_values_array)[1]

log_angle_step = (range_values[2][1]-range_values[2][0])/np.shape(KDE_values_array)[2]

print(mom_step, log_frac_step, log_angle_step)

for i in range(0, np.shape(KDE_values_array)[1]):
    log_frac = i * log_frac_step + log_frac_step/2 + range_values[1][0]
    log_frac_bins = np.append(log_frac_bins, log_frac)

for i in range(0, np.shape(KDE_values_array)[2]):
    log_angle = i * log_angle_step + log_angle_step/2 + range_values[2][0]
    log_angle_bins = np.append(log_angle_bins, log_angle)





# print(np.shape(input_mom_scattering_data))

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

for i in range(1, np.shape(muon_kinematics_at_plane)[0]):
	print(' ')
	# print(muon_kinematics_at_plane[0])
	# index = find_nearest(input_mom_scattering_data, muon_kinematics_at_plane[i][0])
	# print(input_mom_scattering_data[index])
	print(muon_kinematics_at_plane[i][0])
	
	bin_index_of_input_mom = np.where(mom_bins==find_nearest(mom_bins, muon_kinematics_at_plane[i][0]))[0][0]

	twoD_KDE_then = KDE_values_array[bin_index_of_input_mom]

	# print(np.shape(twoD_KDE_then))

	# print(twoD_KDE_then)
	raveled = np.ravel(twoD_KDE_then)

	# print(np.shape(raveled))

	partner_array = [i for i in range(0,np.shape(raveled)[0])]
	# print(partner_array)

	pick = np.random.choice(partner_array,replace=True, p=raveled/np.sum(raveled))

	bins_in_2d = np.where(twoD_KDE_then == raveled[pick])
	# print(bins_in_2d)
	# print(bins_in_2d[0][0], bins_in_2d[1][0])

	# print(log_frac_bins[bins_in_2d[0][0]], log_angle_bins[bins_in_2d[1][0]])

	random_log_frac = log_frac_step*np.random.uniform() - log_frac_step/2
	random_log_angle = log_angle_step*np.random.uniform() - log_angle_step/2

	log_frac_sampled = log_frac_bins[bins_in_2d[0][0]] + random_log_frac
	log_angle_sampled = log_angle_bins[bins_in_2d[1][0]] + random_log_angle

	# generated = np.append(generated, [[input_mom, log_frac_sampled, log_angle_sampled]], axis=0)


	angle_to_scatter = math.exp(log_angle_sampled)
	fraction_of_mom_lost = math.exp(log_frac_sampled)
	print(angle_to_scatter, fraction_of_mom_lost)

	#generate random phi

	phi = np.random.uniform()*math.pi

	# print(phi)


	momentum = [0,0,(1 - fraction_of_mom_lost) * muon_kinematics_at_plane[i][0]]

	print(momentum)

	theta_x = angle_to_scatter

	mom_roted_theta_x_around_y = [momentum[0]*np.cos(theta_x)+momentum[2]*np.sin(theta_x),momentum[1],-momentum[0]*np.sin(theta_x)+momentum[2]*np.cos(theta_x)]

	print(mom_roted_theta_x_around_y)

	mom_roted_theta_x_around_y_rotated_phi_around_z = [mom_roted_theta_x_around_y[0]*np.cos(phi)-mom_roted_theta_x_around_y[1]*np.sin(phi),mom_roted_theta_x_around_y[0]*np.sin(phi)+mom_roted_theta_x_around_y[1]*np.cos(phi),mom_roted_theta_x_around_y[2]]

	print(mom_roted_theta_x_around_y_rotated_phi_around_z)

	mom_in_muon_frame = mom_roted_theta_x_around_y_rotated_phi_around_z



	theta_x = np.arctan(np.divide(-muon_kinematics_at_plane[i][5],muon_kinematics_at_plane[i][7]))


	mom_in_muon_frame_roted_theta_x_around_y = [mom_in_muon_frame[0]*np.cos(-theta_x)+mom_in_muon_frame[2]*np.sin(-theta_x),mom_in_muon_frame[1],-mom_in_muon_frame[0]*np.sin(-theta_x)+mom_in_muon_frame[2]*np.cos(-theta_x)]

	theta_y = np.arctan(np.divide(muon_kinematics_at_plane[i][6],muon_kinematics_at_plane[i][7]))

	mom_after_scatter_in_forward_frame = [mom_in_muon_frame_roted_theta_x_around_y[0],mom_in_muon_frame_roted_theta_x_around_y[1]*np.cos(-theta_y)-mom_in_muon_frame_roted_theta_x_around_y[2]*np.sin(-theta_y),mom_in_muon_frame_roted_theta_x_around_y[1]*np.sin(-theta_y)+mom_in_muon_frame_roted_theta_x_around_y[2]*np.cos(-theta_y)]




	save(1,muon_kinematics_at_plane[i][1],muon_kinematics_at_plane[i][2],muon_kinematics_at_plane[i][3],muon_kinematics_at_plane[i][4],mom_after_scatter_in_forward_frame[0],mom_after_scatter_in_forward_frame[1],mom_after_scatter_in_forward_frame[2],99,99)

	muon_history = [[muon_kinematics_at_plane[i][1],muon_kinematics_at_plane[i][2],muon_kinematics_at_plane[i][3],muon_kinematics_at_plane[i][4],mom_after_scatter_in_forward_frame[0],mom_after_scatter_in_forward_frame[1],mom_after_scatter_in_forward_frame[2]]]

	# print(np.shape(muon_history))

	muon_scattering_history_scatter = np.append(muon_scattering_history_scatter,[muon_history],axis=0)

	quit()

muon_scattering_history = np.append(muon_scattering_history_pre_scatter,muon_scattering_history_scatter, axis=1)

print(muon_scattering_history, np.shape(muon_scattering_history))


np.save('muon_scattering_history_%d'%jobid,muon_scattering_history)


f_save.Write()
f_save.Close()








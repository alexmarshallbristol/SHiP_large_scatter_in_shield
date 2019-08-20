import ROOT 
import numpy as np
import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import math
from matplotlib.colors import LogNorm
import glob 


f = ROOT.TFile.Open('msc_test.root')
# Get ya tree
tree = f.Get("cbmsim")

# Get length of tree
N = tree.GetEntries()

print(int(N), 'GetEntries')
	# # For every entry in tree

data_array = np.empty((0,3))
for i in xrange(N):
# for i in range(0,5):
	if i % 10000 == 0: print(i)
	tree.GetEntry(i)

	mctracks = tree.MCTrack
	vetopoint = tree.vetoPoint

	check = False
	print_this_info = False
	# print(' ')
	# for m in mctracks:
	# 	# help(m)
	# 	# pz_in = m.GetPz()
	# 	# print(pz_in, m.GetPdgCode())
	# 	# if pz_in < 0:
	# 		# print(file,m.GetStartX(),m.GetStartY(), m.GetStartZ())
	# 		# quit(

	# 	if m.GetProcID() != 0:
	# 		# help(m)
	# 		print(i, m.GetProcID(),m.GetProcName(), m.GetPdgCode())
	# 		print_this_info = True
	# 	check = True

	# 	continue
		
	# print('here', pz_in_2)
	# quit()

	# if check == True:
	for m in vetopoint:
		# help(m)
		# print(m.PdgCode(), m.GetTrackID())
		if m.PdgCode() == 13 and m.GetTrackID() == 0:
			# data_array = np.append(data_array, [[mctracks[0].GetPz() ,m.GetX(), m.GetY(), m.GetZ(), m.GetPx(), m.GetPy(), m.GetPz()]],axis=0)
			# print(mctracks[0].GetPz() ,m.GetX(), m.GetY(), m.GetZ(), m.GetPx(), m.GetPy(), m.GetPz())
			# print(' ')

			# print(mctracks[0].GetPz())
			# print(np.sqrt(np.add(m.GetPx()**2,np.add(m.GetPy()**2,m.GetPz()**2))))

			# # print(np.arccos(mctracks[0].GetPz()*m.GetPz()))

			# print(np.arccos((mctracks[0].GetPz()*m.GetPz())/(np.sqrt(np.add(m.GetPx()**2,np.add(m.GetPy()**2,m.GetPz()**2)))*mctracks[0].GetPz())))

			data_array = np.append(data_array,[[mctracks[0].GetPz(),np.sqrt(np.add(m.GetPx()**2,np.add(m.GetPy()**2,m.GetPz()**2))),np.arccos((mctracks[0].GetPz()*m.GetPz())/(np.sqrt(np.add(m.GetPx()**2,np.add(m.GetPy()**2,m.GetPz()**2)))*mctracks[0].GetPz()))]],axis=0)
			# if print_this_info == True:
			# 	print('initial mom',mctracks[0].GetPz())
			# 	print('final mom',np.sqrt(np.add(m.GetPx()**2,np.add(m.GetPy()**2,m.GetPz()**2))))
			# 	print('scatter angle',np.arccos((mctracks[0].GetPz()*m.GetPz())/(np.sqrt(np.add(m.GetPx()**2,np.add(m.GetPy()**2,m.GetPz()**2)))*mctracks[0].GetPz())))
			# 	print(' ')

print('Complete.')

np.save('Scattering_momentum',data_array)


# 142






# 
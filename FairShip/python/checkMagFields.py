import ROOT,sys
import rootUtils as ut
import shipunit as u
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# def run():
#   fGeo = ROOT.gGeoManager
#   n = fGeo.FindNode(5, 5, -6620)
#   f = n.GetVolume().GetMaterial()
#   print(f)
#   f = n.GetVolume().GetField()
#   # if f:
#   print(f.GetFieldValue()[0], f.GetFieldValue()[1],f.GetFieldValue()[2])


def run(z):
  fGeo = ROOT.gGeoManager

  data = np.empty((0,3))
  # [x,y,material]

  for i in range(-100, 100):
    for j in range(-100, 100):
      # print(i,j,z)
      n = fGeo.FindNode(i*10,j*10,z)
      f = str(n.GetVolume().GetMaterial())

      f =f[27:-16]
      # <ROOT.TGeoMaterial object ("iron") at 0xd465510>
      # print(f)
      if f == 'air':
        # print()
        data = np.append(data, [[i*10,j*10,0]], axis=0)
      elif f == '"iron':
        data = np.append(data, [[i*10,j*10,1]], axis=0)
      elif f == 'steel':
        data = np.append(data, [[i*10,j*10,3]], axis=0)
      else:
        print(f)
        data = np.append(data, [[i*10,j*10,2]], axis=0)


      # print(f)
      # quit()
      # # f = n.GetVolume().GetField()
      # # print(f.GetFieldValue()[0], f.GetFieldValue()[1],f.GetFieldValue()[2])


  print(np.shape(np.where(data[:,2]==1)))

  iron = data[np.where(data[:,2]==1)]
  print(np.shape(iron))

  if np.shape(iron)[0] > 0:
    print('min x',np.amin(iron[:,0]))
    print('max x',np.amax(iron[:,0]))
    print('min y',np.amin(iron[:,1]))
    print('max y',np.amax(iron[:,1]))



  plt.scatter(data[:,0], data[:,1], c=data[:,2],edgecolors='none')
  plt.xlim(-500,500)
  plt.ylim(-500,500)
  plt.colorbar()
  plt.savefig('/afs/cern.ch/user/a/amarshal/GEANT_fairship_geo/here_full.png')



def check_material(x,y,z):
  fGeo = ROOT.gGeoManager

  n = fGeo.FindNode(x,y,z)
  f = str(n.GetVolume().GetMaterial())

  f =f[27:-16]

  if f == '"iron':
    f = f[1:]

  return f


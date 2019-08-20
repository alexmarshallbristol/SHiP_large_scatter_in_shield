
import numpy as np

import os

import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-jobid', action='store', dest='jobid', type=int,
					help='jobid')
results = parser.parse_args()
jobid = int(results.jobid)

# z_position = str(np.random.randint(-7074,high=-3471) + np.random.uniform())
z_position = str(np.random.randint(-6542,high=-3471) + np.random.uniform())
# z_position = str(-7065)
print(z_position)


pwd = os.getcwd() + '/'

f_start= open("FairShip/shipgen/FixedTargetGenerator_start.txt","r")
f_end= open("FairShip/shipgen/FixedTargetGenerator_end.txt","r")

f_new = open("FairShip/shipgen/FixedTargetGenerator.txt","w")
f_new = open("FairShip/shipgen/FixedTargetGenerator.txt","a+")

f_new.write(f_start.read())
f_new.write(pwd)
f_new.write(f_end.read())


os.remove("FairShip/shipgen/FixedTargetGenerator.cxx")
os.rename("FairShip/shipgen/FixedTargetGenerator.txt","FairShip/shipgen/FixedTargetGenerator.cxx")

# quit()


f_start= open("FairShip/passive/ShipMuonShield_start.txt","r")
f_end= open("FairShip/passive/ShipMuonShield_end.txt","r")

f_new = open("FairShip/passive/ShipMuonShield.txt","w")
f_new = open("FairShip/passive/ShipMuonShield.txt","a+")

z_position2 = str(float(z_position) - 1)

f_new.write(f_start.read())
f_new.write(z_position2)
f_new.write(f_end.read())


os.remove("FairShip/passive/ShipMuonShield.cxx")
os.rename("FairShip/passive/ShipMuonShield.txt","FairShip/passive/ShipMuonShield.cxx")






f_start= open("FairShip/muonShieldOptimization/run_fixedTarget_start.txt","r")
f_end= open("FairShip/muonShieldOptimization/run_fixedTarget_end.txt","r")

f_new = open("FairShip/muonShieldOptimization/run_fixedTarget.txt","w")
f_new = open("FairShip/muonShieldOptimization/run_fixedTarget.txt","a+")

f_new.write(f_start.read())
f_new.write(z_position)
f_new.write(f_end.read())


os.remove("FairShip/muonShieldOptimization/run_fixedTarget.py")
os.rename("FairShip/muonShieldOptimization/run_fixedTarget.txt","FairShip/muonShieldOptimization/run_fixedTarget.py")





# np.save("scatter_z_value_JOBID",float(z_position)) # potentially add to a shared file on eos 

details = str(jobid) + " " + z_position

details_array = [jobid, float(z_position)]

np.save('Sensitive_plane_position',details_array)



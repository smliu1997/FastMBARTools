import numpy as np
import pandas as pd
import simtk.openmm as mm
import simtk.unit as unit
import simtk.openmm.app as app
import matplotlib.pyplot as plt
import mdtraj
import math
import sys

"""
Run umbrella sampling for butane. 

Code is adapted from FastMBAR tutorial. 
"""

psf = app.CharmmPsfFile('data/butane.psf')
pdb = app.PDBFile('data/butane.pdb')
params = app.CharmmParameterSet('data/top_all35_ethers.rtf', 'data/par_all35_ethers.prm')

## creay an OpenMM system
system = psf.createSystem(params, nonbondedMethod=app.NoCutoff)

## add a harmonic biasing potential on butane dihedral to the OpenMM system
bias_torsion = mm.CustomTorsionForce("0.5*K*dtheta^2; dtheta = min(diff, 2*Pi-diff); diff = abs(theta - theta0)")
bias_torsion.addGlobalParameter("Pi", math.pi)
bias_torsion.addGlobalParameter("K", 1.0)
bias_torsion.addGlobalParameter("theta0", 0.0)
## 3, 6, 9, 13 are indices of the four carton atoms in butane, between which
## the dihedral angle is biased.
bias_torsion.addTorsion(3, 6, 9, 13)
system.addForce(bias_torsion)

## save the OpenMM system of butane
with open("output/system.xml", 'w') as file_handle:
    file_handle.write(mm.XmlSerializer.serialize(system))

## read the OpenMM system of butane
with open("output/system.xml", 'r') as file_handle:
    xml = file_handle.read()
system = mm.XmlSerializer.deserialize(xml)

## read psf and pdb file of butane
psf = app.CharmmPsfFile("data/butane.psf")
pdb = app.PDBFile('data/butane.pdb')

## intergrator parameters
T = 298.15*unit.kelvin  ## temperature
fricCoef = 10/unit.picoseconds ## friction coefficient
stepsize = 1*unit.femtoseconds ## integration step size

## M centers of harmonic biasing potentials
M = 20
theta0 = np.linspace(-math.pi, math.pi, M, endpoint=False)

## the main loop to run umbrella sampling window by window
for i in range(M):
    print(f"sampling at theta0 index: {i} out of {M}")
    ## set simulation
    integrator = mm.LangevinIntegrator(T, fricCoef, stepsize)
    platform = mm.Platform.getPlatformByName('CUDA')
    properties = {'Precision': 'mixed'}
    simulation = app.Simulation(psf.topology, system, integrator, platform, properties)

    ## set force constant K for the biasing potential.
    ## the unit here is kJ*mol^{-1}*nm^{-2}, which is the default unit used in OpenMM
    K = 100
    simulation.context.setParameter("K", K)

    ## set the center of the biasing potential
    simulation.context.setParameter("theta0", theta0[i])

    ## minimize
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(T)

    ## initial equilibrium
    simulation.step(10000)

    ## sampling production. trajectories are saved in dcd files
    dcd_file_name = f"output/traj/traj_{i}.dcd"
    simulation.reporters.append(app.DCDReporter(dcd_file_name, reportInterval=100))
    simulation.reporters.append(app.StateDataReporter(sys.stdout, reportInterval=10000, step=True, time=True, 
                                                      potentialEnergy=True, temperature=True))
    simulation.step(100000)

topology = mdtraj.load_psf("./data/butane.psf")
df = pd.DataFrame()

for i in range(M):
    traj = mdtraj.load_dcd(f"output/traj/traj_{i}.dcd", topology)
    theta = mdtraj.compute_dihedrals(traj, [[3, 6, 9, 13]])
    column_i = round(theta0[i], 6)
    df[str(column_i)] = theta.ravel()
df.to_csv('output/dihedral.csv', index=False)



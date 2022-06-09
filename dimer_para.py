import numpy as np
import readdy
import os
import sys

N_PD=1E-6*6.02E23*1e-15*(1)**3
D_MCP=(25/2)**2/(4*3)*1E-12*1E9
D_MCP=D_MCP
D_PD=1E-4*1E-5*1E9#https://www.gsi-net.com/en/publications/gsi-chemical-database/single/324-CAS-67630.html
D_PD=D_PD/10

# N_PD=100 
#N_PD=10
N_PD=20
N_MCP=2

#steps=800000000*3
steps=800000000*3
size=200
timestep=0.1

Ea=float(sys.argv[1])
n=float(sys.argv[2])


system = readdy.ReactionDiffusionSystem([size, size,size], temperature=298*readdy.units.kelvin)
system.add_species("MCP", D_MCP)
system.add_species("PD", D_PD)

system.potentials.add_harmonic_repulsion("MCP", "MCP", force_constant=10,interaction_distance=60)

if Ea == 0:
	system.potentials.add_harmonic_repulsion("MCP", "PD", force_constant=5,interaction_distance=62)
else:
	system.potentials.add_weak_interaction_piecewise_harmonic("MCP", "PD", force_constant=5, 
				desired_distance=31, depth=Ea, cutoff=33)

#simulation = system.simulation(kernel="CPU")
simulation = system.simulation(kernel="SingleCPU")


simulation.output_file = f"dimer_para_{Ea}_{n}_20.h5"
simulation.record_trajectory(stride=10000)
#simulation.reaction_handler = "UncontrolledApproximation"
#simulation.kernel_configuration.n_threads = 2
#simulation.reaction_handler=False
simulation.evaluate_topology_reactions = False
simulation.evaluate_forces=True
simulation.evaluate_reaction=False


#save_options = {
#    'name': "energy",
#    'chunk_size': 500
#}

#simulation.observe.energy(1000)

X = (np.random.random((int(N_MCP), 3))-0.5)*size
simulation.add_particles(type="MCP", positions=X)
X = (np.random.random((int(N_PD), 3))-0.5)*size
simulation.add_particles(type="PD", positions=X)

simulation.run(n_steps=steps, timestep=timestep)

#trajectory = readdy.Trajectory(f"dimer_para_{Ea}.h5")
#trajectory.convert_to_xyz(particle_radii={'MCP': 150,'PD':1})

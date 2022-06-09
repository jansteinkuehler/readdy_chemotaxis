#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import readdy
import os
import sys
from datetime import datetime
from itertools import product


N_PD=1E-6*6.02E23*1e-15*(1)**3
#D_MCP=(25/2)**2/(4*3)*1E-12*1E9
D_MCP=0.01
D_PD=1E-4*1E-5*1E9#https://www.gsi-net.com/en/publications/gsi-chemical-database/single/324-CAS-67630.html
D_PD=D_PD/10

N_PD=100
N_MCP=15
N_PD_S=1

#PD_rate=0.000005
#MCP_rate=0.000001

PD_rate=float(sys.argv[1])/1000
MCP_rate=float(sys.argv[2])/1000
Ea=float(sys.argv[3])
Ea=float(sys.argv[3])
n_rep=float(sys.argv[4])

# Standard steps
steps=1000000000
# For MCP_rate=0
#steps=400000000


timestep=0.1


# In[45]:


sizex=150
sizey=1500
sizez=150


# In[46]:


steps=steps

now = datetime.now()

#Ea = datetime.timestamp(now)

system = readdy.ReactionDiffusionSystem([sizex, sizey,sizez], temperature=298*readdy.units.kelvin,
                                        periodic_boundary_conditions=[False, False, False])

system.add_species("MCP", D_MCP)
system.add_species("PD", D_PD)
system.add_species("PD_S", 0)
system.potentials.add_harmonic_repulsion("MCP", "MCP", force_constant=10,
                                         interaction_distance=60)


if Ea == 0:
	system.potentials.add_harmonic_repulsion("MCP", "PD", force_constant=5,interaction_distance=62)
else:
	system.potentials.add_weak_interaction_piecewise_harmonic("MCP", "PD", force_constant=5, 
				desired_distance=31, depth=Ea, cutoff=33)

#simulation.observe.rdf(
#    stride=200, 
#    bin_borders=np.linspace(20, 150, 1), 
#    types_count_from=["MCP"], 
#    types_count_to=["PD"], 
#    particle_to_density=1./system.box_volume,save=rdf_save_options)
#,
#    callback=rdf_callback)


# In[47]:


system.potentials.add_box(
    particle_type="PD_S", force_constant=10., origin=np.array([-sizex/2, -sizey/2, -sizez/2])*.9, 
    extent=np.array([sizex, sizey, sizez])*0.9
)


# In[48]:


system.potentials.add_box(
    particle_type="PD", force_constant=10., origin=np.array([-sizex/2, -sizey/2, -sizez/2])*.9, 
    extent=np.array([sizex, sizey, sizez])*0.9
)


# In[49]:


system.potentials.add_box(
    particle_type="MCP", force_constant=10., origin=np.array([-sizex/2, -sizey/2, -sizez/2])*.9, 
    extent=np.array([sizex, sizey, sizez])*0.9
)


# In[50]:

if PD_rate > 0:
	system.reactions.add_fission(
	    name="fis", type_from="PD_S", type_to1="PD_S", type_to2="PD", rate=PD_rate, product_distance=100,
	    weight1=0, weight2=1
	)

if MCP_rate > 0:
	system.reactions.add_fusion(
	    name="fus", type_from1="MCP", type_from2="PD", type_to="MCP", rate=MCP_rate, educt_distance=62,
	    weight1=0, weight2=1
	)


# In[51]:


#simulation = system.simulation(kernel="SingleCPU")
simulation = system.simulation(kernel="CPU")
simulation.output_file = f"gradient_R1R2_{PD_rate*1000}_{MCP_rate*1000}_{Ea}_{n_rep}.h5"

if os.path.exists(simulation.output_file):
	simulation.load_particles_from_latest_checkpoint(f"gradient_R1R2_{PD_rate*1000}_{MCP_rate*1000}_{Ea}_{n_rep}")

#simulation.record_trajectory(stride=1000, chunk_size=1000)
simulation.record_trajectory(stride=10000)
#simulation.reaction_handler = "UncontrolledApproximation"
simulation.kernel_configuration.n_threads = 4
#simulation.kernel_configuration.n_threads = 8
#simulation.reaction_handler=False
simulation.evaluate_topology_reactions = False
simulation.evaluate_forces=True

                                                          
#simulation.observe.energy(10)


# In[52]:



#r=(np.arange(0,N_PD_S,1)-0.5)*sizex/2
#posxy=np.array(list(product(r,repeat=2)))
#pos=np.tile(np.zeros(3),(len(posxy),1))

#for e,n in zip(pos,range(len(posxy))):
#    e[1:3]=posxy[n]
#    e[0]=5
    
#X = (np.ara((int(N_MCP), 3))-0.5)*sizex

#Y = (np.zero((int(N_PD_S), 3))-0.5)*sizey

#Z = (np.zero((int(N_PD_S), 3))-0.5)*sizez

#X = (np.random.random((int(N_MCP), 3))-0.5)*np.array([sizex,sizey,sizez])
X = np.zeros((int(N_MCP),3))
X[:,1]=np.linspace(-sizey/2,sizey/2,int(N_MCP))
simulation.add_particles(type="MCP", positions=X)

simulation.add_particle("PD_S",[0.,-sizey/2*0.9,0.])

simulation.make_checkpoints(stride=100000, output_directory=f"gradient_R1R2_{PD_rate*1000}_{MCP_rate*1000}_{Ea}_{n_rep}",max_n_saves=10)

simulation.run(n_steps=steps, timestep=timestep)



t = readdy.Trajectory(simulation.output_file)



#t.convert_to_xyz(particle_radii={'PD_S': 10,'PD':1,'MCP':62})


# In[ ]:


#print(f"vmd -e {simulation.output_file}.xyz.tcl")


# In[ ]:





# In[ ]:





# In[ ]:





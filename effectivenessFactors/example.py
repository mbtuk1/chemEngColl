"""

    This example for the usage of the LeeSolver class is based Fig.4 and
    Fig. 5.

    Copyright (C) 2020  Markus Boesenhofer

    When using the script, please cite:
    ------------------------------------
    Bösenhofer, M. and Harasek, M. (2021): Non-isothermal effectiveness
    factors in thermo-chemical char conversion, Carbon Resources Conversion,
    in press, DOI: 10.1016/j.crcon.2021.01.004.

    URL: https://doi.org/10.1016/j.crcon.2021.01.004

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import LeeSolver as LS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# create solver object
solver = LS.nonIsothermal()

###############################################################################
# Settings for beta, gamma, and labels
###############################################################################

#O2
gamma=[
25.177202604317,
18.8829019532378,
15.1063215625902,
12.5886013021585,
10.7902296875644
]
beta=[
0.721738635949172,
0.188895402096882,
0.071825664699807,
0.033902855120891,
0.01838767367177
]

labels=[
'T1',
'T2',
'T3',
'T4',
'T5'
]

###############################################################################
# Calculate and plot data for Fig. 5
###############################################################################

print('Creating Figure 5...')

# Thiele modulus
k = 2.0

# Number of discretization points
points = 500

# Create figures
fig1=plt.figure()
ax1=fig1.add_subplot(1,1,1)
ax1.set_title('Fig. 6 - bottom')

fig2=plt.figure()
ax2=fig2.add_subplot(1,1,1)
ax2.set_title('Fig. 6 - top')

# Calculate the concentartion and temperature profiles
for i, (betai, gammai) in enumerate(zip(beta,gamma)):
	roots = solver.solveH1(k,betai,gammai)
	for ri in roots:
		C,T = solver.radialProfiles(ri,k,betai,gammai,points,True)
		print(labels[i]+' eta: ',solver.calcEta(ri,k,betai,gammai,True))
		ax1.plot(np.linspace(0,1,points),C,label=labels[i])
		ax2.plot(np.linspace(0,1,points),T,label=labels[i])

	rootsH2 = solver.solveH2(k,betai,gammai)
	for ri in rootsH2:
		C,T = solver.radialProfiles(ri,k,betai,gammai,points,False)
		print(labels[i]+' eta: ',solver.calcEta(ri,k,betai,gammai,False))
		ax1.plot(np.linspace(ri,1,points),C,label=labels[i])
		ax2.plot(np.linspace(ri,1,points),T,label=labels[i])

# format concentration plot
ax1.legend()
ax1.set_xlabel('normalized radius (-)')
ax1.set_ylabel('normalized concentration (-)')
ax1.grid(which='major',ls='--',c='k')
fig1.savefig('concentrationProfile.png',dpi=150)

# format temperature plot
ax2.legend()
ax2.set_xlabel('normalized radius (-)')
ax2.set_ylabel('normalized temperature (-)')
ax2.grid(which='major',ls='--',c='k')
fig2.savefig('temperatureProfile.png',dpi=150)


###############################################################################
# Calculate and plot data for Fig. 4
###############################################################################

print('Creating Figure 4...')

# increase the number of intervalls if solution doesn't look good
thieleM = np.logspace(-3,2,100)

fig3=plt.figure()
ax3=fig3.add_subplot(1,1,1)
ax3.set_title('Fig. 7')

# determine effectiveness factor profiles
for i, (betai, gammai) in enumerate(zip(beta,gamma)):
	phi, eta = solver.etaProfile(betai,gammai,thieleM)
	ax3.plot(phi,eta,label=labels[i])

ax3.set_yscale('log')
ax3.set_xscale('log')
ax3.legend()
ax3.set_xlabel('Thiele modulus $\Psi$ (-)')
ax3.set_ylabel('effectiveness factor $\eta$ (-)')
ax3.grid(which='major',ls='--',c='k')
fig3.savefig('effectivenessFactorProfiles.png',dpi=150)

# show the figures
plt.show()



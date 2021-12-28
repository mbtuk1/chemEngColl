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
solver = LS.isothermal()

###############################################################################
# Calculate and plot data Fig. 1
###############################################################################

print('Creating Figure 1...')

# Create figures
fig1=plt.figure()
ax1=fig1.add_subplot(1,1,1)
ax1.set_title('Fig. 2')
k=1
# Calculate the concentartion profiles
solver.setOrder(1)
sol = solver.solve(k)
C0 = sol.y[0][0]
r,C = solver.radialProfiles(C0,k)
print('eta: ',solver.calcEta(sol))
ax1.plot(r,C)

# format concentration plot
ax1.set_xlabel('normalized radius (-)')
ax1.set_ylabel('normalized concentration (-)')
ax1.set_yticks(np.linspace(0.85,1,4))
ax1.set_yticklabels(np.round(np.linspace(0.85,1,4),2))
ax1.grid(which='major',ls='--',c='k')
fig1.savefig('isothermalConcentrationProfile.png',dpi=150)

###############################################################################
# Calculate and plot data for Fig. 2
###############################################################################

print('Creating Figure 2...')

# increase the number of intervalls if solution doesn't look good
thieleM = np.logspace(-3,1,150)

fig2=plt.figure()
ax2=fig2.add_subplot(1,1,1)
ax2.set_title('Fig. 4')

# determine effectiveness factor profiles
solver.setOrder(1)
phi, eta = solver.etaProfile(thieleM)
ax2.plot(phi,eta)

ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xlabel('Thiele modulus $\Psi$ (-)')
ax2.set_ylabel('effectiveness factor $\eta$ (-)')
ax2.set_yticks(np.linspace(0.2,1,9))
ax2.set_yticklabels(np.round(np.linspace(0.2,1,9),1))
ax2.grid(which='major',ls='--',c='k')
fig2.savefig('isothermalEffectivenessFactorProfiles.png',dpi=150)


# show the figures
plt.show()



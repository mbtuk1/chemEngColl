"""
    Algorithm for the solution of non-linear boundary value problems, e.g.
    the reaction diffusion equation.

    Copyright (C) 2020  Markus Boesenhofer

    When using the script, please cite:
    ------------------------------------
    Bösenhofer, M. and Harasek, M. (2021): Non-isothermal effectiveness
    factors in thermo-chemical char conversion, Carbon Resources Conversion,
    in press, DOI: 10.1016/j.crcon.2021.01.004.

    URL: https://doi.org/10.1016/j.crcon.2021.01.004

    The script is based on the algorithom proposed in:
    ---------------------------------------------------
    Lee, J., and Kim, DH (2005): An improved shooting method for computation of
    effectiveness factors in porous catalysts, Chemical Engineering Science,
    vol. 60(20), pp. 5569-5573,
    DOI: 10.1016/j.ces.2005.05.027.

    URL: https://doi.org/10.1016/j.ces.2005.05.027

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

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import solve_bvp
from scipy import optimize
import pandas as pd

###############################################################################
# progress bar for integration status
###############################################################################

def progress(ith,tot,item='eta: ',lenbar=70):
    progress = round(ith/tot*100,1)
    fillBar = int(progress*lenbar/100.0)
    bar = '#' * fillBar + '-' * (lenbar - fillBar)
    print(f'\r{item} [{bar}] {progress}% done',end="\r")
    if ith == tot:
        print()

###############################################################################
# non-isothermal solver
###############################################################################

class nonIsothermal:
    '''
    Solver for the non-isothermal effectiveness factors in spherical particles
    '''
    def __init__(self,nPoints=500):
        '''
        Parameters
        --------------------------
        nPoints - int, number of intervals for determining the intervals
        '''
        self.nPoints = nPoints
        self.Psi = 1e-9

    def F(self,y,betai,gammai):
        '''
        Holds the ode system solved for the species concentration profile
        '''
        return y*np.exp(gammai*betai*(1.0-y)/(1.0+betai*(1.0-y)))

    def P(self,zeta,phi):
        '''
        Holds the gradient at the position where C = Psi
        '''
        return self.Psi*(phi/np.tanh(phi*zeta)-1.0/zeta)

    def ode(self,x,y,phi,betai,gammai):
        '''
        Holds the ode system solved for the species concentration profile
        '''
        k = phi
        return [y[1], -2.0*y[1]/x + k**2.0*self.F(y[0],betai,gammai)]

    def optH1Func(self,alphai,phi,betai,gammai):
        '''
        Function to solve for the roots
        '''
        sol = solve_ivp(self.ode,(1e-9,1),[alphai,0], \
                        args=(phi,betai,gammai),method='LSODA', \
                        rtol=1e-6,atol=1e-12, \
                        dense_output=True)
        z=sol.sol([1])
        return z.T[-1,0] - 1.0

    def optH2Func(self,zeta,phi,betai,gammai):
        '''
        Function to solve for the roots
        '''
        sol = solve_ivp(self.ode,(zeta,1),[self.Psi,self.P(zeta,phi)], \
                        args=(phi,betai,gammai),method='LSODA', \
                        rtol=1e-6,atol=1e-12, \
                        dense_output=True)
        z=sol.sol([1])
        return z.T[-1,0] - 1.0

    def getH1Bounds(self,phi,betai,gammai):
        '''
        Determine the bounds where the roots are inbetween
        '''
        # define start concentrations
        alpha = np.linspace(1e-9,1,self.nPoints)

        # get C-1 values at R = 1 for the start concentrations
        h1 = []
        for alphai in alpha:
            h1.append(self.optH1Func(alphai,phi,betai,gammai))

        # remove trailing nans
        index = 0
        for i in range(len(h1)):
            if ~np.isnan(h1[i]):
                index = i
                break
        if index > 0:
            for i in range(len(h1)):
                if np.isnan(h1[i]):
                    h1[i] = h1[index]
        # remove last nans
        index = 0
        for i in range(len(h1)):
            if np.isnan(h1[i]):
                index = i
                break

        for i in range(len(h1)):
            if np.isnan(h1[i]):
                h1[i] = h1[index-1]

        # determine intervalls inbetween which the sign changes
        asign = np.sign(h1)
        signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
        signchange[0] = 0 # set the first element to zero
        alphaRoots = np.where(abs(signchange) == 1.0)[0]

        lowB =[]
        highB=[]
        for i in alphaRoots:
            # no case selection necessary since alpha is monotone
            # increasing
            if ~np.isnan(h1[i]):
                highB.append(alpha[i])
                lowB.append(alpha[i-1])

        return lowB, highB

    def getH2Bounds(self,phi,betai,gammai):
        '''
        Determine the positions where C reaches Psi in the particle
        '''
        # define start positions
        zeta = np.linspace(1e-9,1,self.nPoints)

        # get C-1 values at R = 1 for the start concentrations
        h2 = []
        for zetai in zeta:
            h2.append(self.optH2Func(zetai,phi,betai,gammai))

        # remove trailing nans
        index = 0
        for i in range(len(h2)):
            if ~np.isnan(h2[i]):
                index = i
                break
        if index > 0:
            for i in range(len(h2)):
                if np.isnan(h2[i]):
                    h2[i] = h2[index]
        # remove last nans
        index = 0
        for i in range(len(h2)):
            if np.isnan(h2[i]):
                index = i
                break

        for i in range(len(h2)):
            if np.isnan(h2[i]):
                h2[i] = h2[index-1]

        # determine intervalls inbetween which the sign changes
        asign = np.sign(h2)
        signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
        signchange[0] = 0 # set the first element to zero
        alphaRoots = np.where(abs(signchange) == 1.0)[0]

        lowB =[]
        highB=[]
        for i in alphaRoots:
            # no case selection necessary since alpha is monotone
            # increasing
            if ~np.isnan(h2[i]):
                highB.append(zeta[i])
                lowB.append(zeta[i-1])

        return lowB, highB


    def solveH1(self,phi,betai,gammai):
        '''
        Solve for the function roots to get the C(0) values
        '''
        # obtain bounds inbetween which the roots are
        lowB,highB = self.getH1Bounds(phi,betai,gammai)

        # calculate function roots
        roots = []
        for low,high in zip(lowB,highB):
            sol,err = optimize.brentq(self.optH1Func, low, high, \
                                      args=(phi,betai,gammai), \
                                      xtol=1e-20, rtol=1e-10, \
                                      full_output=True, \
                                      maxiter=1000000)
            if err.converged:
                roots.append(sol)

        return roots

    def solveH2(self,phi,betai,gammai):
        '''
        Solve for the function roots to get the zeta(Psi) values
        '''
        # obtain bounds inbetween which the roots are
        lowB,highB = self.getH2Bounds(phi,betai,gammai)

        # calculate function roots
        roots = []
        for low,high in zip(lowB,highB):
            sol,err = optimize.brentq(self.optH2Func, low, high, \
                                      args=(phi,betai,gammai), \
                                      xtol=1e-20, rtol=1e-10, \
                                      full_output=True, \
                                      maxiter=1000000)
            if err.converged:
                roots.append(sol)

        return roots

    def calcEta(self,C0,phi,betai,gammai,center=True):
        '''
        Caluclate the effectiveness factor for the given C0
        or location of C = Cmin
        '''
        eta = []
        if center:
            loc = np.linspace(1e-10,1,self.nPoints)

            sol = solve_ivp(self.ode,(1e-9,1),[C0,0], \
                            args=(phi,betai,gammai),method='LSODA', \
                            rtol=1e-10,atol=1e-15,first_step=1e-9, \
                            dense_output=True)

            C=sol.sol(loc)[0]
            Theta = 1.0/(betai*(1.0-C) + 1.0)
            eta = (3.0*np.trapz(C*np.exp(-gammai*(Theta-1.0))*(loc**2),loc))
            return eta

        else:
            loc = np.linspace(C0,1,self.nPoints)

            sol = solve_ivp(self.ode,(C0,1),[self.Psi,self.P(C0,phi)], \
                            args=(phi,betai,gammai),method='LSODA', \
                            rtol=1e-10,atol=1e-15,first_step=1e-9, \
                            dense_output=True)

            C=sol.sol(loc)[0]
            Theta = 1.0/(betai*(1.0-C) + 1.0)
            eta = (3.0*np.trapz(C*np.exp(-gammai*(Theta-1.0))*(loc**2),loc))
            return eta

    def radialProfiles(self,C0,phi,betai,gammai,nPoints=500,center=True):
        '''
        Calculate the concentration and temperature profile 
        '''
        if center:
            loc = np.linspace(1e-10,1,nPoints)

            sol = solve_ivp(self.ode,(1e-9,1),[C0,0], \
                            args=(phi,betai,gammai),method='LSODA', \
                            rtol=1e-10,atol=1e-15,first_step=1e-9, \
                            dense_output=True)

            C=sol.sol(loc)[0]
            Theta = (betai*(1.0-C) + 1.0)
            return C,Theta

        else:
            loc = np.linspace(C0,1,nPoints)

            sol = solve_ivp(self.ode,(C0,1),[self.Psi,self.P(C0,phi)], \
                            args=(phi,betai,gammai),method='LSODA', \
                            rtol=1e-10,atol=1e-15,first_step=1e-9, \
                            dense_output=True)

            C=sol.sol(loc)[0]
            Theta = (betai*(1.0-C) + 1.0)
            return C,Theta

    def etaProfile(self,beta,gamma,Thiele):
        '''
        Calculate the effectiveness factor profile 
        '''
        print(f'Calculating eta vs phi for beta={round(beta,2)} ' \
              f'and gamma={round(gamma,2)}')
        etas=[]
        phis=[]
        elems = len(Thiele)-1
        # loop over all Thiele moduli
        for ith,k in enumerate(Thiele):
            progress(ith,elems)
            # get roots objective fucntion H1
            roots = self.solveH1(k,beta,gamma)
            for ri in roots:
                etas.append(self.calcEta(ri,k,beta,gamma))
                phis.append(k)

            # get roots objective fucntion H2
            rootsH2 = self.solveH2(k,beta,gamma)
            for ri in rootsH2:
               etas.append(self.calcEta(ri,k,beta,gamma,False))
               phis.append(k)

        # remove negative values from list
        etas=np.array(etas)
        phis=np.array(phis)
        etas[etas >= 0.0]
        phis[phis >= 0.0]
        etas=etas.tolist()
        phis=phis.tolist()
        # sort the coordinate pairs - result is solution dependent!
        # the points are sorted based on the distance of their logarithmic
        # values
        phiPlot=[phis[0]]
        etaPlot=[etas[0]]

        del(phis[0])
        del(etas[0])

        # loop until all points are sorted
        while len(phis) > 0:
            dist=[]
            for phii,etai in zip(phis,etas):
                phiDist = np.log10(phiPlot[-1]) - np.log10(phii)
                etaDist = np.log10(etaPlot[-1]) - np.log10(etai)
                dist.append(np.sqrt(etaDist*etaDist+phiDist*phiDist))

            # set next element
            index = dist.index(min(dist))
            phiPlot.append(phis[index])
            etaPlot.append(etas[index])
            # delete used element from list
            del(phis[index])
            del(etas[index])

        # return the sorted Thiele modulus, eta arrays
        return phiPlot,etaPlot

"""
###############################################################################
# isothermal solver - currently not working because of issues when passing 
# arguments to the bc class member
###############################################################################
class isothermal:
    '''
    Solver for the isothermal effectiveness factors in spherical particles
    '''
    def __init__(self,nPoints=500):
        '''
        Parameters
        --------------------------
        ode - func, ode system
        bc  - func, boundary conditions
        F   - func, disturbance function
        nPoints - int, number of intervals for determining the intervals
        '''
        self.nPoints = nPoints
        self.Psi = 1e-9
        # Think about the option to pass other functions
        #self.ode = ode
        #self.bc = bc
        #self.F = F

    def ode(self,x,y,phi):
        '''
        Holds the ode system solved for the species concentration profile
        '''
        k = phi
        return [y[1], -2.0*y[1]/x + phi**2.0*self.F(y[0])]

    def F(self,y):
        '''
        Holds the ode system solved for the species concentration profile
        '''
        return -y

    def bc(self,ya,yb,y0):
        '''
        Holds the ode system solved for the species concentration profile
        '''
        y2 = -phi**2*F(y0)/6.0
        return [ya[0]-y0-y2**self.Psi**2.0,ya[1]-2.0*y2*self.Psi,yb[0]-1.0]

    def calcEta(self,C0,phi):
        '''
        Caluclate the effectiveness factor for the given C0
        or location of C = Cmi
        '''
        sol = solve_ivp(self.ode,(1e-9,1),[C0,0], \
                        args=(phi),method='LSODA', \
                        rtol=1e-10,atol=1e-15,first_step=1e-9, \
                        dense_output=True)

        if sol.success:
            x = np.linspace(0,1,self.nPoints)
            eta = (np.trapz(sol.sol(x)[0]*(x**2.0),x)/np.trapz(1.0*(x**2.0),x))
            return eta
        else:
            print('error: integration failed in eta calculation!')
            exit()


    def radialProfiles(self,C0,phi):
        '''
        Calculate the concentration and temperature profile 
        '''
        sol = solve_ivp(self.ode,(1e-9,1),[C0,0], \
                        args=(phi),method='LSODA', \
                        rtol=1e-10,atol=1e-15,first_step=1e-9, \
                        dense_output=True)

        if sol.success:
            x = np.linspace(0,1,self.nPoints)
            return x,sol.sol(x)
        else:
            print('error: integration failed in concentration profile calculation!')
            exit()


    def etaProfile(self,Thiele):
        '''
        Calculate the effectiveness factor profile 
        '''
        etas=[]
        phis=[]
        elems = len(Thiele)-1

        xinit = [self.Psi,1.0]
        yinit = [[0.1,1.0],[0.0,0.0]]

        for ith,k in enumerate(Thiele):
            progress(ith,elems)
            res = solve_bvp(self.ode,lambda ya,yb,y0 : self.bc(ya,yb,y0,k), \
                            xinit,yinit,p=[0.1], \
                            tol=1e-4, bc_tol=1e-4, max_nodes=1e5,verbose=0)
            if res.success:
                x = np.linspace(a,1,61);
                xinit = x
                yinit = res.sol(x)
                etas.append(eta)
                phis.append(k)

        return phis, etas
"""


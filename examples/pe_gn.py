# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl

from msobox.ind.rk4classic import RK4Classic
from msobox.pe.gn import GN

class MF(object):
    def ffcn(self, f, t, x, p, u):
        f[0] = - p[0]*(x[0] - p[1])

    def ffcn_dot(self, f, f_d, t, x, x_d, p, p_d, u, u_d):
        f[0] = - p[0]*(x[0] - p[1])
        f_d[0,:] = -p_d[0,:]*(x[0] - p[1]) \
                     + p[0]*p_d[1,:] \
                     - p[0]*x_d[0,:]

    def hfcn(self, h, t, x, p, u):
        h[0] = x[0]

    def hfcn_dot(self, h, h_d, t, x, x_d, p, p_d, u, u_d):
        h[0] = x[0]
        h_d[...] = x_d[...]

    def sfcn(self, s, t, x, p, u):
        s[0] = 5.e-1

    def sfcn_dot(self, s, s_d, t, x, x_d, p, p_d, u, u_d):
        s[0]   = 5.e-1
        s_d[0] = 0

mf = MF()
ind = RK4Classic(mf)

gn = GN(mf, ind, 100, 1, 2, 0, 1)

# reference solution
gn.x0[:] = 4
gn.p[:] = [3,4]
std = 0.1
gn.simulate_measurements()
pl.plot(gn.ts, gn.hs_ref[:,0], '-k')
pl.errorbar(gn.ts, gn.es[:,0], gn.ss[:,0], fmt='.')
pl.ion()
pl.pause(0.01)

# estimate
gn.x0[0] = 0
gn.p[0]  = 4
gn.p[1]  = 3

for i in range(10):
    s   = gn.step()
    s  *= 1
    gn.v  += s
    pl.title("x(t)")
    pl.plot(gn.ts, gn.hs[:, 0])
    pl.pause(0.01)


C = np.linalg.inv(gn.tJ.T.dot(gn.tJ))

for i in range(gn.NX):
    print('s%02d = %.3f +/- %.1f%%'%(i+1, gn.v[i], 100.*C[i,i]**0.5/np.abs(gn.v[i])))

for i in range(gn.NX, gn.NX + gn.NP):
    print('p%02d = %.3f +/- %.1f%%'%(i+1, gn.v[i], 100.*C[i,i]**0.5/np.abs(gn.v[i])))

print(C)

import json
print(json.dumps(gn.es.tolist()))

input("press key to continue")

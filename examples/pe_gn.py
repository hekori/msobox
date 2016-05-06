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
        s[0] = 1.e-1

    def sfcn_dot(self, s, s_d, t, x, x_d, p, p_d, u, u_d):
        s[0]   = 1.e-1
        s_d[0] = 0

MF.ffcn_dot = MF.ffcn_dot
MF.hfcn_dot = MF.hfcn_dot

mf = MF()
ind = RK4Classic(mf)

gn = GN(mf, ind, 100, 1, 2, 0, 1)

# reference solution
gn.x0[:] = 1
gn.p[:] = [3,4]
std = 0.1
gn.simulate_measurements(std)
pl.plot(gn.ts, gn.hs_ref[:,0], '-k')
pl.errorbar(gn.ts, gn.es[:,0], std, fmt='.')
pl.ion()
pl.pause(0.1)

# estimate
gn.x0[0] = 8
gn.p[0]  = 4
gn.p[1]  = 3

for i in range(5):
    s   = gn.step()
    s  *= 1
    gn.p  += s[:2]
    gn.x0 += s[2]
    pl.title("x(t)")
    pl.plot(gn.ts, gn.hs[:, 0])
    pl.pause(2)


# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
from msobox.ind.rk4classic import RK4Classic


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


mf = MF()
ind = RK4Classic(mf)

NTS = 100
P   = 3 
ts   = np.linspace(0, 1, 100)
hs   = np.zeros((100, 1))
hs_d = np.zeros((100, 1, P))

xp = np.array([2., 3., 4.])
x  = xp[:1]
p  = xp[1:]
q = np.zeros(0)

xp_d = np.zeros( xp.shape + (P,))
xp_d = np.eye(xp.size)
x_d  = xp_d[:1, :]
p_d  = xp_d[1:, :]
q_d = np.zeros( q.shape + (P,))

for i in range(ts.size-1):
    mf.hfcn_dot(hs[i, ...], hs_d[i, ...], ts[i],x, x_d, p, p_d, q, q_d)
    x[:], x_d[...] = ind.fo_forward(ts[i:i+2], x, x_d, p, p_d, q, q_d)

i = ts.size - 1
mf.hfcn_dot(hs[i, ...], hs_d[i, ...], ts[i],x, x_d, p, p_d, q, q_d)

pl.plot(ts, hs[:, 0], '-k', label='$h$')
pl.plot(ts, hs_d[:, 0, 0], '--k', label=r'$\frac{dh}{dx_0}$')
pl.plot(ts, hs_d[:, 0, 1], '-.k', label=r'$\frac{dh}{dp_1}$')
pl.plot(ts, hs_d[:, 0, 2], '-.k', label=r'$\frac{dh}{dp_2}$')
pl.legend(loc='best')

pl.show()

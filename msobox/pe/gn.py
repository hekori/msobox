# Author: Sebastian F. Walter

import os
import sys
import numpy as np
import json
import datetime

from msobox.mf.tapenade import Differentiator
from msobox.mf.fortran import BackendFortran

class GN(object):
    """Solves least squares problems via Gauss-Newton."""

    def __init__(self, mf, ind, NTS, NX, NP, NQ, NH):
        # set parameters
        self.mf        = mf
        self.ind       = ind
        self.NTS       = NTS
        self.NX        = NX
        self.NP        = NP
        self.NQ        = NQ
        self.NH        = NH
        self.ts        = np.linspace(0, 1, NTS)
        self.x         = np.zeros(NX)
        self.v         = np.zeros(NX + NP)
        self.x0        = self.v[:NX]
        self.p         = self.v[NX:]
        self.q         = np.zeros(NQ)
        self.h         = np.zeros(NH)
        self.P         = self.p.size + self.x0.size
        self.x_d       = np.zeros((NX, self.P))
        self.p_d       = np.zeros((NP, self.P))
        self.q_d       = np.zeros((NQ, self.P))
        self.h_d       = np.zeros((NH, self.P))
        self.hs        = np.zeros((NTS, NH))
        self.hs_d      = np.zeros((NTS, NH, self.P))
        self.es        = np.zeros((NTS, NH))       # eta
        self.ss        = np.zeros((NTS, NH))       # sigma
        self.F         = np.zeros((NTS, NH))
        self.J         = np.zeros((NTS, NH, self.P))

        self.hs_ref    = np.zeros((NTS, NH))


    def simulate_measurements(self):

        mf, ind      = self.mf, self.ind
        x, x0, h     = self.x, self.x0, self.h
        ts, p, q     = self.ts, self.p, self.q
        hs_ref, es   = self.hs_ref, self.es
        ss           = self.ss
        NTS          = self.NTS

        self.x[:]  = self.x0
        self.h[:]  = 0.

        for i in range(ts.size-1):
            mf.hfcn(h, ts[i], x, p, q)
            mf.sfcn(ss[i,:], ts[i], x, p, q)
            hs_ref[i, ...] =  h
            x[:] = ind.zo_forward(ts[i:i+2], x, p, q)


        i = ts.size - 1
        mf.hfcn(h, ts[i], x,  p, q)
        mf.sfcn(ss[i,:], ts[i], x, p, q)
        hs_ref[i, ...] =  h
        es[:, :] = hs_ref + np.random.randn(NTS,self.NH)*self.ss


    def step(self):
        """ compute delta_x """
        ts, x, x_d, p, p_d   = self.ts, self.x, self.x_d, self.p, self.p_d
        q, q_d, h, h_d       = self.q, self.q_d, self.h, self.h_d
        hs, hs_d             = self.hs, self.hs_d
        F, J                 = self.F, self.J
        NTS, NH, P           = self.NTS, self.NH, self.P
        NX, NP               = self.NX, self.NP
        h, h_d               = self.h, self.h_d
        x0                   = self.x0
        mf, ind              = self.mf, self.ind
        es, ss               = self.es, self.ss

        x[:]            = x0
        x_d[:, :NX]     = np.eye(NX)
        p_d[:, NX:]     = np.eye(NP)
        h[:]            = np.zeros(1)
        h_d[: :]        = np.zeros((1, P))

        for i in range(ts.size-1):
            mf.hfcn_dot(h, h_d, ts[i],x, x_d, p, p_d, q, q_d)

            hs[i, ...]   = h
            hs_d[i, ...] = h_d
            x[:], x_d[...] = ind.fo_forward(ts[i:i+2], x, x_d, p, p_d, q, q_d)

        i =ts.size - 1
        mf.hfcn_dot(self.h, h_d, ts[i], x, x_d, p, p_d, q, q_d)
        hs[i, ...]   =  h
        hs_d[i, ...] = h_d

        F = (hs - es)/ss
        J = hs_d/ss[:,:,np.newaxis]

        self.tF = tF = F.reshape((NTS*NH))
        self.tJ = tJ = J.reshape((NTS*NH, P))

        Q, R = np.linalg.qr(tJ)
        tmp1 = Q.T.dot(tF)
        tmp2 = - np.linalg.solve(R, tmp1)
        return tmp2


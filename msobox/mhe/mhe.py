# Author: Sebastian F. Walter
#         Manuel Kudruss

import os
import sys
import numpy
import json
import datetime
import tempfile
import scipy.linalg
from numpy.testing import assert_almost_equal

from mhe_plot_data import PlotData

# from msobox.mf.tapenade import Differentiator
# from msobox.mf.fortran import BackendFortran

class MHE(object):
    """
    Moving Horizon real-time state and parameter Estimation (MHE)

    as described in the paper

    "Schnelle Algorithmen fuer die Zustands- und Parameterschaetzung",
    Diehl et. al., 2006

    """
    # flag for debug output
    VERBOSE = False

    def debug_out(self, string):
        """writes output to std_out when VERBOSE==True"""
        if self.VERBOSE:
            sys.stdout.write(str(string))

    def __init__(self, mf, ind, dt=0.05, M=20, **kwargs):
        """

        Parameters
        ----------

        :mf: object with member functions mf.ffcn, mf.hfcn etc.

        :dt:  time increment of equidistant time grid
        :M:   number of nodes in the parameter estimation time grid

        """

        self.mf = mf   # model functions
        self.ind = ind # integration scheme

        # dimensions
        self.M = M

        self.NY  = mf.NY
        self.NZ  = mf.NZ
        self.NX  = self.NY + self.NZ
        self.NP  = mf.NP
        self.NU  = mf.NU
        self.NQ  = self.NU*self.M*2
        self.NQI = self.NU*2         # number of q in one control interval
        self.NH  = mf.NH

        self.P = self.NX + self.NP

        # define time grid

        #                         past          present
        #            --------- parameter est. ---->|
        #   meas     *    *    *    *    *    *    *
        #          [           M                     ]

        self.dt = dt
        self.ts = numpy.linspace(0, (self.M-1)*dt, self.M)

        # current state and workspace arrays
        # initialize state and parameter arrays
        self.x  = numpy.zeros(self.NX)
        self.y  = self.x[:self.NY]
        self.z  = self.x[self.NY:]
        self.p  = numpy.zeros(self.NP)

        # initialize controls (past time and therefore fixed)
        self.q     = numpy.zeros((self.M, self.NU))

         # values at time node 0
        self.h = numpy.zeros(self.NH)

        # tilde values at time node 0
        self.xtilde = numpy.zeros(self.NX)
        self.ytilde = self.xtilde[:self.NY]
        self.ztilde = self.xtilde[self.NY:]

        self.htilde = numpy.zeros(self.NH)

         # arrival costs of parameter estimation at time node 0
        self.xbar = numpy.zeros(self.NX)
        self.ybar = self.xbar[:self.NY]
        self.zbar = self.xbar[self.NY:]
        self.pbar = numpy.zeros(self.NP)

        # Arrival Cost Sigma_Bar Inv
        # (ACSBI.T * ACSBI)^-1 is the covariance matrix of the arrival cost
        self.ACSBI  = numpy.zeros((self.NX + self.NP, self.NX + self.NP))

        # CovEta**2 is the diagonal of the covariance matrix of the measurements at time node L
        self.CovEta = numpy.zeros(self.NH)

        # CovPenalty.T * CovPenalty is the covariance matrix of the quadratic penalty term
        self.CovPenalty = numpy.zeros((self.NX + self.NP, self.NX + self.NP))

        # Covariance matrix C = (Jc.T Jc)^{-1} of the parameter estimation
        self.C = numpy.zeros((self.NX + self.NP, self.NX + self.NP))

        # temporary variables used in self.update_arrival_cost
        self.Xx  = numpy.zeros((self.NX, self.NX))
        self.Xp  = numpy.zeros((self.NX, self.NP))
        self.Xq  = numpy.zeros((self.q.size, self.NQI))

        # FIXME: are these variables used?
        self.Hx = numpy.zeros((self.NH, self.NX))
        self.Hp = numpy.zeros((self.NH, self.NP))

        # multiple shooting variables for GGN
        self.s       = numpy.zeros((self.M, self.NX))
        self.sd      = self.s[:, :self.NY]
        self.sa      = self.s[:, self.NY:]

        # reference trajectory using true parameters
        # and true states
        self.true_model_initialized = False # needed for real model response
        self.p_ref      = numpy.zeros(self.NP) #, e.g. true parameter
        self.s_ref      = numpy.zeros(self.s.shape)

        # structured GGN matrices
        self.F1   = numpy.zeros((self.M, self.NH))
        self.F1AC = numpy.zeros(self.NX + self.NP)

        self.J1AC = numpy.zeros((self.NX + self.NP, self.NX + self.NP))

        self.J1s = numpy.zeros((self.M, self.NH, self.NX))
        self.J1p = numpy.zeros((self.M, self.NH, self.NP))

        self.F2   = numpy.zeros((self.M - 1, self.NX))

        self.J2s = numpy.zeros((self.M - 1, self.NX, self.NX))
        self.J2p = numpy.zeros((self.M - 1, self.NX, self.NP))

        # condensation matrices
        self.Z = numpy.zeros((self.M, self.NX, self.NX + self.NP))
        self.e = numpy.zeros((self.M, self.NX))
        self.Z_o = numpy.zeros((self.M, self.NX, self.NX + self.NP))
        self.e_o = numpy.zeros((self.M, self.NX))

        # condensed matrix
        tmp = self.M * self.NH  + self.ACSBI.shape[0]
        self.FC = numpy.zeros(tmp)
        self.JC = numpy.zeros((tmp, self.NX + self.NP))

        # # seed matrices for directional derivatives for GGN
        # # P: number of directional derivatives
        # # D: degree of directional derivatives
        # P, D  = self.NX + self.NP, 1
        # self.Vxd = numpy.zeros((P, D, self.y .size))
        # self.Vxa = numpy.zeros((P, D, self.z .size))
        # self.Vq  = numpy.zeros((P, D, self.q.size))
        # self.Vp  = numpy.zeros((P, D, self.p.size))

        # allocate useful constant matrices
        self.eyeNX   = numpy.zeros((self.NX, self.P))
        self.eyeNP   = numpy.zeros((self.NP, self.P))
        self.zerosNU = numpy.zeros((self.NU, self.P))
        self.eyeNX[:, :self.NX] = numpy.eye(self.NX)
        self.eyeNP[:, self.NX:] = numpy.eye(self.NP)

        self.x_dot = numpy.zeros((self.NX, self.P))
        self.p_dot = numpy.zeros((self.NP, self.P))
        self.q_dot = numpy.zeros(self.q.shape + (self.P,))

        # measurements, model response and errors
        self.etas   = numpy.zeros((self.M, self.NH))
        self.hs     = numpy.zeros((self.M, self.NH))
        self.sigmas = numpy.ones((self.M, self.NH))

        # check setup
        self.check()

        # initialize integrator
        self.plot_data = PlotData(self)

    def check(self):
        """
        Checks whether the problem setup is consistent.
        """
        if self.M < self.NX:
            raise Exception('require self.M >= self.NX but provided self.M = %d and self.NX = %d'%(self.M, self.NX))

    def shift(self):
        """
        shifts measurements, i.e., moves
        self.etas[nmess][nfun][:-1, :] = self.etas[nmess][nfun][1:, :]
        and sets the latest entries to zero
        """

        self.debug_out('-'*80+'\n')
        self.debug_out('mhe.shift()\n')
        self.debug_out('-'*80+'\n')

        # shift time grid
        self.ts[:-1] = self.ts[1:]
        self.ts[-1] = self.ts[-2] + self.dt

        # shift control grid
        # automatically repeat latest control
        self.q[:-1, :] = self.q[1:, :]
        self.q[-1, :] = self.q[-2, :]

        # shift multiple shooting values
        self.s[:-1, :] = self.s[1:, :]

        # shift reference
        self.s_ref[:-1, :] = self.s_ref[1:, :]

        # shift measurements and standard deviations
        # NOTE Latest eta = 0 and sigma = 1.0.
        #      This is used in the preparation phase to initialize F1, J1, F2, J2
        self.etas[:self.M-1, :]   = self.etas[1:, :]
        self.sigmas[:self.M-1, :] = self.sigmas[1:, :]
        self.etas[self.M-1:, :]   = 0.0
        self.sigmas[self.M-1:, :] = 1.0

        self.plot_data.shift()

    def update_arrival_cost(self):
        """
        Computes update for arrival costs as described in Section 2.1.


        The measurements and stds at
        self.etas[0, :]
        self.sigmas[0, :]
        are incorporated into the arrival cost.
        Both lists are not altered in this function.
        """

        self.debug_out('-'*80+'\n')
        self.debug_out('mhe.update_arrival_cost()\n')
        self.debug_out('-'*80+'\n')

        # renaming for convenience
        # ########################

        # ex           = self.ex
        ACSBI        = self.ACSBI
        CovPenalty   = self.CovPenalty
        Xx           = self.Xx
        Xp           = self.Xp
        x            = self.x           # state when time is at node L
        xtilde       = self.xtilde
        htilde       = self.htilde
        xbar         = self.xbar
        pbar         = self.pbar

        # temporary variables
        xp    = numpy.zeros(x.shape)

        eta, sigma = self.etas[0,:], self.sigmas[0,:]

        # STEP 1: prepare inputs for integration
        # ######################################

        t0, tend = self.ts[0], self.ts[1]

        # self.Vxd[:self.NY, 0, :self.NY]      = numpy.eye(self.NY)
        # self.Vxd[self.NX:, 0, :]             = 0.

        # if self.NZ != 0:
        #     # FIXME:
        #     # when self.NZ != 0, then z = z(x,p)
        #     # and thus a consistent Vxa has to be computed
        #     err_str  = 'need to implement self.NZ != 0\n'
        #     err_str += 'and find consistent Vxa before calling the integrator'
        #     raise NotImplementedError(err_str)

        # self.Vp[:self.NX, 0, :]            = 0.
        # self.Vp[self.NX:, 0, :]            = numpy.eye(self.NP)

        # self.Vq[...]                     = 0.

        # # STEP 2: perform integration
        # # ###########################
        # self._update_rwh_iwh(0)

        # self.integrator.plugin.dosave = False
        # xdp, xap, Vxdp, Vxap = self.integrator.forward(
        #                               [t0, tend],
        #                               self.y , self.z , self.q.ravel(), self.p,
        #                               self.Vxd, self.Vxa, self.Vq, self.Vp)
        # self.integrator.plugin.dosave = True

        if self.NZ > 0:
            raise NotImplementedError("need integration scheme for DAEs")

        self.x[:], self.x_dot[:,:] = self.ind.fo_forward(
            self.ts[0:2],
            self.x, self.x_dot,
            self.p, self.p_dot,
            self.q[0, :], self.q_dot[0, :])

        # STEP 3: post-process result of integration
        # ##########################################

        # store state x_{L+1}
        xp[:] = self.x

        # store Xx
        Xx[:, :] = self.x_dot[:, :self.NX]

        # store Xp
        Xp[:, :] = self.x_dot[:, self.NX:]


        # print 'self.Vxd, self.Vxa, self.Vq, self.Vp=\n',self.Vxd, self.Vxa, self.Vq, self.Vp
        # print 'self.y , self.z , self.q.ravel(), self.p =\n',self.y , self.z , self.q.ravel(), self.p
        # print 't0, tend=', t0, tend
        # print 'xdp=\n', xdp
        # print 'Vxdp=\n', Vxdp
        # print 'Xx=\n', Xx
        # print 'Xp=\n', Xp

        # store xtilde
        xtilde[:] = xp - Xx.dot(x) - Xp.dot(self.p)

        htilde[:] = 0.
        H = numpy.zeros((self.NH, self.NX + self.NP))
        Hx = H[:, :self.NX]
        Hp = H[:, self.NX:]

        # tmp variables to compute Jacobians


        # cnt = 0
        # for i in range(self.ex.Nmess):
        #     i = int(i)
        #     for j in range(ex.get_mess(int(i)).Nfun):
        #         j = int(j)
        #         nh = ex.get_mess(i).get_nh(j)

        #         ex.get_mess(i).X_MFCN(j,
        #                               self.NX,                                  # number directions
        #                               self.ts[:1],                   # time
        #                               self.x,                                   # state
        #                               eyeNX,                                    # state directions
        #                               eyeNX.shape[1],                           # leading dim of state directions
        #                               htilde[cnt:cnt+nh],                       # measurement response
        #                               Hx[cnt:cnt+nh, :],                        # measurement response directions
        #                               Hx.shape[1],                              # leading dim of state directions
        #                               self.p,
        #                               self.q,
        #                               self.rwh, self.iwh, self.iflag)

        #         ex.get_mess(i).P_MFCN(j,
        #                               self.NP,                                  # number directions
        #                               self.ts[:1],                   # time
        #                               self.x,                                   # state
        #                               htilde[cnt:cnt+nh],                       # measurement response
        #                               Hp[cnt:cnt+nh, :],                        # measurement response directions
        #                               Hp.shape[1],                              # leading dim of state directions
        #                               self.p,
        #                               eyeNP,                                    # state directions
        #                               eyeNP.shape[1],                           # leading dim of state directions
        #                               self.q,
        #                               self.rwh, self.iwh, self.iflag)

        #         cnt += nh

        self.mf.hfcn_d_xpu_v(
            htilde, H,
            self.ts[:1],
            self.x, self.eyeNX,
            self.p, self.eyeNP,
            self.q[0, :], self.zerosNU
            # self.P
        )

        htilde[:] -= Hx.dot(self.x)
        htilde[:] -= Hp.dot(self.p)

        # STEP 4: evaluate new weights and estimates
        # ##########################################

        Nxp = self.NX + self.NP
        NH  = self.NH

        # FIXME: do not reallocate each time
        A = numpy.zeros((2 * Nxp + NH, 2 * Nxp))
        b = numpy.zeros(2 * Nxp + NH)
        tmp = numpy.zeros((Nxp, Nxp))

        # build tmp (used to build A)
        tmp[:self.NX, :self.NX] = Xx
        tmp[:self.NX, self.NX:] = Xp
        tmp[self.NX:, self.NX:] = numpy.eye(self.NP)

        # build A
        A[:Nxp, :Nxp]                =  ACSBI
        A[Nxp:Nxp + NH, :self.NX]    = -Hx/sigma[:, numpy.newaxis]
        A[Nxp:Nxp + NH, self.NX:Nxp] = -Hp/sigma[:, numpy.newaxis]
        A[Nxp + NH:, :Nxp]           = -CovPenalty.dot(tmp)
        A[Nxp + NH:, Nxp:]           =  CovPenalty

        # build b
        b[:Nxp]          = ACSBI[:,:self.NX].dot(xbar)
        b[:Nxp]         += ACSBI[:,self.NX:].dot(pbar)

        b[Nxp:Nxp+NH] = (htilde - eta)/sigma
        b[Nxp+NH:]    = CovPenalty[:,:self.NX].dot(xtilde)

        # print 'CovPenalty=\n', CovPenalty
        # print self.ts
        # print 'A=\n', A
        # print 'b=\n', b

        Q, R = numpy.linalg.qr(A)
        r = - Q.T.dot(b)
        self.ACSBI[...] = R[Nxp:2*Nxp, Nxp:]

        v = - numpy.linalg.solve(ACSBI, r[Nxp:2*Nxp])

        self.xbar[:] = v[:self.NX]
        self.pbar[:] = v[self.NX:]

    def simulate_arrival_cost(self):
        """
        updates arrival cost that it will not influence parameter estimation

        xbar = s[0]
        pbar = p
        ACSBI = eye

        resulting in AC = 0.

        """

        self.debug_out('-'*80+'\n')
        self.debug_out('mhe.simulate_arrival_cost()\n')
        self.debug_out('-'*80+'\n')

        self.xbar = self.s[0, :]
        self.pbar = self.p
        self.ACSBI[...] = numpy.eye(self.ACSBI.shape[0])

    def preparation_phase(self):
        """
        build J1 and J2 matrix

        Reference

        Vorlesung Numerik 2, 2013
        Kapitel 4.2 Das verallgemeinerte Gauss-Newton-Verfahren
        Optimierung mit Differentialgleichungen
        H.G. Bock, M. Diehl, K. Mombaur, S. Sager
        Heidelberg, Leuven, Toulouse


        Uncondensed problem
        -------------------

        ::

            [ J1.T J1    J2.T ]  [ delta_s0     ]       [ J1.T F1 ]
            [                 ]  [ delta_s1     ]       [         ]
            [                 ]  [ ...          ]       [         ]
            [                 ]  [ delta_sNms-1 ]  =  - [         ]
            [                 ]  [ delta_p      ]       [         ]
            [                 ]  [--------------]       [         ]
            [   J2          0 ]  [ lambda       ]       [   F2    ]


        where::

            F1 = [   F0   ]
                 [   F1   ]
                 [   ..   ]
                 [ FNms-1 ]


            F2 = [   x(0) - s0                ]
                 [   x(t1; s0) - s1           ]
                 [       ...                  ]
                 [   x(t1; sNms-2) - sNms-1   ]


            J1 =  [ dF0/ds0                                      dF0/dp       ]
                  [   0         dF1/ds1                          dF1/dp       ]
                  [                        ...                    ...         ]
                  [                             dFNms-1/dsNms-1  dFNms-1/dp   ]
                  [  ACSBI[:,:NX]                              ACSBI[:, -NP:] ]


                  [ dx(1)/ds0    -1                                dx(1)/dp    ]
            J2 =  [           dx(2)/ds1   -1                       dx(2)/dp    ]
                  [                      ...                       ....        ]
                  [                         dx(Nms-1)/dsNms-2  -1  dx(Nms-1)/dp]


        """

        self.debug_out('-'*80+'\n')
        self.debug_out('mhe.preparation_phase()\n')
        self.debug_out('-'*80+'\n')

        # STEP 1: for all multiple shooting intervals
        #         compute J1s, J1p, etc
        for ci in range(self.M-1):
            self._init_ms_node(ci)
            self._calculate_measnode(ci)
            self._preparation_phase_integration(ci)

        # initialize last s value by integration
        self.s[self.M-1,:] = self.x
        self.F2[self.M-2,:][...] = 0.0

        # store x(t_M) to compute F2 in the OED
        self.xM = self.x.copy()

        # last node is a multiple shooting node
        self._init_ms_node(self.M-1)
        self._calculate_measnode(self.M-1)

        # STEP 2: arrival cost
        tmp = numpy.hstack([self.s[0, :] - self.xbar,
                            self.p       - self.pbar])
        self.F1AC[:]     = numpy.dot(self.ACSBI, tmp)
        self.J1AC[:, :]  = self.ACSBI

        # STEP 3: condensing
        self._preparation_phase_condensing()

    def _preparation_phase_condensing(self):
        """
        compare Vorlesung Numerik II, Sager, Eqn 4.45
        D = dsi/ds1
        E = dsi/dp
        """

        Z = self.Z
        e = self.e

        e[0] = 0
        Z[0, :, :] = 0.
        Z[0, :, :self.NX] = numpy.eye(self.NX)
        for nms in range(self.M-1):
            e[nms+1]              = self.J2s[nms].dot(e[nms]) + self.F2[nms]
            Z[nms+1, :, :self.NX] = self.J2s[nms].dot(Z[nms, :, :self.NX])
            Z[nms+1, :, self.NX:] = self.J2s[nms].dot(Z[nms, :, self.NX:]) + self.J2p[nms]

        ## condense J1s and J1p into JC
        for nms in range(self.M):
            a, b = nms*self.NH, (nms+1)*self.NH
            self.FC[a:b] = self.J1s[nms].dot(e[nms]) + self.F1[nms]
            self.JC[a:b, :self.NX] = self.J1s[nms].dot(Z[nms, :, :self.NX])
            self.JC[a:b, self.NX:] = self.J1s[nms].dot(Z[nms, :, self.NX:])\
                                     + self.J1p[nms]

        ### arrival costs at the bottom of JC
        self.FC[self.M*self.NH:]    = self.F1AC
        self.JC[self.M*self.NH:, :] = self.J1AC


    def _init_ms_node(self, ci):
        """
        sets current state self.x to s

        also resets directional derivatives to identity matrices
        """

        ## STEP 0: setup directions for integration and calculate_measnode
        # self.Vxd[:self.NY, 0, :self.NY]      = numpy.eye(self.NY)
        # self.Vxd[self.NX:, 0, :]             = 0.

        # self.Vp[:self.NX, 0, :]              = 0.
        # self.Vp[self.NX:, 0, :]              = numpy.eye(self.NP)

        # self.Vq[...]                         = 0.

        self.x_dot[...] = 0.
        self.x_dot[:, :self.NY] = numpy.eye(self.NY)

        # initialize multiple shooting nodes for integration
        self.x[:] = self.s[ci, :]

        # post-process results of integration
        # store Xx
        # self.Xx[:self.NY, :] = self.Vxd[:self.NX, 0, :].T
        # self.Xx[self.NY:, :] = self.Vxa[:self.NX, 0, :].T
        self.Xx[:, :] = self.x_dot[:, :self.NY]

    def _preparation_phase_integration(self, ci):
        """
        integratation of interval ci
        """

        # setup integration horizon and control function
        # discretization
        # t0, tend = self.ts[ci], self.ts[ci+1]
        # self._update_rwh_iwh(ci)
        # # perform integration
        # self.y [:], self.z [:], self.Vxd[...], self.Vxa[...] = \
        #      self.integrator.forward(
        #                     [t0, tend],
        #                     self.y , self.z , self.q.ravel(), self.p,
        #                     self.Vxd, self.Vxa, self.Vq, self.Vp)

        self.x[:], self.x_dot[:, :] = self.ind.fo_forward(self.ts[ci:ci+2], self.x, self.x_dot, self.p, self.p_dot, self.q[ci, ...], self.q_dot[ci, ...])

        ## build J2
        # self.J2s[ci, :self.NY, :] = self.Vxd[:self.NX, 0, :].T
        # self.J2s[ci, self.NY:, :] = self.Vxa[:self.NX, 0, :].T
        # self.J2p[ci, :self.NY, :] = self.Vxd[self.NX:, 0, :].T
        # self.J2p[ci, self.NY:, :] = self.Vxa[self.NX:, 0, :].T
        # self.F2[ci, :]            = self.x - self.s[ci+1, :]

        self.J2s[ci, :, :]        = self.x_dot[:, :self.NX]
        self.J2p[ci, :, :]        = self.x_dot[:, self.NX:]
        self.F2[ci, :]            = self.x - self.s[ci+1, :]

    def simulate_measurement(self, simulate_error=False):
        """

        This function uses the initial condition self.s_ref[0, :]
        and performs an integration,
        using self.p_ref as parameters and self.q for the control discretization.
        It uses the computed values at self.ts to populate

        * self.etas
        * self.sigmas

        using the sfcn and mfcn functions. If the function is called
        with the argument `simulate_error` it additionally adds some random
        noise to self.etas.

        Note: Call it right before the estimation phase!

        Parameters
        ----------

        :simulate_error:        simulates standard deviation for measurement
                                using sigma function, i.e. h *= s*error

        Returns
        -------

        self.etas[M-1, : ] and self.sigmas[M-1, :]

        """
        self.debug_out('-'*80+'\n')
        self.debug_out('mhe.simulate_measurement()\n')
        self.debug_out('-'*80+'\n')

        if not simulate_error:
            self.debug_out('NOTE: No measurement error is simulated!\n')

        eta   = numpy.zeros(self.NH)
        sigma = numpy.ones(self.NH)
        # ex = self.ex

        # # integrate over whole horizon part by part
        # # but do not save
        # dosave = self.integrator.plugin.dosave
        # self.integrator.plugin.dosave = False

        # print c0, cf
        self.x[:] = self.s_ref[0, :]

        for j in range(self.M):
            self.s_ref[j, :] = self.x[:]

            # for nmess in range(ex.Nmess):
            #     nmess = int(nmess)
            #     for nfun in range(ex.get_mess(nmess).Nfun):
            #         nfun = int(nfun)
            #         nh = ex.get_mess(nmess).get_nh(nfun)

            #         self._update_rwh_iwh(j)

            #         # get measurements and standard deviations

            #         h = numpy.zeros(nh)
            #         s = numpy.zeros(nh)

            #         tmpa = self.m2a[nmess][nfun]
            #         tmpb = self.m2b[nmess][nfun]

            #         # compute model response
            #         ex.get_mess(nmess).MFCN(nfun,
            #                                 self.ts[j:j+1],
            #                                 self.x,
            #                                 h,
            #                                 self.p_ref,
            #                                 self.q.ravel(),
            #                                 self.rwh,
            #                                 self.iwh,
            #                                 numpy.array([0], dtype=numpy.int32) )

            #         # compute model response deviation
            #         ex.get_mess(nmess).SFCN(nfun,
            #                                 self.ts[j:j+1],
            #                                 self.x,
            #                                 s,
            #                                 self.p_ref,
            #                                 self.q.ravel(),
            #                                 self.rwh,
            #                                 self.iwh,
            #                                 numpy.array([0], dtype=numpy.int32) )


            #         # save values
            #         eta[tmpa:tmpb][...]   = h
            #         sigma[tmpa:tmpb][...] = s

            h = numpy.zeros(self.NH)
            s = numpy.zeros(self.NH)

            self.mf.hfcn(h, self.ts[j:j+1], self.x, self.p, self.q.ravel())

            # TODO: implement sfcn
            # self.mf.sfcn(s, t, x, p, u)

            # add simulated measurement error
            if simulate_error:
                error = numpy.random.randn(eta.size)*sigma
                eta += error

            self.etas[j, :] = eta
            self.sigmas[j, :] = sigma

            if j == self.M - 1:
                # self.M - 1 is the last time node
                break

            # t0, tend = self.ts[j], self.ts[j + 1]

            # self._update_rwh_iwh(j)
            # self.y [...], self.z [...] = self.integrator.forward(
            #     [t0, tend],
            #     self.y , self.z , self.q.ravel(), self.p_ref)

            self.x[:] = self.ind.zo_forward(self.ts[j:j+2], self.x, self.p_ref, self.q[j, ...])

        # self.integrator.plugin.dosave = dosave
        return eta, sigma

    def _calculate_measnode(self, ci):
        """
        computes dh/dp and sigma and stores it in
        F1, J1s, J1p
        """

        # ex = self.ex
        htilde = self.htilde
        htilde[:] = 0.

        H = numpy.zeros((self.NH, self.NX + self.NP))
        Hx = H[:, :self.NX]
        Hp = H[:, self.NX:]

        Xx = self.Xx

        # for nmess in range(ex.Nmess):
        #     nmess = int(nmess)
        #     for nfun in range(ex.get_mess(nmess).Nfun):
        #         nfun = int(nfun)
        #         nh = ex.get_mess(nmess).get_nh(nfun)

        #         self._update_rwh_iwh(ci)

        #         # get measurements and standard deviations
        #         eta   = self.etas  [ci, self.m2a[nmess][nfun]:self.m2b[nmess][nfun]]
        #         sigma = self.sigmas[ci, self.m2a[nmess][nfun]:self.m2b[nmess][nfun]]

        #         h = numpy.zeros(nh)
        #         h_p = numpy.zeros((nh, self.NP))
        #         h_x = numpy.zeros((nh, self.NX))

        #         tmpa = self.m2a[nmess][nfun]
        #         tmpb = self.m2b[nmess][nfun]

        #         # STEP 1: compute J1s
        #         ex.get_mess(nmess).X_MFCN(nfun,
        #                               self.NX,
        #                               self.ts[ci:ci+1],
        #                               self.x,
        #                               Xx, self.NX,
        #                               h,
        #                               h_x, self.NX,
        #                               self.p, self.q,
        #                               self.rwh, self.iwh,
        #                               numpy.array([0], dtype=numpy.int32) )

        #         self.J1s[ci, tmpa:tmpb, :]  = h_x
        #         self.J1s[ci, tmpa:tmpb, :] /= sigma[:, numpy.newaxis]
        #         self.F1[ci, tmpa:tmpb]      = (h - eta) / sigma

        #         # STEP 2: compute J1p
        #         h[...] = 0.0
        #         h_p[...] = 0.0

        #         ex.get_mess(nmess).P_MFCN(nfun,
        #                               self.NP,
        #                               self.ts[ci:ci+1],
        #                               self.x,
        #                               h,
        #                               h_p, self.NP,
        #                               self.p,
        #                               numpy.eye(self.NP), self.NP,
        #                               self.q,
        #                               self.rwh, self.iwh,
        #                               numpy.array([0], dtype=numpy.int32) )

        #         self.J1p[ci, tmpa:tmpb, :]  = h_p
        #         self.J1p[ci, tmpa:tmpb, :] /= sigma[:, numpy.newaxis]

        #         # save model response
        #         self.hs[ci, tmpa:tmpb] = h[:]

        self.mf.hfcn_d_xpu_v(
            htilde, H,
            self.ts[ci:ci+1],
            self.x, self.eyeNX,
            self.p, self.eyeNP,
            self.q[ci], self.zerosNU
        )

        # TODO: implement sfcn here and compute hfcn/sfcn


    def set_control(self, q):
        self.debug_out('-'*80+'\n')
        self.debug_out('mhe.set_control(q)\n')
        self.debug_out('-'*80+'\n')

        q = numpy.asarray(q)
        self.q[-1, :] = q[:]

        # when q is changed, recalculate last model response
        self._calculate_measnode(self.M-1)

    def set_measurement(self, eta, sigma):
        self.debug_out('-'*80+'\n')
        self.debug_out('set_measurement\n')
        self.debug_out('-'*80+'\n')

        # check dimensions
        assert len(eta) == self.NH,\
            "wrong dimension of new measurement array eta: {}!={}".format(len(eta), self.NH)
        assert len(sigma) == self.NH, \
            "wrong dimension of new measurement array sigma: {}!={}".format(len(sigma), self.NH)

        eta   = numpy.asarray(eta)
        sigma = numpy.asarray(sigma)

        self.etas[-1,:] = eta[:]
        self.sigmas[-1,:] = sigma[:]

        # NOTE do not update since the values are written by MHEDO.oed
        ci = self.M-1
        self.F1[ci, :] -= eta
        self.F1[ci, :] /= sigma
        self.J1s[ci, :, :] /= sigma[:, numpy.newaxis]
        self.J1p[ci, :, :] /= sigma[:, numpy.newaxis]

        # update condensing
        # divide last rows of J1 by sigma
        a = - (self.NH + self.NP + self.NX)
        b = a + self.NH
        self.FC[a:b] -= eta
        self.FC[a:b] /= sigma
        self.JC[a:b, :] /= sigma[:, numpy.newaxis]

    def optimize(self):
        """
        solve min_deltav 0.5 * ||F1 + J1*deltav||^2
              s.t.               F2 + J2*deltav = 0

        where deltav = (deltas_0, ..., deltas_M, deltap)

        using the nullspace method.

        reference: Numerical optimization
                   Nocedal, Jorge and Wright, Stephen J
                   2006

        on null-space factorization method, p. 428 (447 pdf)
        on application of null-psace method, p. 457 (476 pdf)

        Implementation
        --------------

        find [Y|Z], s.t. J2*Z = 0, J2*Y non-singular partition
        deltav = Y*delta_v_y + Z*delta_v_z

        it then holds that

        STEP1: delta_v_y = -(J2*Y)^{-1} * F2
        STEP2: delta_v_z = (Z^T J_1^T J_1 Z)^{-1} * Z^T*J_1^T * (F_1 - J_1*Y*delta_v_y)

        """

        self.debug_out('-'*80+'\n')
        self.debug_out('estimation_phase\n')
        self.debug_out('-'*80+'\n')

        #print 'svd(self.JC)', numpy.linalg.svd(self.JC)[1]

        Q, R = numpy.linalg.qr(self.JC)
        tmp1 = -Q.T.dot(self.FC)
        delta_v = scipy.linalg.solve_triangular(R, tmp1)
        d = self.e + self.Z.dot(delta_v)

        self.s[:, :] += d.reshape((self.M, self.NX))
        self.p[:]       += delta_v[-self.NP:]

        # store current C for plotting, etc.
        Rinv = numpy.linalg.inv(R)
        C = numpy.dot(Rinv, Rinv.T)

        # store current p for plotting, etc.
        self.plot_data.p[self.M-1, :]     = self.p
        self.plot_data.C[self.M-1, :, :]  = C
        self.plot_data.JC[self.M-1, :, :] = self.JC


    def _update_rwh_iwh(self, j):
        """
        1) prepares rwh and iwh
        2) prepares the plugin for plotting

        :j:     int
                current control interval
                first control interval is 0
        """

        offset = 0
        for nu in range(self.NU):
            # self.rwh[1+2*nu:3+2*nu]     = self.ts[j:j+2] # for mode 4
            self.rwh[1+nu]  = self.ts[j]
            self.iwh[1+nu*3:1+(nu+1)*3] = [1+offset, 3, j+1]
            offset += numpy.prod(self.q.shape[1:])

        self.plot_data.plugin.set_interval(j)

    def simulate_s(self):
        """
        uses self.s[0,:] to compute self.s[1:,:]
        using self.p and self.q
        """
        # # enable save into t_list and x_list
        # tmp = self.integrator.plugin.dosave
        # self.integrator.plugin.dosave = False

        self.x[:] = self.s[0, :]

        # integrate over whole horizon part by part
        for j, node in enumerate(self.ts[:-1]):
            # t0, tend = self.ts[j], self.ts[j + 1]
            # # self._update_rwh_iwh(j)
            # self.y [...], self.z [...] = self.integrator.forward(
            #     [t0, tend],
            #     self.y , self.z , self.q.ravel(), self.p)

            self.x[:] = self.ind.zo_forward(self.ts[j:j+2], self.x, self.p, self.q[j, ...])
            self.s[j+1, :] = self.x
        # self.integrator.plugin.dosave = tmp

    def print_parameter_correlations(self):
        """
        get dependent s and p combinations using the SVD
        """

        self.preparation_phase()
        print 'computing U s V^T = self.JC[:self.M * self.NH, :]'
        U,s,VT = numpy.linalg.svd(self.JC[:self.M * self.NH, :])

        i = numpy.where(s <= 1.e-12)

        V = VT.T
        print 'V=\n', V
        print 'singular values =\n', s
        print 'dependent parameters =\n', V[:,i]


    def __str__(self):
        """
        returns the current problem state in form of a string
        """

        retval  = '\nCurrent estimates\n'
        retval +=   '---------\n\n'

        retval += 'xd = %s\n'%str(self.y )
        if len(self.z ) > 0:
            retval += 'xa = %s\n'%str(self.z )
        retval += 'p  = %s\n'%str(self.p)

        retval += '\nArrival Cost\n'
        retval +=   '------------\n\n'

        retval += 'xbar  = %s\n'%str(self.xbar)
        retval += 'pbar  = %s\n'%str(self.pbar)
        retval += 'ACSBI =\n%s\n'%str(self.ACSBI)

        retval += '\nMeasurements and standard deviations\n'
        retval +=   '------------------------------------\n\n'

        retval += 'on the estimation horizon of length = %d\n\n'%self.M

        for nnode in range(self.M):
            eta, sigma = self.etas[nnode], self.sigmas[nnode]
            retval += 'eta[%4d] = %s   '%(nnode, str(eta))
            retval += 'sigma[%4d] = %s\n'%(nnode, str(sigma))

        return retval


def _show_state_information(func):
    """
    decorator to wrap members and indicate call by output to std, when in
    VERBOSE mode.
    """
    def wrapper(self, *args, **kwargs):
        self.debug_out('-'*80+'\n')
        cls = str(self.__class__).strip('<>').split()[1].strip("'")
        self.debug_out('{cls}.{func}()\n'.format(cls=cls,
                                                 func=func.__name__))
        ret = func(self, *args, **kwargs)
        self.debug_out('-'*80+'\n')
        return ret

    return wrapper

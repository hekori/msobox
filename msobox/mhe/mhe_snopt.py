import sys
import os
import numpy
import scipy
from scipy import linalg as linalg
from matplotlib import pyplot as plt
import snopt

from .mhe import MHE
from .mhe import _show_state_information

class MHE_SNOPT(MHE):

    @_show_state_information
    def __init__(self, path_to_vplan_ini, mako_vars={}, dt=0.05, M=20, major=-1):
        """
        """
        MHE.__init__(self, path_to_vplan_ini, mako_vars=mako_vars, dt=dt, M=M)

        # state constraints
        self.b_lo = -numpy.inf * numpy.ones((self.M, self.NX)) # lower bound
        self.b_up =  numpy.inf * numpy.ones((self.M, self.NX)) # upper bound

        # parameter constraints
        self.p_lo = -numpy.inf * numpy.ones((self.NP)) # lower bound
        self.p_up =  numpy.inf * numpy.ones((self.NP)) # upper bound

        # number of variables =
        #          number of shooting variables #s
        self.NV1 = self.M * self.NX
        #          number of parameters  #p
        self.NV2 = self.NP

        # number of constraints =
        #          number of state continuity conditions
        self.NC1 = (self.M - 1) * self.NX
        #          number of path constraints
        #self.NC2 = self.M * self.NX

        #          number of parameter constraints
        #self.NC3 = self.NP

        self.NC  = self.NC1# + self.NC2 + self.NC3

        # number of equality constraints
        self.NEC = self.NC1

        # number of inequality constraints
        self.NIC = 0 #self.NC2 + self.NC3

        # xi = (s, p,)
        self.NXI = self.NV1 + self.NV2

        # setup SNOPT
        snopt.check_memory_compatibility()
        self.minrw  = numpy.zeros(1, dtype=numpy.int32)
        self.miniw  = numpy.zeros(1, dtype=numpy.int32)
        self.mincw  = numpy.zeros(1, dtype=numpy.int32)

        self.rw     = numpy.zeros(1000000, dtype=numpy.float64)
        self.iw     = numpy.zeros(1000000, dtype=numpy.int32)
        self.cw     = numpy.zeros(8*500, dtype=numpy.character)

        self.Cold   = numpy.array([0], dtype=numpy.int32)
        self.Basis  = numpy.array([1], dtype=numpy.int32)
        self.Warm   = numpy.array([2], dtype=numpy.int32)

        self.xi     = numpy.zeros(self.NXI, dtype=numpy.float64)
        self.xilow  = numpy.zeros(self.NXI, dtype=numpy.float64)  # lower bound on xi
        self.xiupp  = numpy.zeros(self.NXI, dtype=numpy.float64)  # upper bound on xi
        self.ximul  = numpy.zeros(self.NXI, dtype=numpy.float64)
        self.F      = numpy.zeros(1 + self.NC, dtype=numpy.float64)
        self.Flow   = numpy.zeros(1 + self.NC, dtype=numpy.float64)
        self.Fupp   = numpy.zeros(1 + self.NC, dtype=numpy.float64)
        self.Fmul   = numpy.zeros(1 + self.NC, dtype=numpy.float64)

        self.ObjAdd  = numpy.zeros(1, dtype=numpy.float64) # value added to the objective value for printing purposes
        self.xistate = numpy.zeros(self.NXI, dtype=numpy.int32)
        self.Fstate  = numpy.zeros(1 + self.NC, dtype=numpy.int32)

        self.INFO   = numpy.zeros(1, dtype=numpy.int32)
        self.ObjRow = numpy.zeros(1, dtype=numpy.int32) # index of the objective row in F
        self.n      = numpy.zeros(1, dtype=numpy.int32) # = self.NXI
        self.neF    = numpy.zeros(1, dtype=numpy.int32) # = 1 + self.NC

        # linear constraints
        self.neA     = numpy.zeros(1, dtype=numpy.int32)
        self.lenA    = numpy.zeros(1, dtype=numpy.int32)
        self.lenA[0] = (1 + self.NC)*self.NXI # overestimate
        self.iAfun   = numpy.zeros(self.lenA[0], dtype=numpy.int32)
        self.jAvar   = numpy.zeros(self.lenA[0], dtype=numpy.int32)
        self.A       = numpy.zeros(self.lenA[0], dtype=numpy.float64)

        # nonlinear constraints: G is the sparse Jacobian of F
        self.neG         = numpy.zeros(1, dtype=numpy.int32)
        self.lenG        = numpy.zeros(1, dtype=numpy.int32)
        self.lenG[0]     = (1 + self.NC)*self.NXI
        self.iGfun       = numpy.zeros(self.lenG[0], dtype=numpy.int32)
        self.jGvar       = numpy.zeros(self.lenG[0], dtype=numpy.int32)

        self.nxname = numpy.zeros(1, dtype=numpy.int32)
        self.nFname = numpy.zeros(1, dtype=numpy.int32)

        self.nxname[0] = 1
        self.nFname[0] = 1

        self.xinames = numpy.zeros(1*8,   dtype=numpy.character)
        self.Fnames  = numpy.zeros(1*8,   dtype=numpy.character)
        self.Prob    = numpy.zeros(200*8, dtype=numpy.character)

        self.iSpecs   = numpy.zeros(1, dtype=numpy.int32)
        self.iSumm    = numpy.zeros(1, dtype=numpy.int32)
        self.iPrint   = numpy.zeros(1, dtype=numpy.int32)

        self.iSpecs[0] = 4
        self.iSumm [0] = 6
        self.iPrint[0] = 9

        self.printname = numpy.zeros(200*8, dtype=numpy.character)
        self.specname  = numpy.zeros(200*8, dtype=numpy.character)

        self.nS   = numpy.zeros(1, dtype=numpy.int32)
        self.nInf = numpy.zeros(1, dtype=numpy.int32)

        self.sInf = numpy.zeros(1, dtype=numpy.float64)

        self.DerOpt     = numpy.zeros(1, dtype=numpy.int32)
        self.Major      = numpy.zeros(1, dtype=numpy.int32)
        self.iSum       = numpy.zeros(1, dtype=numpy.int32)
        self.iPrt       = numpy.zeros(1, dtype=numpy.int32)
        self.strOpt     = numpy.zeros(200*8, dtype=numpy.character)

        # open output files using snfilewrappers.[ch] */
        specn  = "snmhe.spc"
        printn = "snmhe.out"
        self.specname [:len(specn)]  = list(specn)
        self.printname[:len(printn)] = list(printn)

        # Open the print file, fortran style */
        # file is written to directory
        # print os.path.abspath(os.curdir)

        snopt.snopenappend(self.iPrint, self.printname, self.INFO)
        # ================================================================== */
        # First,  sninit_ MUST be called to initialize optional parameters   */
        # to their default values.                                           */
        # ================================================================== */

        snopt.sninit(self.iPrint, self.iSumm, self.cw, self.iw, self.rw)
        # Set up the problem to be solved.                       */
        # No derivatives are set in this case.                   */
        # NOTE: To mesh with Fortran style coding,               */
        #       it ObjRow must be treated as if array F          */
        #       started at 1, not 0.  Hence, if F(0) = objective */
        #       then ObjRow should be set to 1.                  */

        # Assign the dimensions of the constraint Jacobian

        self.neF[0]    = 1 + self.NC
        self.n[0]      = self.NXI

        self.ObjRow[0] = 1 # NOTE: Me must add one to mesh with fortran
        self.ObjAdd[0] = 0

        # ------------------------------------------------------------------
        # The parameters iPrt and iSum may refer to the Print and Summary
        # file respectively.  Setting them to 0 suppresses printing.
        # ------------------------------------------------------------------
        self.DerOpt[0] = 0 # use finite differences
        self.strOpt_s = "Derivative option"
        self.strOpt[:len(self.strOpt_s)] = list(self.strOpt_s)
        snopt.snseti(self.strOpt, self.DerOpt, self.iPrt, self.iSum, self.INFO, self.cw, self.iw, self.rw)

        self.Major[0] = major # major iterations
        self.strOpt_s = "Major iterations limit"
        self.strOpt[:len(self.strOpt_s)] = list(self.strOpt_s)
        snopt.snseti(self.strOpt, self.Major, self.iPrt, self.iSum, self.INFO, self.cw, self.iw, self.rw)

    @_show_state_information
    def shift(self):
        """extend MHE.shift()"""
        super(MHE_SNOPT, self).shift()

        # extensions to MHE
        # shift bfcn lower and upper bounds
        self.b_lo[:self.M-1, :] = self.b_lo[1:, :]
        self.b_up[:self.M-1, :] = self.b_up[1:, :]

    @_show_state_information
    def get_new_constraints(self, b_lo=None, b_up=None):
        # set new constraints for states
        shp = (self.NX, )
        if b_lo == None:
            self.b_lo[-1, :] = self.b_lo[-2, :]
        else:
            assert b_lo.shape == shp,\
                "wrong dimension of b_lo: {}!={}".format(b_lo.shape, shp)
            self.b_lo[-1, :] = b_lo

        if b_up == None:
            self.b_up[-1, :] = self.b_up[-2, :]
        else:
            assert b_up.shape == shp,\
                "wrong dimension of b_up: {}!={}".format(b_up.shape, shp)
            self.b_up[-1, :] = b_up

    def sp2xi(self):
        """
        builds the optimization vector xi from s and p
        """
        self.xi[:self.NV1] = self.s.ravel()
        self.xi[self.NV1:] = self.p.ravel()

    def xi2sp(self):
        """
        builds s and p from xi
        """
        self.s[:, :] = self.xi[:self.NV1].reshape(self.s.shape)
        self.p[:]    = self.xi[self.NV1:].reshape(self.p.shape)


    @_show_state_information
    def update_constraints(self):
        """

        F =       [ Phi ]  # objective row
                  [ F2  ]  # matching conditions

        """

        self.xistate[:] =   0
        self.INFO[0]    =   0

        # bounds on s-variables
        self.xilow[:self.NV1] = self.b_lo[:, :].ravel()
        self.xiupp[:self.NV1] = self.b_up[:, :].ravel()

        # bounds on p-variables
        self.xilow[self.NV1:] = self.p_lo[:].ravel()
        self.xiupp[self.NV1:] = self.p_up[:].ravel()

        # bounds on objective
        self.Flow[0] = -1e6
        self.Fupp[0] =  1e6
        self.Fmul[0] =    0

        # bounds for matching conditions
        self.Flow[1:1+self.NC1] = 0
        self.Fupp[1:1+self.NC1] = 0
        self.Fmul[1:1+self.NC1] = 0

        # bounds for dynamic constraints bfcn
        #self.Flow[1+self.NC1:1+self.NC1+self.NC2] = self.b_lo[:, :].ravel()
        #self.Fupp[1+self.NC1:1+self.NC1+self.NC2] = self.b_up[:, :].ravel()
        #self.Fmul[1+self.NC1:1+self.NC1+self.NC2] = 0

        # bounds for parameter constraints
        #self.Flow[1+self.NC1+self.NC2:1+self.NC1+self.NC2+self.NC3] = self.p_lo[:].ravel()
        #self.Fupp[1+self.NC1+self.NC2:1+self.NC1+self.NC2+self.NC3] = self.p_up[:].ravel()
        #self.Fmul[1+self.NC1+self.NC2:1+self.NC1+self.NC2+self.NC3] = 0

    def update_jacobian_sparsity(self):
        """
        G (i.e. the Jacobian of F) is a dense matrix
        """

        # FIXME this should not be so hard, maybe this improves performance

        self.neG[0] = 0
        self.neA[0] = 0

        for i in range(1+self.NC):
            for j in range(self.NXI):
                self.iGfun[self.neG[0]] = 1+i
                self.jGvar[self.neG[0]] = 1+j
                self.neG[0] += 1

    def usrfg(self, status, xi, needF, neF, F, needG, neG, G, cu, iu, ru):
        """
        ==================================================================
        Computes the nonlinear objective and constraint terms.

        The triples (G(k),iGfun(k),jGvar(k)), k = 1,2,...,neG, define
        the sparsity pattern and values of the nonlinear elements
        of the Jacobian.
        ==================================================================
        """
        self.xi[:] = xi[:]  # update member xi
        self.xi2sp()        # update variables

        if needF[0] > 0 or needG[0] > 0:
            self.preparation_phase()

        # evaluation of problem
        if( needF[0] > 0 ):
            F[:]                              = 0
            #  phi = F1^T * F1
            F[0]                              = self.F1.ravel().dot(self.F1.ravel())
            F[1:1+self.NC1]                   = self.F2[:,:].ravel()
            #F[1+self.NC1:1+self.NC1+self.NC2] = self.s.ravel()
            #F[1+self.NC1+self.NC2:1+self.NC1+self.NC2+self.NC] = self.p.ravel()

        # evaluation of gradient
        if( needG[0] > 0 ):
            # prepare gradient
            neG[0] = 0
            G[:] = 0.

            # row 1: objective gradient
            # =========================
            #       xi = [s[0], ..., s[M-1], p]
            #      phi = F1^T * F1
            # grad phi = (F1^T * J1s, F1^T * J1p)
            tmp = G[neG[0]:neG[0]+self.NXI]

            for i in range(self.M):
                tmp[(i)*self.NX:(i+1)*self.NX] = \
                        self.F1[i,:].ravel().dot(self.J1s[i,:,:])
                tmp[self.NV1:] += \
                        self.F1[i,:].ravel().dot(self.J1p[i,:,:])

            neG[0] += self.NXI
            assert neG[0] == self.NXI

            # row 2 etc: constraint Jacobian
            # =================================
            # matching conditions

            tmp = G[neG[0]:neG[0]+self.NC1*self.NXI].reshape((self.NC1, self.NXI))

            #  xi = [ s[0]   s[1]    s[2]   ...     s[M-1] s[M] | p        ]
            # tmp = [ J2s[0] -I                                 | J2p[0]   ]
            #       [        J2s[1]  -I                         | J2p[1]   ]
            #       [                J2s[2] -I                  | ...      ]
            #       [                           ... ...         | ...      ]
            #       [                               J2s[M-1] -I | J2p[M-1] ]

            # unity matrix part
            I = numpy.eye(self.NX)

            # other rows
            for k in range(0, self.M-1):
                a = k * self.NX
                b = a + self.NX
                c = a
                d = c + self.NX
                tmp[a:b, c:d] = self.J2s[k, :, :]

                c = c + self.NX
                d = c + self.NX
                tmp[a:b, c:d] = -I

                c = self.NV1
                d = c + self.NP
                tmp[a:b, c:d] = self.J2p[k, :, :]

            neG[0] += self.NC1*self.NXI
            assert neG[0] == self.NXI + self.NC1*self.NXI

            # dynamic constraints bfcn
            # FIXME: we assume here bfcn = s
            #tmp = G[neG[0]:neG[0]+self.NC2*self.NXI].reshape((self.NC2, self.NXI))
            #tmp[:, :self.NC2] = numpy.eye(self.NC2)
            #neG[0] += self.NC2*self.NXI
            #assert neG[0] == self.NXI + self.NC1*self.NXI + self.NC2*self.NXI

            ## enforce continuity of controls
            #tmp = G[neG[0]:neG[0]+self.NC3*self.NXI].reshape((self.NC3, self.NXI))
            #tmp[:, -self.NC3:] = numpy.eye(self.NC3)
            #neG[0] += self.NC3*self.NXI
            #assert neG[0] == self.NXI + self.NC1*self.NXI + self.NC2*self.NXI + self.NC3*self.NXI

            # check if dimensions are correct
            assert neG[0] == self.lenG[0]

    @_show_state_information
    def optimize(self):

        def usrfg(status, xi, needF, neF, F, needG, neG, G, cu, iu, ru):
            """ simple wrapper for usrfg """
            self.usrfg(status, xi, needF, neF, F, needG, neG, G, cu, iu, ru)

        self.update_constraints()
        self.update_jacobian_sparsity()
        self.sp2xi()

        #snopt.snjac(self.INFO, self.neF, self.n, usrfg,
        #         self.iAfun, self.jAvar, self.lenA, self.neA, self.A,
        #         self.iGfun, self.jGvar, self.lenG, self.neG,
        #         self.xi, self.xilow, self.xiupp, self.mincw, self.miniw, self.minrw,
        #         self.cw, self.iw, self.rw, self.cw, self.iw, self.rw)

        # print 'self.neG=',self.neG

        snopt.snopta(self.Cold, self.neF, self.n, self.nxname, self.nFname,
               self.ObjAdd, self.ObjRow, self.Prob, usrfg,
               self.iAfun, self.jAvar, self.lenA, self.neA, self.A,
               self.iGfun, self.jGvar, self.lenG, self.neG,
               self.xilow, self.xiupp, self.xinames, self.Flow, self.Fupp, self.Fnames,
               self.xi, self.xistate, self.ximul, self.F, self.Fstate, self.Fmul,
               self.INFO, self.mincw, self.miniw, self.minrw,
               self.nS, self.nInf, self.sInf, self.cw,  self.iw,  self.rw, self.cw,  self.iw,  self.rw )

        self.xi2sp()

        self.plot_data.C[self.M-1, :, :]  = numpy.linalg.inv(self.JC.T.dot(self.JC))
        self.plot_data.JC[self.M-1, :, :] = self.JC
        self.plot_data.p[self.M-1, :]     = self.p


    @_show_state_information
    def __del__(self):
        snopt.snclose(self.iPrint)
        snopt.snclose(self.iSpecs)

    # make them static methods when everything is decorated to be able to call them from derived class
    _show_state_information = staticmethod(_show_state_information)

# -*- coding: utf-8 -*-
"""
Wrapper class for SNOPT sparse NLP solver.
"""
try:
    import snopt
except ImportError:
    err_s = "SNOPT is a third-party software a needs proper licensing!"
    raise ImportError(err_s)

import os
import numpy as np
import scipy.linalg as lg
import scipy.sparse as sp


# ------------------------------------------------------------------------------
class SNOPT(object):

    """Wrapper class for SNOPT sparse NLP solver."""

    INF = 1.1e+20

    def __init__(self, NV, NC):
        """Set up SNOPT solver."""
        # assign variables
        self.NV = NV
        self.NC = NC

        # info flag
        self.INFO = np.zeros((1,), dtype=np.int32)
        self.STATE = "setup"

        # problem dimensions
        # NOTE: Me must add one to mesh with fortran
        self.ObjRow = np.array([1], dtype=np.int32)  # index of the obj row
        self.n = np.zeros((1,), dtype=np.int32)
        self.nF = np.zeros((1,), dtype=np.int32)

        self.nF[0] = 1 + NC
        self.n[0] = NV

        # checks if data types have proper sizes
        snopt.check_memory_compatibility()

        # work arrays
        self.minrw = np.zeros((1), dtype=np.int32)
        self.miniw = np.zeros((1), dtype=np.int32)
        self.mincw = np.zeros((1), dtype=np.int32)

        self.rw = np.zeros((10000,), dtype=np.float64)
        self.iw = np.zeros((10000,), dtype=np.int32)
        self.cw = np.zeros((8*500,), dtype=np.character)

        # Start is an integer that specifies how a starting point is to be
        # obtained:
        #   Start = 0 (Cold start) requests that the CRASH procedure be used,
        #             unless an Old basis, Insert, or Load file is specified.
        #   Start = 1 is the same as 0 but more meaningful when a basis file is
        #             given.
        #   Start = 2 (Warm start) means that xstate and Fstate define a valid
        #             starting point (perhaps from an earlier call, though not necessarily).
        self.cold = np.array([0], dtype=np.int32)
        self.basis = np.array([1], dtype=np.int32)
        self.warm = np.array([2], dtype=np.int32)

        # derivative mode
        self.deropt = np.zeros((1,), dtype=np.int32)

        # current state variable
        self.x = np.zeros((self.n,), dtype=np.float64)  # current system state

        self.xlow = np.zeros((self.n,), dtype=np.float64)  # lower bound
        self.xlow[:] = -self.INF

        self.xupp = np.zeros((self.n,), dtype=np.float64)  # upper bound
        self.xupp[:] = self.INF

        self.xmul = np.zeros((self.n,), dtype=np.float64)  # dual variables

        # add objective as well
        self.F = np.zeros((self.nF,), dtype=np.float64)     # actual value

        self.Flow = np.zeros((self.nF,), dtype=np.float64)  # lower bounds
        self.Flow[:] = -self.INF

        self.Fupp = np.zeros((self.nF,), dtype=np.float64)  # upper bounds
        self.Fupp[:] = self.INF

        self.Fmul = np.zeros((self.nF,), dtype=np.float64)  # dual variables

        # ObjAdd is a constant that will be added to the objective row
        # F(Objrow) for printing purposes. Typically, ObjAdd = 0.0d+0.
        self.ObjAdd = np.zeros((1,), dtype=np.float64)

        # initial values for the solution process
        # xstate usually contains a set of initial states for each variable x.
        self.xstate = np.zeros((self.n,), dtype=np.int32)
        # Fstate sometimes contains a set of initial states for the problem
        # functions F
        self.Fstate = np.zeros((self.nF,), dtype=np.int32)

        # linear part in objective
        self.neA = np.zeros((1,), dtype=np.int32)
        self.lenA = np.zeros((1,), dtype=np.int32)
        self.neA[0] = 0  # valid entries
        self.lenA[0] = self.n * self.n  # memory for sparse matrix
        self.iAfun = np.zeros((self.lenA[0],), dtype=np.int32)  # row coordinate
        self.jAvar = np.zeros((self.lenA[0],), dtype=np.int32)  # col coordinate
        self.A = np.zeros((self.lenA[0],), dtype=np.float64)  # (row, col) value

        # sparse Jacobian of problem
        self.neG = np.zeros((1,), dtype=np.int32)
        self.lenG   = np.zeros((1,), dtype=np.int32)
        self.neG[0] = 0
        self.lenG[0] = self.n * self.nF
        self.iGfun = np.zeros((self.lenG[0],), dtype=np.int32)  # row coordinate
        self.jGvar = np.zeros((self.lenG[0],), dtype=np.int32)  # col coordinate
        # NOTE: G would be (row, col) = value but is handled by SNOPT7
        self.x_d = np.eye(self.n)  # directions for gradient evaluation

        self.nxname = np.zeros((1,), dtype=np.int32)
        self.nFname = np.zeros((1,), dtype=np.int32)

        self.nxname[0] = 1  # no names are provided
        self.nFname[0] = 1  # no names are provided

        self.xnames = np.zeros((1*8,), dtype=np.character)
        self.Fnames = np.zeros((1*8,), dtype=np.character)
        self.Prob = np.zeros((200*8,), dtype=np.character)

        self.iSpecs   = np.array([4], dtype=np.int32)
        self.iSumm    = np.array([6], dtype=np.int32)
        self.iPrint   = np.array([9], dtype=np.int32)

        self.iSpecs[0] = 0
        # self.iSumm[0] = 0
        # self.iPrint[0] = 0

        self.printname = np.zeros((200*8,), dtype=np.character)
        self.specname  = np.zeros((200*8,), dtype=np.character)

        self.nS   = np.zeros((1,), dtype=np.int32)
        self.nInf = np.zeros((1,), dtype=np.int32)
        self.sInf = np.zeros((1,), dtype=np.float64)

        # open output files using snfilewrappers.[ch]
        specn = ""  # os.path.join("/", "tmp", "sntoya.spc")
        printn = os.path.join("/", "tmp", "snopt7.out")
        self.specname [:len(specn)]  = list(specn)
        self.printname[:len(printn)] = list(printn)

        # Open the print file, fortran style
        if self.iPrint >= 0 and len(self.printname) > 0:
            snopt.snopenappend(self.iPrint, self.printname, self.INFO)

        # initialize SNOPT
        # initialize SNOPT
        # First,  sninit_ MUST be called to initialize optional parameters
        # to their default values.
        snopt.sninit(self.iPrint, self.iSumm, self.cw, self.iw, self.rw)

        # set up problem to be solved
        # self.setup(
        #     INFO, Prob, nF, n, iAfun, jAvar, lenA, neA, A,
        #     iGfun, jGvar, lenG, neG, ObjAdd, ObjRow, xlow, xupp, Flow,
        #     Fupp, x, xstate, Fmul
        # )

        # open spec file
        if self.iSpecs[0] > 0 and self.specname.nonzero():
            snopt.snfilewrapper(
                self.specname, self.iSpecs, self.INFO,
                self.cw, self.iw, self.rw
            )

            if self.INFO[0] != 101:
                err_s = "Warning: Trouble reading specs file {spc}"
                err_s = err_s.format(spc="".join(self.specname))
                print err_s

    def set_derivative_mode(self, der_mode = 0):
        # set options not specified in the spec file
        self.deropt[0] = der_mode

        strOpt_s = "Derivative option"
        strOpt = np.zeros((200*8,), dtype=np.character)
        strOpt[:len(strOpt_s)] = list(strOpt_s)
        snopt.snseti(
            strOpt, self.deropt,
            self.iPrint, self.iSumm, self.INFO,
            self.cw, self.iw, self.rw
        )

    def calc_jacobian(self, call_back):
        snopt.snjac(
            self.INFO, self.nF, self.n,
            call_back,
            self.iAfun, self.jAvar, self.lenA, self.neA, self.A,
            self.iGfun, self.jGvar, self.lenG, self.neG,
            self.x, self.xlow, self.xupp,
            self.mincw, self.miniw, self.minrw,
            self.cw, self.iw, self.rw,
            self.cw, self.iw, self.rw
        )

    def sqp_step(self, call_back):
        """Solve NLP for given initial value."""
        # call SNOPT
        snopt.snopta(
            self.cold, self.nF, self.n,
            self.nxname, self.nFname,
            self.ObjAdd, self.ObjRow, self.Prob,
            call_back,
            self.iAfun, self.jAvar, self.lenA, self.neA, self.A,
            self.iGfun, self.jGvar, self.lenG, self.neG,
            self.xlow, self.xupp, self.xnames,
            self.Flow, self.Fupp, self.Fnames,
            self.x, self.xstate, self.xmul,
            self.F, self.Fstate, self.Fmul,
            self.INFO,
            self.mincw, self.miniw, self.minrw,
            self.nS, self.nInf, self.sInf,
            self.cw, self.iw, self.rw,
            self.cw, self.iw, self.rw
        )

    def __del__(self):
        """Close file handles."""
        snopt.snclose(self.iPrint)
        snopt.snclose(self.iSpecs)


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    """
    Toy0   defines input data for the toy problem discussed in the SnoptA
    Users Guide::

           Minimize                        x(2)
           subject to     x(1)**2      + 4 x(2)**2  <= 4,
                         (x(1) - 2)**2 +   x(2)**2  <= 5,
                     0<=  x(1)
    """
    NV = 2
    NC = 2
    sn = SNOPT(NV=NV, NC=NC)

    # initial value for problem
    sn.xstate[0] =   0
    sn.xstate[1] =   0

    sn.x[0]    =  1.0
    sn.x[1]    =  1.0

    sn.xlow[0] = 0.0  # 0 <= x[0]

    sn.Fupp[1:] = [4.0, 5.0]

    # sn.neA[0]    = 1
    # sn.neG[0]    = 1
    # define gradient information
    # TODO create from scipy.sparse matrix
    sn.neG[0]    = 0
    sn.iGfun[sn.neG[0]] = 1
    sn.jGvar[sn.neG[0]] = 2

    sn.neG[0]       += 1
    sn.iGfun[sn.neG[0]] = 2
    sn.jGvar[sn.neG[0]] = 1

    sn.neG[0]       += 1
    sn.iGfun[sn.neG[0]] = 2
    sn.jGvar[sn.neG[0]] = 2

    sn.neG[0]       += 1
    sn.iGfun[sn.neG[0]] = 3
    sn.jGvar[sn.neG[0]] = 1

    sn.neG[0]       += 1
    sn.iGfun[sn.neG[0]] = 3
    sn.jGvar[sn.neG[0]] = 2

    sn.neG[0] += 1
    # neG[0] = 6

    sn.neA[0] = 0

    def evaluate(status, x, needF, nF, F, needG, neG, G, cu, iu, ru):
        """Function to implement that is used by SNOPT to solve the problem."""
        # set status flag
        status[0] = 0

        if needF[0] != 0:
            F[0] = x[1]  # the objective row!
            F[1] = x[0]**2 + 4.0*x[1]**2
            F[2] = (x[0] - 2.0)**2 + x[1]**2

        if needG[0] != 0:
            G[0] = 1.0  # G(1,2) !
            G[1] = 2.0 * x[0]  # G(2,1) !
            G[2] = 8.0 * x[1]  # G(2,2) !
            G[3] = 2.0 * (x[0] - 2.0)  # G(3,1) !
            G[4] = 2.0 * x[1]  # G(3,2) !

        return 0

    # sn.calc_jacobian(evaluate)
    sn.sqp_step(evaluate)

    print sn.F
    print sn.Fstate

    print sn.x
    print sn.xstate

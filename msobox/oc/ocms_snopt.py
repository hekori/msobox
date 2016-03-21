# -*- coding: utf-8 -*-

"""
===============================================================================

multiple shooting solution of discretized optimal control problems with SNOPT ...

===============================================================================
"""

# system imports
import numpy as np
import datetime as datetime

# local imports
import snopt as snopt

"""
===============================================================================
"""

class OCMS_snopt(object):

    """

    provides functionalities for ...

    """

    """
    ===============================================================================
    """

    def __init__(self, ocp):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        # set attributes
        self.ocp = ocp

    """
    ===============================================================================
    """

    def solve(self, x0, xend, p, q0, s0):

        """

        solve an ocp ...

        input:

        output:

        TODO:

        """

        NQ = self.ocp.NQ + self.ocp.NS * self.ocp.NX           # add the shooting variables as controls
        NC = self.ocp.NC + (self.ocp.NS - 1) * self.ocp.NX     # add matching conditions for shooting nodes

        """
        ===============================================================================
        """

        def setup(inform, Prob, neF, n, iAfun, jAvar, lenA, neA, A,
                  iGfun, jGvar, lenG, neG, ObjAdd, ObjRow, xlow, xupp, Flow,
                  Fupp, x, xstate, Fmul):

            # give the problem a name.
            Prob[:3] = list('ocp')

            # assign the dimensions of the constraint Jacobian
            neF[0]  = 1 + NC
            n[0]    = NQ

            # set the objective row
            ObjRow[0]   = 1
            ObjAdd[0]   = 0
            Flow[0]     = -1e6
            Fupp[0]     = 1e6

            # set the nonlinear constraints of the problem
            for i in xrange(1, self.ocp.NC + 1):
                Flow[i] = -1e6
                Fupp[i] = 0

            # set the equality constraints for the matching conditions
            for i in xrange(self.ocp.NC + 1, NC + 1):
                Flow[i] = 0
                Fupp[i] = 0

            # set the upper and lower bounds for the controls q
            for j in xrange(0, self.ocp.NU):
                xlow[j * self.ocp.NTS:(j + 1) * self.ocp.NTS] = self.ocp.bc[j, 0]
                xupp[j * self.ocp.NTS:(j + 1) * self.ocp.NTS] = self.ocp.bc[j, 1]

            # set the upper and lower bounds for the shooting variables s
            xlow[self.ocp.NQ:] = -1e6
            xupp[self.ocp.NQ:] = 1e6

            # fix the shooting variables s at the boundaries if necessary
            for i in xrange(0, self.ocp.NX):

                if x0[i] is not None:
                    xlow[self.ocp.NQ + i] = x0[i]
                    xupp[self.ocp.NQ + i] = x0[i]

                if xend[-(i + 1)] is not None:
                    xlow[-(i + 1)] = xend[-(i + 1)]
                    xupp[-(i + 1)] = xend[-(i + 1)]

            # set xstate
            xstate[0:NQ] = 0

            # set up pattern for the jacobian
            neG[0] = NQ * (1 + NC)
            l = 0

            for i in xrange(0, NC + 1):
                for j in xrange(0, NQ):
                    iGfun[l + j] = i + 1
                    jGvar[l + j] = j + 1

                l = l + NQ

        """
        ===============================================================================
        """

        def evaluate(status, x, needF, neF, F, needG, neG, G, cu, iu, ru):

            # separate controls and shooting variables for readability
            q = x[:self.ocp.NQ]
            s = x[self.ocp.NQ:]

            if needF[0] != 0:

                # integrate for current setting
                xs = self.ocp.integrate(p, q, s)

                # calculate objective for current controls
                F[0]                    = self.ocp.obj(xs, None, None, p, q, s)

                # evaluate the nonlinear constraints
                F[1:self.ocp.NC + 1]    = self.ocp.c(xs, None, None, p, q, s)

                # add the matching conditions for multiple shooting
                for i in xrange(0, self.ocp.NS - 1):
                    begin           = (self.ocp.NC + 1) + (i * self.ocp.NX)
                    end             = (self.ocp.NC + 1) + ((i + 1) * self.ocp.NX)
                    F[begin:end]    = self.ocp.integrate_interval(i, p, q, s)[-1, :] - self.ocp.convert_s(s)[i + 1, :]

            if needG[0] != 0:

                # integrate and build derivatives for q and x0
                xs, xs_dot_q    = self.ocp.integrate_dq(p, q, s)
                xs_dot_s        = self.ocp.integrate_ds(p, q, s)[1]

                # calculate gradient of objective
                G[0:self.ocp.NQ]                                        = self.ocp.obj_dq(xs, xs_dot_q, None, p, q, s)      # controls
                G[self.ocp.NQ:self.ocp.NQ + self.ocp.NX * self.ocp.NS]  = self.ocp.obj_ds(xs, xs_dot_s, None, p, q, s)      # shooting variables
                l                                                       = self.ocp.NQ + self.ocp.NS * self.ocp.NX           # save position in array G

                # calculate derivatives for constraints
                for i in xrange(0, self.ocp.NC):
                    G[l:l + self.ocp.NQ]                                                = self.ocp.c_dq(xs, xs_dot_q, None, p, q, s)[i, :]      # controls
                    G[l + self.ocp.NQ:l + self.ocp.NQ + self.ocp.NX * self.ocp.NS]      = self.ocp.c_ds(xs, xs_dot_s, None, p, q, s)[i, :]      # shooting variables
                    l                                                                   = l + self.ocp.NQ + self.ocp.NX * self.ocp.NS           # update l

                # calculate derivatives for matching conditions at boundary
                for i in xrange(0, self.ocp.NS - 1):

                    xs_dot_interval_dq  = self.ocp.integrate_interval_dq(i, p, q, s)[1]
                    xs_dot_interval_dx0 = self.ocp.integrate_interval_dx0(i, p, q, s)[1]

                    for j in xrange(0, self.ocp.NX):
                        G[l:l + self.ocp.NQ]                                                            = 0                                             # controls
                        G[l + i]                                                                        = xs_dot_interval_dq[-1, j]                     # controls
                        G[l + self.ocp.NQ:l + self.ocp.NQ + self.ocp.NS * self.ocp.NX]                  = 0                                             # shooting variables
                        G[l + self.ocp.NQ + i * self.ocp.NX:l + self.ocp.NQ + (i + 1) * self.ocp.NX]    = xs_dot_interval_dx0[-1, j, :]                 # shooting variables
                        G[l + self.ocp.NQ + (i + 1) * self.ocp.NX + j]                                  = -1                                            # shooting variables
                        l                                                                               = l + self.ocp.NQ + self.ocp.NS * self.ocp.NX   # update l

            return 0

        """
        ===============================================================================
        """

        snopt.check_memory_compatibility()
        minrw = np.zeros((1), dtype=np.int32)
        miniw = np.zeros((1), dtype=np.int32)
        mincw = np.zeros((1), dtype=np.int32)

        rw = np.zeros((1000000,), dtype=np.float64)
        iw = np.zeros((1000000,), dtype=np.int32)
        cw = np.zeros((10000,), dtype=np.character)

        Cold  = np.array([0], dtype=np.int32)
        Basis = np.array([1], dtype=np.int32)
        Warm  = np.array([2], dtype=np.int32)

        x    = np.append(q0, s0)
        x    = np.array(x, dtype=np.float64)
        xlow = np.zeros((NQ,), dtype=np.float64)
        xupp = np.zeros((NQ,), dtype=np.float64)
        xmul = np.zeros((NQ,), dtype=np.float64)
        F    = np.zeros((1 + NC,), dtype=np.float64)
        Flow = np.zeros((1 + NC,), dtype=np.float64)
        Fupp = np.zeros((1 + NC,), dtype=np.float64)
        Fmul = np.zeros((1 + NC,), dtype=np.float64)

        ObjAdd = np.zeros((1,), dtype=np.float64)

        xstate = np.zeros((NQ,), dtype=np.int32)
        Fstate = np.zeros((1 + NC,), dtype=np.int32)

        INFO   = np.zeros((1,), dtype=np.int32)
        ObjRow = np.zeros((1,), dtype=np.int32)
        n      = np.zeros((1,), dtype=np.int32)
        neF    = np.zeros((1,), dtype=np.int32)

        lenA   = np.zeros((1,), dtype=np.int32)
        lenA[0] = NQ * (1 + NC)

        iAfun = np.zeros((lenA[0],), dtype=np.int32)
        jAvar = np.zeros((lenA[0],), dtype=np.int32)

        A     = np.zeros((lenA[0],), dtype=np.float64)

        lenG   = np.zeros((1,), dtype=np.int32)
        lenG[0] = NQ * (1 + NC)

        iGfun = np.zeros((lenG[0],), dtype=np.int32)
        jGvar = np.zeros((lenG[0],), dtype=np.int32)

        neA = np.zeros((1,), dtype=np.int32)
        neG = np.zeros((1,), dtype=np.int32)

        nxname = np.zeros((1,), dtype=np.int32)
        nFname = np.zeros((1,), dtype=np.int32)

        nxname[0] = 1
        nFname[0] = 1

        xnames = np.zeros((1 * 8,), dtype=np.character)
        Fnames = np.zeros((1 * 8,), dtype=np.character)
        Prob   = np.zeros((200 * 8,), dtype=np.character)

        iSpecs  = np.zeros((1,), dtype=np.int32)
        iSumm   = np.zeros((1,), dtype=np.int32)
        iPrint  = np.zeros((1,), dtype=np.int32)

        iSpecs[0] = 4
        iSumm [0] = 6
        iPrint[0] = 9

        printname = np.zeros((200 * 8,), dtype=np.character)
        specname  = np.zeros((200 * 8,), dtype=np.character)

        nS   = np.zeros((1,), dtype=np.int32)
        nInf = np.zeros((1,), dtype=np.int32)
        sInf = np.zeros((1,), dtype=np.float64)

        # open output files using snfilewrappers.[ch] */
        specn  = self.ocp.path + "/snopt.spc"
        printn = self.ocp.path + "/output/ocms-" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-snopt.out"
        specname[:len(specn)]   = list(specn)
        printname[:len(printn)] = list(printn)

        # Open the print file, fortran style */
        snopt.snopenappend(iPrint, printname, INFO)

        # initialize snopt to its default parameter
        snopt.sninit(iPrint, iSumm, cw, iw, rw)

        # set up problem to be solved
        setup(INFO, Prob, neF, n, iAfun, jAvar, lenA, neA, A,
              iGfun, jGvar, lenG, neG, ObjAdd, ObjRow, xlow, xupp, Flow,
              Fupp, x, xstate, Fmul)

        # open spec file
        snopt.snfilewrapper(specname, iSpecs, INFO, cw, iw, rw)

        if INFO[0] != 101:
            print("Warning: Trouble reading specs file %s \n" % (specname))

        # set options not specified in the spec file
    #        iPrt   = np.array([0], dtype=np.int32)
    #        iSum   = np.array([0], dtype=np.int32)
    #        strOpt = np.zeros((200*8,), dtype=np.character)

    #        DerOpt = np.zeros((1,), dtype=np.int32)
    #        DerOpt[0] = 1
    #        strOpt_s = "Derivative option"
    #        strOpt[:len(strOpt_s)] = list(strOpt_s)
    #        snopt.snseti(strOpt, DerOpt, iPrt, iSum, INFO, cw, iw, rw)

        # call snopt
        snopt.snopta(Cold, neF, n, nxname, nFname,
                     ObjAdd, ObjRow, Prob, evaluate,
                     iAfun, jAvar, lenA, neA, A,
                     iGfun, jGvar, lenG, neG,
                     xlow, xupp, xnames, Flow, Fupp, Fnames,
                     x, xstate, xmul, F, Fstate, Fmul,
                     INFO, mincw, miniw, minrw,
                     nS, nInf, sInf, cw, iw, rw, cw, iw, rw)

        snopt.snclose(iPrint)
        snopt.snclose(iSpecs)

        return x[:self.ocp.NQ], x[self.ocp.NQ:], F[0], F[1:], -Fmul[1:]

"""
===============================================================================
"""
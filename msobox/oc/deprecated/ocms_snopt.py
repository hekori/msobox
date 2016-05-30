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

    def solve(self):

        """

        solve an ocp ...

        input:

        output:

        TODO:

        """

        x0 = self.ocp.x0
        xend = self.ocp.xend
        p = self.ocp.p
        q0 = self.ocp.q0
        s0 = self.ocp.s0

        NQ = self.ocp.NQ + self.ocp.NS   # add the shooting variables as controls
        NC = self.ocp.NC + self.ocp.NMC  # add matching conditions for shooting nodes

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
            ObjRow[0] = 1
            ObjAdd[0] = 0
            Flow[0]   = -1e6
            Fupp[0]   = 1e6

            # set the upper and lower bnds for the inequality constraints
            Flow[1:1 + self.ocp.NCG] = -1e6
            Fupp[1:1 + self.ocp.NCG] = 0

            # set the upper and lower bnds for the equality constraints
            Flow[1 + self.ocp.NCG:1 + self.ocp.NC] = 0
            Fupp[1 + self.ocp.NCG:1 + self.ocp.NC] = 0

            # set the upper and lower bnds for the matching conditions
            Flow[1 + self.ocp.NC:] = 0
            Fupp[1 + self.ocp.NC:] = 0

            # set the upper and lower bnds for the controls q
            for i in xrange(0, self.ocp.NU):
                xlow[i * self.ocp.NTS:(i + 1) * self.ocp.NTS] = self.ocp.bnds[i, 0]
                xupp[i * self.ocp.NTS:(i + 1) * self.ocp.NTS] = self.ocp.bnds[i, 1]

            # set the upper and lower bnds for the shooting variables s
            xlow[self.ocp.NQ:] = -1e6
            xupp[self.ocp.NQ:] = 1e6

            # fix the shooting variables s at the boundaries if necessary
            for i in xrange(0, self.ocp.NX):

                if x0[i] is not None:
                    xlow[self.ocp.NQ + i] = x0[i]
                    xupp[self.ocp.NQ + i] = x0[i]

                if xend[i] is not None:
                    xlow[self.ocp.NQ + self.ocp.NS - self.ocp.NX + i] = xend[i]
                    xupp[self.ocp.NQ + self.ocp.NS - self.ocp.NX + i] = xend[i]

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

                # integrate
                xs = self.ocp.integrate(p, q, s)

                # evaluate the objective
                F[0] = self.ocp.obj(xs, None, None, None, p, q, s)

                # evaluate the inequality constraints
                F[1:self.ocp.NCG + 1] = self.ocp.ineqc(xs, None, None, None, p, q, s)

                # evaluate the equality constraints
                F[self.ocp.NCG + 1:self.ocp.NC + 1] = self.ocp.eqc(xs, None, None, None, p, q, s)

                # evaluate the matching conditions
                F[self.ocp.NC + 1:] = self.ocp.mc(xs, None, None, None, p, q, s)

            if needG[0] != 0:

                # integrate
                xs, xs_dot_q = self.ocp.integrate_dq(p, q, s)
                xs_dot_s     = self.ocp.integrate_ds(p, q, s)[1]

                # calculate derivatives of objective
                G[0:self.ocp.NQ]  = self.ocp.obj_dq(xs, xs_dot_q, None, None, p, q, s)[1]  # controls
                G[self.ocp.NQ:NQ] = self.ocp.obj_ds(xs, xs_dot_s, None, None, p, q, s)[1]  # shooting variables
                l                 = NQ

                # calculate derivatives of inequality constraints
                ineqc_dq = self.ocp.ineqc_dq(xs, xs_dot_q, None, None, p, q, s)[1]
                ineqc_ds = self.ocp.ineqc_ds(xs, xs_dot_s, None, None, p, q, s)[1]

                for i in xrange(0, self.ocp.NCG):
                    G[l:l + self.ocp.NQ]      = ineqc_dq[i, :] # controls
                    G[l + self.ocp.NQ:l + NQ] = ineqc_ds[i, :] # shooting variables
                    l                         = l + NQ

                # calculate derivatives of equality constraints
                eqc_dq = self.ocp.eqc_dq(xs, xs_dot_q, None, None, p, q, s)[1]
                eqc_ds = self.ocp.eqc_ds(xs, xs_dot_s, None, None, p, q, s)[1]

                for i in xrange(0, self.ocp.NCH):
                    G[l:l + self.ocp.NQ]      = eqc_dq[i, :] # controls
                    G[l + self.ocp.NQ:l + NQ] = eqc_ds[i, :] # shooting variables
                    l                         = l + NQ

                # calculate derivatives of matching conditions
                mc_dq = self.ocp.mc_dq(xs, xs_dot_q, None, None, p, q, s)[1]
                mc_ds = self.ocp.mc_ds(xs, xs_dot_s, None, None, p, q, s)[1]

                for i in xrange(0, self.ocp.NMC):
                    G[l:l + self.ocp.NQ]      = mc_dq[i, :] # controls
                    G[l + self.ocp.NQ:l + NQ] = mc_ds[i, :] # shooting variables
                    l                         = l + NQ

                ################################  DEBUG: CHECK SECOND DERIVATIVES:

                # xs, xs_dot_q1, xs_dot_q2, xs_dot_dqdq = self.ocp.integrate_dqdq(p, q, s)
                # xs, xs_dot_s1, xs_dot_s2, xs_dot_dsds = self.ocp.integrate_dsds(p, q, s)
                # xs, xs_dot_p1, xs_dot_p2, xs_dot_dpdp = self.ocp.integrate_dpdp(p, q, s)
                # xs, xs_dot_s1, xs_dot_p2, xs_dot_dsdp = self.ocp.integrate_dsdp(p, q, s)
                # xs, xs_dot_s1, xs_dot_q2, xs_dot_dsdq = self.ocp.integrate_dsdq(p, q, s)
                # xs, xs_dot_p1, xs_dot_q2, xs_dot_dpdq = self.ocp.integrate_dpdq(p, q, s)

                # print self.ocp.c_dqdq(xs, xs_dot_q1, xs_dot_q2, xs_dot_dqdq, p, q, s)
                # print self.ocp.c_dsds(xs, xs_dot_s1, xs_dot_s2, xs_dot_dsds, p, q, s)
                # print self.ocp.c_dpdp(xs, xs_dot_p1, xs_dot_p2, xs_dot_dpdp, p, q, s)
                # print self.ocp.c_dsdp(xs, xs_dot_s1, xs_dot_p2, xs_dot_dsdp, p, q, s)
                # print self.ocp.c_dsdq(xs, xs_dot_s1, xs_dot_q2, xs_dot_dsdq, p, q, s)
                # print self.ocp.c_dpdq(xs, xs_dot_p1, xs_dot_q2, xs_dot_dpdq, p, q, s)

                # print self.ocp.mc_dqdq(xs, xs_dot_q1, xs_dot_q2, xs_dot_dqdq, p, q, s)
                # print self.ocp.mc_dsds(xs, xs_dot_s1, xs_dot_s2, xs_dot_dsds, p, q, s)
                # print self.ocp.mc_dpdp(xs, xs_dot_p1, xs_dot_p2, xs_dot_dpdp, p, q, s)
                # print self.ocp.mc_dsdp(xs, xs_dot_s1, xs_dot_p2, xs_dot_dsdp, p, q, s)
                # print self.ocp.mc_dsdq(xs, xs_dot_s1, xs_dot_q2, xs_dot_dsdq, p, q, s)
                # print self.ocp.mc_dpdq(xs, xs_dot_p1, xs_dot_q2, xs_dot_dpdq, p, q, s)

                # raw_input()

                ################################

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

        A = np.zeros((lenA[0],), dtype=np.float64)

        lenG    = np.zeros((1,), dtype=np.int32)
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

        iSpecs = np.zeros((1,), dtype=np.int32)
        iSumm  = np.zeros((1,), dtype=np.int32)
        iPrint = np.zeros((1,), dtype=np.int32)

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
        printn = self.ocp.path + "/output/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-snopt.out"
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

        return x[:self.ocp.NQ], x[self.ocp.NQ:], F[0], F[1:self.ocp.NC + 1], -Fmul[1:self.ocp.NC + 1], F[self.ocp.NC + 1:], -Fmul[self.ocp.NC + 1:]

"""
===============================================================================
"""
# -*- coding: utf-8 -*-
"""
Wrapper class for SNOPT sparse NLP solver.
"""
try:
    import snopt
except ImportError:
    err_s = "SNOPT is a third-party software a needs proper licensing!"
    raise ImportError(err_s)


# ------------------------------------------------------------------------------
class SNOPT(object):

    """Wrapper class for SNOPT sparse NLP solver."""

    def __init__(self, NV, NC):
        """Set up SNOPT solver."""
        # assign variables
        self.NV = NV
        self.NC = NC

        # initialize SNOPT
        snopt.check_memory_compatibility()
        minrw = np.zeros((1), dtype=np.int32)
        miniw = np.zeros((1), dtype=np.int32)
        mincw = np.zeros((1), dtype=np.int32)

        rw = np.zeros((10000,), dtype=np.float64)
        iw = np.zeros((10000,), dtype=np.int32)
        cw = np.zeros((8*500,), dtype=np.character)

        Cold  = np.array([0], dtype=np.int32)
        Basis = np.array([1], dtype=np.int32)
        Warm  = np.array([2], dtype=np.int32)

        x    = np.zeros((NV,), dtype=np.float64)
        xlow = np.zeros((NV,), dtype=np.float64)
        xupp = np.zeros((NV,), dtype=np.float64)
        xmul = np.zeros((NV,), dtype=np.float64)
        F    = np.zeros((1+NC,), dtype=np.float64)
        Flow = np.zeros((1+NC,), dtype=np.float64)
        Fupp = np.zeros((1+NC,), dtype=np.float64)
        Fmul = np.zeros((1+NC,), dtype=np.float64)

        ObjAdd = np.zeros((1,), dtype=np.float64)

        xstate = np.zeros((NV,), dtype=np.int32)
        Fstate = np.zeros((1+NC,), dtype=np.int32)

        INFO   = np.zeros((1,), dtype=np.int32)
        ObjRow = np.zeros((1,), dtype=np.int32)
        n      = np.zeros((1,), dtype=np.int32)
        neF    = np.zeros((1,), dtype=np.int32)

        lenA   = np.zeros((1,), dtype=np.int32)
        lenA[0] = NV * (1 + NC)

        iAfun = np.zeros((lenA[0],), dtype=np.int32)
        jAvar = np.zeros((lenA[0],), dtype=np.int32)

        A     = np.zeros((lenA[0],), dtype=np.float64)

        lenG   = np.zeros((1,), dtype=np.int32)
        lenG[0] = NV * (1 + NC)

        iGfun = np.zeros((lenG[0],), dtype=np.int32)
        jGvar = np.zeros((lenG[0],), dtype=np.int32)

        neA = np.zeros((1,), dtype=np.int32)
        neG = np.zeros((1,), dtype=np.int32)

        nxname = np.zeros((1,), dtype=np.int32)
        nFname = np.zeros((1,), dtype=np.int32)

        nxname[0] = 1
        nFname[0] = 1

        xnames = np.zeros((1*8,), dtype=np.character)
        Fnames = np.zeros((1*8,), dtype=np.character)
        Prob   = np.zeros((200*8,), dtype=np.character)

        iSpecs   = np.zeros((1,), dtype=np.int32)
        iSumm    = np.zeros((1,), dtype=np.int32)
        iPrint   = np.zeros((1,), dtype=np.int32)

        iSpecs[0] = 4
        iSumm [0] = 6
        iPrint[0] = 9

        printname = np.zeros((200*8,), dtype=np.character)
        specname  = np.zeros((200*8,), dtype=np.character)

        nS   = np.zeros((1,), dtype=np.int32)
        nInf = np.zeros((1,), dtype=np.int32)
        sInf = np.zeros((1,), dtype=np.float64)

        # open output files using snfilewrappers.[ch] */
        specn  = os.path.join("tmp", "sntoya.spc")
        printn = os.path.join("tmp", "sntoya.out")
        specname [:len(specn)]  = list(specn)
        printname[:len(printn)] = list(printn)

        # Open the print file, fortran style */
        snopt.snopenappend(iPrint, printname, INFO)

        # First,  sninit_ MUST be called to initialize optional parameters   */
        # to their default values.                                           */
        snopt.sninit(iPrint, iSumm, cw, iw, rw)

        # set up problem to be solved
        self.setup(
            INFO, Prob, neF, n, iAfun, jAvar, lenA, neA, A,
            iGfun, jGvar, lenG, neG, ObjAdd, ObjRow, xlow, xupp, Flow,
            Fupp, x, xstate, Fmul
        )

        # open spec file
        snopt.snfilewrapper(specname, iSpecs, INFO, cw, iw, rw)

        if INFO[0] != 101:
            print("Warning: Trouble reading specs file %s \n" % (specname))

    def solve(self, v):
        """Solve NLP for given initial value."""
        # set up problem to be solved
        self.setup(
            INFO, Prob, neF, n, iAfun, jAvar, lenA, neA, A,
            iGfun, jGvar, lenG, neG, ObjAdd, ObjRow, xlow, xupp, Flow,
            Fupp, x, xstate, Fmul
        )

        # call SNOPT
        snopt.snopta(
            Cold, neF, n, nxname, nFname,
            ObjAdd, ObjRow, Prob, evaluate,
            iAfun, jAvar, lenA, neA, A,
            iGfun, jGvar, lenG, neG,
            xlow, xupp, xnames, Flow, Fupp, Fnames,
            x, xstate, xmul, F, Fstate, Fmul,
            INFO, mincw, miniw, minrw,
            nS, nInf, sInf, cw, iw, rw, cw, iw, rw
        )

    def _setup(
        self, inform, Prob, neF, n, ObjAdd, ObjRow,
        xlow, xupp, Flow, Fupp, x, xstate, Fmul
    ):
    """
    On exit:
    inform      is 0 if there is enough storage, 1 otherwise.
    neF         is the number of problem functions
                (objective and constraints, linear and nonlinear).
    n           is the number of variables.
    xlow        holds the lower bounds on x.
    xupp        holds the upper bounds on x.
    Flow        holds the lower bounds on F.
    Fupp        holds the upper bounds on F.

    xstate(1:n) is a set of initial states for each x  (0,1,2,3,4,5).
    x (1:n)     is a set of initial values for x.
    Fmul(1:neF) is a set of initial values for the dual variables.

    ==================================================================
    """
    # Give the problem a name.
    Prob[:4] = list('Toy0')

    # Assign the dimensions of the constraint Jacobian */

    neF[0]    = 3
    n[0]      = 2

    ObjRow[0] = 1 # NOTE: Me must add one to mesh with fortran */
    ObjAdd[0] = 0

    # Set the upper and lower bounds. */
    xlow[0]   =   0.0
    xlow[1]   =  -1e6
    xupp[0]   =   1e6
    xupp[1]   =   1e6
    xstate[0] =   0
    xstate[1] =   0

    Flow[0] = -1e6
    Flow[1] = -1e6
    Flow[2] = -1e6
    Fupp[0] =  1e6
    Fupp[1] =  4.0
    Fupp[2] =  5.0
    x[0]    =  1.0
    x[1]    =  1.0

    def __del__(self):
        """Close file handles."""
        snopt.snclose(iPrint)
        snopt.snclose(iSpecs)


# ------------------------------------------------------------------------------

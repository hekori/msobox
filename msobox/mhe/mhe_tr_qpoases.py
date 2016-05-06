import sys
import os
import numpy
import scipy
from scipy import linalg as linalg
from matplotlib import pyplot as plt
import qpoases

from qpoases import PyQProblemB as QProblemB
from qpoases import PyOptions as Options
from qpoases import PyPrintLevel as PrintLevel


from .mhe import MHE
from .mhe import _show_state_information

class MHE_TR_QPOASES(MHE):
    """
    Trust Region algorithm using qpOASES.

    See Nocedal "Numerical Optimization" 2nd ed.
    Algorithm 4.1

    """

    @_show_state_information

    def __init__(self, mf, ind, dt=0.05, M=20, major=-1):
        """
        """
        MHE.__init__(self, mf, ind, dt=dt, M=M)

        self.NV = self.NX + self.NP

        self.lb = numpy.ones(self.NV) * (-numpy.infty)
        self.ub = numpy.ones(self.NV) * (numpy.infty)

        self.f     = -numpy.infty
        self.g     = numpy.zeros(self.NV, dtype=float)
        self.H     = numpy.zeros((self.NV, self.NV), dtype=float)
        self.tr_lb = numpy.ones(self.NV) * (-numpy.infty)
        self.tr_ub = numpy.ones(self.NV) * (numpy.infty)
        self.d     = numpy.zeros(self.NV)

        # basic trust region algorithm
        self.m          = - numpy.infty # model value m(x)
        self.m_old = 0 # previous model value
        self.f_old = 0 # previous f
        self.AI         = 0 # actual improvement
        self.PI         = 0 # predicted improvement
        self.eta        = 1./8.
        self.Delta_hat  = 1.
        self.delta      = self.Delta_hat

        self.qproblem      = QProblemB(self.NV)
        options            = Options()
        options.printLevel = PrintLevel.NONE
        self.qproblem.setOptions(options)

        self.nWSR = 1000

        self.qproblem.init(self.H, self.g, self.tr_lb, self.tr_ub, self.nWSR)

    def optimize(self):
        """
        min_d  m(d) = 0.5 * d.T * H * d + d.T * g + f
        s.t. |d| < delta

        returns 1 if another trust region iteration is necessary, 0 otherwise

        Example:

        while self.optimize() == 1:
            self.preparation_phase()
        """


        # build and solve QP to obtain d

        # objective function
        self.H[:,:] = self.JC.T.dot(self.JC)
        self.g[:]   = self.JC.T.dot(self.FC)

        # constraints
        self.tr_lb[:] = -self.delta
        self.tr_ub[:] =  self.delta

        self.qproblem.init(self.H, self.g, self.tr_lb, self.tr_ub, self.nWSR)
        self.qproblem.getPrimalSolution(self.d)


        self.f     = 0.5 * sum(self.FC**2)
        self.m     = 0.5 * self.d.T.dot(self.H).dot(self.d) + self.d.T.dot(self.g) + self.f

        self.AI    = self.f_old - self.f
        self.PI    = self.m_old - self.m

        print 'self.AI/self.PI=', self.AI/self.PI
        print 'self.delta=', self.delta

        if self.AI/self.PI < 0.25:
            self.delta *= 0.25

        else:
            if self.AI/self.PI > 0.75 and abs(numpy.linalg.norm(self.d) - self.delta) < 1.e-8:
                self.delta = min(2.*self.delta, self.Delta_hat)
            else:
                pass

        if self.AI/self.PI > self.eta:

            delta_s = self.e + self.Z.dot(self.d)
            delta_s = delta_s.reshape((self.M, self.NX))
            delta_p =  self.d[-self.NP:]

            # print 'delta_s=\n', delta_s
            # print 'delta_p=\n', delta_p

            proposed_s = self.s + delta_s
            proposed_p = self.p + delta_p

            self.s[:, :] += delta_s
            self.p[:]    += delta_p

            # if numpy.all(proposed_s >= self.lb[:self.NX]) and numpy.all(proposed_s <= self.ub[:self.NX]):
            #     self.s[:, :] += delta_s
            # else:
            #     print 'reject s'

            # if numpy.all(proposed_p >= self.lb[self.NX:]) and numpy.all(proposed_p <= self.ub[self.NX:]):
            #     self.p[:]    += delta_p

            # else:
            #     print 'reject p'


        self.plot_data.C[self.M-1, :, :]  = numpy.linalg.inv(self.H)
        self.plot_data.JC[self.M-1, :, :] = self.JC
        self.plot_data.p[self.M-1, :]     = self.p

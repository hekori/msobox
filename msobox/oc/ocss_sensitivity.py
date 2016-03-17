# -*- coding: utf-8 -*-

"""
===============================================================================

sensitivity analysis of optimal control problems discretized for single shooting ...

===============================================================================
"""

# system imports
from __future__ import division
import numpy as np

# local imports
pass

# setting options
np.set_printoptions(threshold=np.nan)  # print all array elements

"""
===============================================================================
"""

class OCSS_sensitivity(object):

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

    def dp(self, x0, p, q, mul):

        """

        ...

        input:

        output:

        TODO:

        """

        NQ  = self.ocp.NQ
        NP  = self.ocp.NP
        NC  = self.ocp.NC
        NTS = self.ocp.NTS

        if NC > 0:
            # evaluate the active constraints
            NCA, ca = self.active(x0, p, q)
            print ca, NCA

            # apply some reshaping for concatenation later on
            multi = np.reshape(mul[ca], (NCA,))

            # calculate all desired derivatives
            c_dq    = np.reshape(self.ocp.c_dq(x0, p, q)[ca, :], (NCA, NQ))
            c_dp    = np.reshape(self.ocp.c_dp(x0, p, q)[ca, :], (NCA, NP))
            c_dqdq  = np.reshape(self.ocp.c_dqdq(x0, p, q)[ca, :, :], (NCA, NQ, NQ))
            c_dqdp  = np.reshape(self.ocp.c_dqdp(x0, p, q)[ca, :, :], (NCA, NQ, NP))
            c_dpdp  = np.reshape(self.ocp.c_dpdp(x0, p, q)[ca, :, :], (NCA, NP, NP))

            # build up derivatives of the lagrange function by looping through active constraints
            lagrange_dp     = self.ocp.obj_dp(x0, p, q)
            lagrange_dq     = self.ocp.obj_dq(x0, p, q)
            lagrange_dpdp   = self.ocp.obj_dpdp(x0, p, q)
            lagrange_dqdq   = self.ocp.obj_dqdq(x0, p, q)
            lagrange_dqdp   = self.ocp.obj_dqdp(x0, p, q)

            for i in xrange(0, NCA):
                lagrange_dp     = lagrange_dp + multi[i] * c_dp[i, :]
                lagrange_dq     = lagrange_dq + multi[i] * c_dq[i, :]
                lagrange_dpdp   = lagrange_dpdp + multi[i] * c_dpdp[i, :, :]
                lagrange_dqdq   = lagrange_dqdq + multi[i] * c_dqdq[i, :, :]
                lagrange_dqdp   = lagrange_dqdp + multi[i] * c_dqdp[i, :, :]

            # concatenate matrices to kkt matrix and rhs
            kkt1    = np.concatenate((lagrange_dqdq, c_dq.transpose()), axis=1)
            kkt2    = np.concatenate((c_dq, np.zeros((NCA, NCA))), axis=1)
            kkt     = np.concatenate((kkt1, kkt2), axis=0)
            rhs     = np.concatenate((lagrange_dqdp, c_dp), axis=0)

            # calculate first order sensitivites of q and mul
            combined_dp = - np.dot(np.linalg.inv(kkt), rhs)

            # set sensitivites of mul
            mul_dp      = np.zeros((NC, NP))
            mul_dp[ca]  = combined_dp[NQ:, :]

            # set sensitivities of q
            q_dp    = np.zeros((NQ, NP))
            q_dp    = combined_dp[:NQ, :]

            # calculate first and second order sensitivites of the optimal value
            sol_dp       = lagrange_dp
            sol_dpdp     = np.dot(q_dp.transpose(), lagrange_dqdq)
            sol_dpdp     = np.dot(sol_dpdp, q_dp) + 2 * np.dot(lagrange_dqdp.transpose(), q_dp).transpose() + lagrange_dpdp

            q_dp_full = q_dp

        else:
            # build up derivatives of the lagrange function
            lagrange_dp     = self.ocp.obj_dp(x0, p, q)
            lagrange_dq     = self.ocp.obj_dq(x0, p, q)
            lagrange_dpdp   = self.ocp.obj_dpdp(x0, p, q)
            lagrange_dqdq   = self.ocp.obj_dqdq(x0, p, q)
            lagrange_dqdp   = self.ocp.obj_dqdp(x0, p, q)

            # HACK: manually trim last q out of matrices
            for i in xrange(0, self.ocp.NU):
                lagrange_dq     = np.delete(lagrange_dq, NTS * (i + 1) - (1 + i), 0)
                lagrange_dqdq   = np.delete(lagrange_dqdq, NTS * (i + 1) - (1 + i), 0)
                lagrange_dqdq   = np.delete(lagrange_dqdq, NTS * (i + 1) - (1 + i), 1)
                lagrange_dqdp   = np.delete(lagrange_dqdp, NTS * (i + 1) - (1 + i), 0)

            # calculate first order sensitivites of q and set mul to None
            q_dp    = - np.dot(np.linalg.inv(lagrange_dqdq), lagrange_dqdp)
            mul_dp  = None

            # calculate first and second order sensitivites of the solution
            sol_dp     = lagrange_dp
            sol_dpdp   = np.dot(q_dp.transpose(), lagrange_dqdq)
            sol_dpdp   = np.dot(sol_dpdp, q_dp) + 2 * np.dot(lagrange_dqdp.transpose(), q_dp).transpose() + lagrange_dpdp

            # HACK: add last q again
            q_dp_full = np.zeros((NQ, NP))

            for i in xrange(0, self.ocp.NU):
                q_dp_full[NTS * i:NTS * (i + 1) - 1] = \
                    q_dp[(NTS - 1) * i:(NTS - 1) * (i + 1)]

        return q_dp_full, mul_dp, sol_dp, sol_dpdp

    """
    ===============================================================================
    """

    def taylor(self, x0, p, q, mul, p_new):

        """

        ...

        input:

        output:

        TODO:

        """

        # calculate sensitivities
        q_dp, mul_dp, sol_dp, sol_dpdp = self.dp(x0, p, q, mul)

        # use taylor approximations to estimate the new values
        mul_new  = (mul + mul_dp * (p_new - p)) if (mul_dp is not None) else None
        q_new    = q + np.dot(q_dp, (p_new - p))
        sol_new  = self.ocp.obj(x0, p, q) + np.dot(sol_dp, (p_new - p)) + 1 / 2 * np.dot(np.dot((p_new - p).transpose(), sol_dpdp), (p_new - p))

        return q_new, mul_new, sol_new

    """
    ===============================================================================
    """

    def calculate_mul(self, x0, p, q):

        """

        ...

        input:

        output:

        TODO: implement RQ factiorization or update scipy

        """

        pass

    """
    ===============================================================================
    """

    def domain(self, x0, p, q, mul):

        """

        ...

        input:

        output:

        TODO:

        """

        NP = len(p)

        # evaluate the constraints for later use
        constraints_val = self.ocp.c(p, q)
        NC = len(constraints_val)
        constraints_totalder = np.zeros((NC, len(p)))

        # evaluate the active constraints and calculate sensitivites
        NCA, ca = self.active(p, q)
        q_dp, mul_dp, sol_dp, sol_dpdp = self.dp(p, q, mul)

        # calculate the total derivative of the constraints dG/dp
        for j in xrange(0, NC):

            for i in xrange(0, NP):

                constraints_totalder[j, i] = np.dot(self.ocp.c_dq(p, q)[j, :].transpose(), q_dp[:, i]) + self.ocp.c_dp(p, q)[j, i]

        # set up array to contain the results
        perturbations = np.zeros((NP, NC))

        # iterate through constraints and p
        for j in xrange(0, NC):

            for i in xrange(0, NP):

                # calculate the maximal perturbation allowed if constraint is active
                if j in ca:

                    if mul_dp[j, i] != 0:  # TODO: APPLY OTHER CRITERIA?

                        print p[i]
                        print mul[j] / mul_dp[j, i]

                        perturbations[i, j] = p[i] - mul[j] / mul_dp[j, i]

                    else:

                        perturbations[i, j] = - np.sign(mul[j]) / np.sign(mul_dp[j, i]) * np.inf

                # calculate the maximal perturbation allowed if constraint is not active
                elif j not in ca:

                    if constraints_totalder[j, i] != 0:

                        perturbations[i, j] = p[i] - constraints_val[j] / constraints_totalder[j, i]

                    else:

                        perturbations[i, j] = - np.sign(constraints_val[j]) / np.sign(constraints_totalder[j, i]) * np.inf

        # find maximal perturbations in each line
        domain = np.zeros((NP, 2))

        for i in xrange(0, NP):

            domain[i, 0] = np.min(perturbations[i, :])
            domain[i, 1] = np.max(perturbations[i, :])

        return domain

    """
    ===============================================================================
    """

    def active(self, x0, p, q):

        """

        ...

        input:

        output:

        TODO:

        """

        p = np.array(p)

        # evaluate the active constraints
        ca = []
        NCA = 0
        val = self.ocp.c(x0, p, q)

        if self.ocp.NC > 0:

            for i in xrange(0, val.size):

                if val[i] >= -1e-6:

                    ca = ca + [i]
                    NCA = NCA + 1

        return NCA, ca

    """
    ===============================================================================
    """

    def larger_perturbations(self, x0, p, q):

        """

        ...

        input:

        output:

        TODO: implement strategies for perturbations leaving the sensitivity domain

        """

        pass

"""
===============================================================================
"""
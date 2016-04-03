# -*- coding: utf-8 -*-

"""
===============================================================================

sensitivity analysis of optimal control problems discretized for single shooting ...

===============================================================================
"""

# system imports
import numpy as np

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

    def calculate_sensitivities(self, x_opt, x_opt_dp, x_opt_dq,
                                x_opt_dpdp, x_opt_dqdq, x_opt_dpdq,
                                p, q_opt, s_opt, mul_opt):

        """

        ...

        input:

        output:

        TODO:

        """

        if self.ocp.NC > 0:

            # calculate all desired derivatives
            c_dp   = self.ocp.c_dp(x_opt, x_opt_dp, None, None, p, q_opt, s_opt)[1]
            c_dq   = self.ocp.c_dq(x_opt, x_opt_dq, None, None, p, q_opt, s_opt)[1]
            c_dpdp = self.ocp.c_dpdp(x_opt, x_opt_dp, x_opt_dp, x_opt_dpdp, p, q_opt, s_opt)[3]
            c_dqdq = self.ocp.c_dqdq(x_opt, x_opt_dq, x_opt_dq, x_opt_dqdq, p, q_opt, s_opt)[3]
            c_dpdq = self.ocp.c_dpdq(x_opt, x_opt_dp, x_opt_dq, x_opt_dpdq, p, q_opt, s_opt)[3]

            # build up derivatives of the lagrange function by looping through active constraints
            lagrange_dp   = self.ocp.obj_dp(x_opt, x_opt_dp, None, None, p, q_opt, s_opt)[1]
            lagrange_dq   = self.ocp.obj_dq(x_opt, x_opt_dq, None, None, p, q_opt, s_opt)[1]
            lagrange_dpdp = self.ocp.obj_dpdp(x_opt, x_opt_dp, x_opt_dp, x_opt_dpdp, p, q_opt, s_opt)[3]
            lagrange_dqdq = self.ocp.obj_dqdq(x_opt, x_opt_dq, x_opt_dq, x_opt_dqdq, p, q_opt, s_opt)[3]
            lagrange_dpdq = self.ocp.obj_dpdq(x_opt, x_opt_dp, x_opt_dq, x_opt_dpdq, p, q_opt, s_opt)[3]

            for i in self.ca:
                lagrange_dp   = lagrange_dp + mul_opt[i] * c_dp[i, :]
                lagrange_dq   = lagrange_dq + mul_opt[i] * c_dq[i, :]
                lagrange_dpdp = lagrange_dpdp + mul_opt[i] * c_dpdp[i, :, :]
                lagrange_dqdq = lagrange_dqdq + mul_opt[i] * c_dqdq[i, :, :]
                lagrange_dpdq = lagrange_dpdq + mul_opt[i] * c_dpdq[i, :, :]

            # trim and reshape matrices for concatenation
            c_dp   = np.reshape(c_dp[self.ca, :], (self.NCA, self.ocp.NP))
            c_dq   = np.reshape(c_dq[self.ca, :], (self.NCA, self.ocp.NQ))
            c_dpdp = np.reshape(c_dpdp[self.ca, :, :], (self.NCA, self.ocp.NP, self.ocp.NP))
            c_dqdq = np.reshape(c_dqdq[self.ca, :, :], (self.NCA, self.ocp.NQ, self.ocp.NQ))
            c_dpdq = np.reshape(c_dpdq[self.ca, :, :], (self.NCA, self.ocp.NP, self.ocp.NQ))

            lagrange_dqdp = lagrange_dpdq.transpose()
            c_dqdp        = c_dpdq.transpose()

            # HACK: set lagrange_dqdq[-1, -1] = 1 to avoid singular matrix
            lagrange_dqdq[-1, -1] = 1

            # concatenate matrices to kkt matrix and rhs
            kkt1 = np.concatenate((lagrange_dqdq, c_dq.transpose()), axis=1)
            kkt2 = np.concatenate((c_dq, np.zeros((self.NCA, self.NCA))), axis=1)
            kkt  = np.concatenate((kkt1, kkt2), axis=0)
            rhs  = np.concatenate((lagrange_dqdp, c_dp), axis=0)

            # calculate first order sensitivites of q and mul
            combined_dp = -np.linalg.solve(kkt, rhs)

            # set sensitivities of mul
            mul_dp          = np.zeros((self.ocp.NC, self.ocp.NP))
            mul_dp[self.ca] = combined_dp[self.ocp.NQ:, :]

            # set sensitivities of q
            q_dp = np.zeros((self.ocp.NQ, self.ocp.NP))
            q_dp = combined_dp[:self.ocp.NQ, :]

            # calculate first and second order sensitivites of the optimal value
            F_dp   = lagrange_dp
            F_dpdp = np.dot(q_dp.transpose(), lagrange_dqdq)
            F_dpdp = np.dot(F_dpdp, q_dp) + 2 * np.dot(lagrange_dpdq, q_dp).transpose() + lagrange_dpdp

            q_dp_full = q_dp

        else:

            # build up derivatives of the lagrange function
            lagrange_dp   = self.ocp.obj_dp(x_opt, x_opt_dp, None, None, p, q_opt, s_opt)[1]
            lagrange_dq   = self.ocp.obj_dq(x_opt, x_opt_dq, None, None, p, q_opt, s_opt)[1]
            lagrange_dpdp = self.ocp.obj_dpdp(x_opt, x_opt_dp, x_opt_dp, x_opt_dpdp, p, q_opt, s_opt)[3]
            lagrange_dqdq = self.ocp.obj_dqdq(x_opt, x_opt_dq, x_opt_dq, x_opt_dqdq, p, q_opt, s_opt)[3]
            lagrange_dpdq = self.ocp.obj_dpdq(x_opt, x_opt_dp, x_opt_dq, x_opt_dpdq, p, q_opt, s_opt)[3]

            lagrange_dqdp = lagrange_dpdq.transpose()

            # HACK: set lagrange_dqdq[-1, -1] = 1 to avoid singular matrix
            lagrange_dqdq[-1, -1] = 1

            # # HACK: manually trim last q out of matrices
            # for i in xrange(0, self.ocp.NU):
            #     lagrange_dq   = np.delete(lagrange_dq, self.ocp.NTS * (i + 1) - (1 + i), 0)
            #     lagrange_dqdq = np.delete(lagrange_dqdq, self.ocp.NTS * (i + 1) - (1 + i), 0)
            #     lagrange_dqdq = np.delete(lagrange_dqdq, self.ocp.NTS * (i + 1) - (1 + i), 1)
            #     lagrange_dpdq = np.delete(lagrange_dpdq, self.ocp.NTS * (i + 1) - (1 + i), 0)

            # calculate first order sensitivites of q and set mul to None
            q_dp   = -np.linalg.solve(lagrange_dqdq, lagrange_dqdp)
            mul_dp = None

            # calculate first and second order sensitivites of the solution
            F_dp   = lagrange_dp
            F_dpdp = np.dot(q_dp.transpose(), lagrange_dqdq)
            F_dpdp = np.dot(F_dpdp, q_dp) + 2 * np.dot(lagrange_dpdq, q_dp).transpose() + lagrange_dpdp

            q_dp_full = q_dp

            # # HACK: add last q again
            # q_dp_full = np.zeros((self.ocp.NQ, self.ocp.NP))

            # for i in xrange(0, self.ocp.NU):
            #     q_dp_full[self.ocp.NTS * i:self.ocp.NTS * (i + 1) - 1] = q_dp[(self.ocp.NTS - 1) * i:(self.ocp.NTS - 1) * (i + 1)]

        # set results as attributes
        self.q_dp   = q_dp_full
        self.mul_dp = mul_dp
        self.F_dp   = F_dp
        self.F_dpdp = F_dpdp

    """
    ===============================================================================
    """

    def calculate_approximations(self, x_opt, p, q_opt, s_opt, mul_opt, p_new):

        """

        ...

        input:

        output:

        TODO:

        """

        mul_approx = mul_opt + np.dot(self.mul_dp, (p_new - p))
        q_approx   = q_opt + np.dot(self.q_dp, (p_new - p))
        F_approx   = self.ocp.obj(x_opt, None, None, None, p, q_opt, s_opt) + np.dot(self.F_dp, (p_new - p)) + \
                     1 / 2 * np.dot(np.dot((p_new - p).transpose(), self.F_dpdp), (p_new - p))

        # # use taylor approximations to estimate the new values
        # if self.NCA > 0:

        #     mul_approx = mul_opt + np.dot(self.mul_dp, (p_new - p))
        #     q_approx   = q_opt + np.dot(self.q_dp, (p_new - p))
        #     F_approx   = self.ocp.obj(x_opt, None, None, None, p, q_opt, s_opt) + np.dot(self.F_dp, (p_new - p)) + \
        #                  1 / 2 * np.dot(np.dot((p_new - p).transpose(), self.F_dpdp), (p_new - p))

        # else:

        #     mul_approx = None
        #     q_approx   = q_opt + np.dot(self.q_dp, (p_new - p))
        #     F_approx   = self.ocp.obj(x_opt, None, None, None, p, q_opt, s_opt) + np.dot(self.F_dp, (p_new - p)) + \
        #                  1 / 2 * np.dot(np.dot((p_new - p).transpose(), self.F_dpdp), (p_new - p))

        # set results as attributes
        self.q_approx   = q_approx
        self.mul_approx = mul_approx
        self.F_approx   = F_approx

    """
    ===============================================================================
    """

    def calculate_sensitivity_domain(self, x0, p, q, mul):

        """

        ...

        input:

        output:

        TODO:

        """

        # evaluate the constraints for later use
        constraints_val      = self.ocp.c(p, q)
        NC                   = len(constraints_val)
        constraints_totalder = np.zeros((NC, len(p)))

        # calculate sensitivites
        q_dp, mul_dp, F_dp, F_dpdp = self.dp(p, q, mul)

        # calculate the total derivative of the constraints dG/dp
        for j in xrange(0, NC):

            for i in xrange(0, self.ocp.NP):

                constraints_totalder[j, i] = np.dot(self.ocp.c_dq(p, q)[j, :].transpose(), q_dp[:, i]) + self.ocp.c_dp(p, q)[j, i]

        # set up array to contain the results
        perturbations = np.zeros((self.ocp.NP, NC))

        # iterate through constraints and p
        for j in xrange(0, NC):

            for i in xrange(0, self.ocp.NP):

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
        domain = np.zeros((self.ocp.NP, 2))

        for i in xrange(0, self.ocp.NP):

            domain[i, 0] = np.min(perturbations[i, :])
            domain[i, 1] = np.max(perturbations[i, :])

        return domain

    """
    ===============================================================================
    """

    def determine_active_constraints(self, x_opt, p, q_opt, s_opt):

        """

        ...

        input:

        output:

        TODO:

        """

        # evaluate the active constraints
        self.ca = []
        c = self.ocp.c(x_opt, None, None, None, p, q_opt, s_opt)

        if self.ocp.NC > 0:
            for i in xrange(0, c.size):

                if c[i] >= -1e-6:
                    self.ca = self.ca + [i]

        # set number of active constraints
        self.NCA = len(self.ca)

"""
===============================================================================
"""
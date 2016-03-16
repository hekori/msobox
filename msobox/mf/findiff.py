"""Finite difference back-end of indegrator."""

import os
import sys
import numpy


class BackendFiniteDifferences(object):

    """
    Finite difference back-end of indegrator.
    """

    _methods = (
        'ffcn',
        'ffcn_dot',
        'ffcn_bar',
        'ffcn_ddot',
    )

    def __init__(self, ffcn):
        """
        Load the provided pure Python back-end.

        Where ffcn is either:

        1)  a callable  of the form ``def ffcn(f, t, x, p, u)``
        2) or a path to a directory containing a file ffcn.py
        """
        if hasattr(ffcn, '__call__'):
            self.ffcn = ffcn
        else:
            self.path = os.path.abspath(ffcn)
            self.dir  = os.path.dirname(self.path)

            sys.path.insert(0, self.dir)
            try:
                import ffcn
            except ImportError:
                err_str = "Could not load 'ffcn' from module {}".format(ffcn)
                raise ImportError(err_str)

            self.ffcn = ffcn.ffcn

    def ffcn(self, f, t, x, p, u):
        """
        Evaluate right-hand side ffcn.

        Parameters
        ----------
        t : scalar
            current time for evaluation
        x : array-like (NX)
            current differential states of the system
        p : array-like (NP)
            current parameters of the system
        u : array-like (NU)
            current control input of the system

        Returns
        -------
        f : array-like (NX)
            evaluated right-hand side function
        """
        self.ffcn(f, t, x, p, u)

    def ffcn_dot(self, f, f_dot, t, x, x_dot, p, p_dot, u, u_dot):
        """
        Evaluate right-hand side first derivative using finite differences.

        Parameters
        ----------
        t : scalar
            current time for evaluation
        x : array-like (NX)
            current differential states of the system
        x_dot : array-like (NX, P)
        p : array-like (NP)
            current parameters of the system
        p_dot : array-like (NP, P)
        u : array-like (NU)
            current control input of the system
        u_dot : array-like (NU, P)

        Returns
        -------
        f : array-like (NX)
            evaluated right-hand side function
        f_dot : array-like (NX, P)
            evaluated right-hand side function
        """
        # evaluate nominal solution
        self.ffcn(f, t, x, p, u)
        tmp = f.copy()

        # evaluate directional derivatives
        assert (  # check number of directions
            f_dot.shape[1] == x_dot.shape[1] ==
            p_dot.shape[1] == u_dot.shape[1]
        ), 'Different number of directions in inputs'
        P = f_dot.shape[1]  # get number of directions
        EPS = 1e-08  # perturbation of finite difference

        for i in range(P):
            # disturb in x direction
            self.ffcn(
                f_dot[:, i],
                t,
                x + EPS*x_dot[:, i],
                p + EPS*p_dot[:, i],
                u + EPS*u_dot[:, i]
            )
            # calculate finite difference
            f_dot[:, i] = (f_dot[:, i] - f) / EPS

            # # disturb in x direction
            # self.ffcn(
            #     f_dot[:, i],
            #     t,
            #     x + EPS*x_dot[:, i],
            #     p,
            #     u
            # )
            # # calculate finite difference
            # f_dot[:, i] = (f_dot[:, i] - f) / EPS

            # # disturb in p direction
            # self.ffcn(
            #     f_dot[:, i],
            #     t,
            #     x,
            #     p + EPS*p_dot[:, i],
            #     u
            # )
            # # calculate finite difference
            # f_dot[:, i] = (f_dot[:, i] - f) / EPS

            # # disturb in u direction
            # self.ffcn(
            #     f_dot[:, i],
            #     t,
            #     x,
            #     p,
            #     u + EPS*u_dot[:, i]
            # )
            # # calculate finite difference
            # f_dot[:, i] = (f_dot[:, i] - f) / EPS


    def ffcn_bar(self, f, f_bar, t, x, x_bar, p, p_bar, u, u_bar):
        """
        Dummy reverse derivative of right-hand side function.

        Parameters
        ----------
        t : scalar
            current time for evaluation
        x : array-like (NX)
            current differential states of the system
        x_bar : array-like (,)
        p : array-like ()
            current parameters of the system
        p_bar : array-like (,)
        u : array-like ()
            current control input of the system
        u_bar : array-like (,)

        Returns
        -------
        f : array-like (NX)
            evaluated right-hand side function
        f_bar : array-like (,)
        """
        err_str('This is not possible with finite differences?')
        raise NotImplementedError(err_str)

    def ffcn_ddot(
        self,
        f, f_dot2, f_dot1, f_ddot,
        t,
        x, x_dot2, x_dot1, x_ddot,
        p, p_dot2, p_dot1, p_ddot,
        u, u_dot2, u_dot1, u_ddot
    ):
        """
        Dummy second order forward derivative of right-hand side function.

        Parameters
        ----------
        t : scalar
            current time for evaluation
        x : array-like (NX)
            current differential states of the system
        x_dot2 : array-like (,)
        x_dot1 : array-like (,)
        x_ddot : array-like (,)
        p : array-like (NO)
            current parameters of the system
        p_dot2 : array-like (,)
        p_dot1 : array-like (,)
        p_ddot : array-like (,)
        u : array-like (NU)
            current control input of the system
        u_dot2 : array-like (,)
        u_dot1 : array-like (,)
        u_ddot : array-like (,)

        Returns
        -------
        f : array-like (NX)
            evaluated right-hand side function
        f_dot2 : array-like (,)
        f_dot1 : array-like (,)
        f_ddot : array-like (,)
        """
        raise NotImplementedError

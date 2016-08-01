#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rocket car example python model functions and derivatives."""


# ------------------------------------------------------------------------------
class MF(object):

    """Model function definitions of the rocket car example."""

    def lfcn(self, l, t, x, u):
        """
        Lagrange objective of the rocket car example, which is one possible
        formulation of the minimal time optimal control problem, in the form
        of::

            min int_0^T L(t, x, u) dt = int_0^T 1 dt = T
             T

        .. NOTE::
            add an additional state to efficiently solve the objective.

        .. NOTE::
            this has to be rescaled as well

        """
        l[0] = 1.0

    def lfcn_dot(self, l, l_dot, t, x, x_dot, u, u_dot):
        """Implement first-order forward derivative of lfcn."""
        l_dot[0, :] = 0.0
        l[0] = 1.0

    def mfcn(self, m, t, x, p):
        """
        Mayer objective of the rocket car example, which is one possible
        formulation of the minimal time optimal control problem, in the form
        of::

            min E(T, x(T), p) = T
             T
        """
        m[0] = p[0]

    def mfcn_dot(self, m, m_dot, t, x, x_dot, p, p_dot):
        """Implement first-order forward derivative of mfcn."""
        m_dot[0, :] = p_dot[0, :]
        m[0] = p[0]

    def ffcn(self, f, t, x, u):
        """
        Model of the rocket car with time transformation on unit interval.

        The reduced order model of the point mass with rocket engine given by
        the control -1 <= u(t) <= 1 for all t in [0, T] is give by::

            x_dot[0] = x[1],
            x_dot[1] = u[0],

        """
        f[0] = x[1]
        f[1] = u[0]

    def ffcn_dot(self, f, f_dot, t, x, x_dot, u, u_dot):
        """
        Implement first-order forward derivative.
        """
        f_dot[0, :] = x_dot[1, :]
        f_dot[1, :] = u_dot[0, :]

        f[0] = x[1]
        f[1] = u[0]

    def rfcn_0(self, r, t, x):
        """
        Start point constraint ensuring the reaching of a fixed end position and
        velocity, given by::

            r(0, x) = [x[0] - 0] = 0
                      [x[1] - 0]
        """
        r[0] = x[0]
        r[1] = x[1]

    def rfcn_0_dot(self, r, r_dot, t, x, x_dot):
        r_dot[0, :] = x_dot[0, :]
        r_dot[1, :] = x_dot[1, :]

        r[0] = x[0]
        r[1] = x[1]

    def rfcn_f(self, r, t, x):
        """
        Endpoint constraint ensuring the reaching of a fixed end position and
        velocity, given by::
        """
        r[0] = x[0] - 1.0
        r[1] = x[1]

        return None

    def rfcn_f_dot(self, r, r_dot, t, x, x_dot):
        r_dot[0, :] = x_dot[0, :]
        r_dot[1, :] = x_dot[1, :]

        r[0] = x[0] - 1.0
        r[1] = x[1]

        return None



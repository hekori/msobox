# -*- coding: utf-8 -*-
"""Implementation of explicit Euler integration scheme."""

import numpy

from msobox.ind.base import IND

class ExplicitEuler(IND):

    def zo_forward(self, ts, x0, p, q):
        """
        Solve nominal differential equation using an explicit Euler scheme.

        Parameters
        ----------
        ts : array-like (NTS,)
            time grid for integration
        x0 : array-like (NX,)
            initial value of the problem
        p : array-like (NP)
            current parameters of the system
        q : array-like (NU)
            current control discretization of the system
        """
        # check if dimensions fit
        self.zo_check(ts, x0, p, q)

        # store initial value
        self.xs[0, :] = x0

        # integrate forward
        for i in range(self.NTS-1):
            # update control discretization
            self.update_u(i, self.ts[i])

            # calculate step size
            h = self.ts[i+1] - self.ts[i]

            # evaluate one-step function
            self.mf.ffcn(
                self.f, self.ts[i:i+1], self.xs[i, :], self.p, self.u
            )

            # compute next state
            self.xs[i + 1, :] = self.xs[i,:] + h*self.f

        return self.xs[-1, ...]


    def fo_forward(self, ts, x0, x0_dot, p, p_dot, q, q_dot):
        """
        Solve nominal differential equation and evaluate first-order forward
        sensitivities of the differential states using an explicit Euler scheme.

        Parameters
        ----------
        ts : array-like (NTS,)
            time grid for integration
        x0 : array-like (NX,)
            initial value of the problem
        x0_dot : array-like (NX, P)
            forward directions for derivative evaluation wrt. x0
        p : array-like (NP,)
            Current parameters of the system
        p_dot : array-like (NP, P)
            forward directions for derivative evaluation wrt. p
        q : array-like (NU,)
            current control discretization of the system
        q_dot : array-like (NU, P)
            forward directions for derivative evaluation wrt. q
        """
        self.fo_check(ts, x0, x0_dot, p, p_dot, q, q_dot)

        self.xs[0, :] = x0
        self.xs_dot[0, :, :] = x0_dot

        for i in range(self.NTS-1):
            self.update_u_dot(i, self.ts[i])
            h = self.ts[i+1] - self.ts[i]

            self.xs_dot[i + 1, :, :] = self.xs_dot[i,:, :]
            self.xs[i + 1, :] = self.xs[i, :]

            self.mf.ffcn_dot(
                self.f, self.f_dot,
                self.ts[i:i+1],
                self.xs[i, :], self.xs_dot[i, :, :],
                self.p, self.p_dot,
                self.u, self.u_dot
            )

            self.xs_dot[i + 1, :, :] += h*self.f_dot
            self.xs[i + 1, :] += h*self.f

        return self.xs[-1, ...], self.xs_dot[-1, ...]


    def so_forward(self, ts,
                           x0, x0_dot2, x0_dot1, x0_ddot,
                           p,   p_dot2,  p_dot1, p_ddot,
                           q,   q_dot2,  q_dot1, q_ddot):
        """
        Solve nominal differential equation and evaluate first-order as well as
        second-order forward sensitivities of the differential states using an
        explicit Euler scheme.

        Parameters
        ----------
        ts : array-like (NTS,)
            time grid for integration
        x0 : array-like (NX,)
            initial value of the problem
        x0_dot2 : array-like (NX, P)
            first-order forward directions for derivative evaluation wrt. x0
        x0_dot1 : array-like (NX, P)
            first-order forward directions for derivative evaluation wrt. x0
        x0_ddot : array-like (NX, P)
            second-order forward directions for derivative evaluation wrt. x0
        p : array-like (NP,)
            Current parameters of the system
        p_dot2 : array-like (NP, P)
            first-order forward directions for derivative evaluation wrt. p
        p_dot1 : array-like (NP, P)
            first-order forward directions for derivative evaluation wrt. p
        p_ddot : array-like (NP, P)
            second-order forward directions for derivative evaluation wrt. p
        q : array-like (NU,)
            current control discretization of the system
        q_dot2 : array-like (NU, P)
            first-order forward directions for derivative evaluation wrt. q
        q_dot1 : array-like (NU, P)
            first-order forward directions for derivative evaluation wrt. q
        q_ddot : array-like (NU, P)
            second-order forward directions for derivative evaluation wrt. q
        """
        self.so_check(ts,
                 x0, x0_dot2, x0_dot1, x0_ddot,
                 p,   p_dot2,  p_dot1, p_ddot,
                 q,   q_dot2,  q_dot1, q_ddot)

        self.xs[0, :] = x0
        self.xs_dot1[0, :, :] = x0_dot1
        self.xs_dot2[0, :, :] = x0_dot2
        self.xs_ddot[0, :, :, :] = x0_ddot

        for i in range(self.NTS-1):
            self.update_u_ddot(i, self.ts[i])
            h = self.ts[i+1] - self.ts[i]

            self.xs_ddot[i + 1, :, :, :] = self.xs_ddot[i, :, :, :]
            self.xs_dot1[i + 1, :, :] = self.xs_dot1[i, :, :]
            self.xs_dot2[i + 1, :, :] = self.xs_dot2[i, :, :]
            self.xs[i + 1, :] = self.xs[i, :]

            self.mf.ffcn_ddot(
                self.f, self.f_dot2, self.f_dot1, self.f_ddot,
                self.ts[i:i+1],
                self.xs[i, :], self.xs_dot2[i, :, :], self.xs_dot1[i, :, :], self.xs_ddot[i, :, :, :],
                self.p, self.p_dot2, self.p_dot1, self.p_ddot,
                self.u, self.u_dot2, self.u_dot1, self.u_ddot
            )

            self.xs_ddot[i + 1, :, :, :] += h*self.f_ddot
            self.xs_dot1[i + 1, :, :] += h*self.f_dot1
            self.xs_dot2[i + 1, :, :] += h*self.f_dot2
            self.xs[i + 1, :] += h*self.f

        return self.xs[-1, ...], self.xs_dot1[-1, ...], self.xs_dot2[-1, ...], self.xs_ddot[-1, ...]


    def fo_reverse(self, xs_bar):
        """
        Solve nominal differential equation and evaluate first-order forward
        sensitivities of the differential states using an explicit Euler scheme.

        .. note:: Assumes zo_forward before calling fo_reverse!

        Parameters
        ----------
        xs_bar : array-like (NX, Q)
            backward directions for evaluation of derivatives wrt. x0, p, q.
        """

        t = numpy.zeros(1)
        self.xs_bar = xs_bar.copy()

        self.x0_bar = numpy.zeros(self.x0.shape)
        self.f_bar = numpy.zeros(self.f.shape)
        self.p_bar = numpy.zeros(self.p.shape)
        self.q_bar = numpy.zeros(self.q.shape)
        self.u_bar = numpy.zeros(self.u.shape)

        for i in range(self.NTS-1)[::-1]:
            h = self.ts[i+1] - self.ts[i]

            # reverse
            t[0] = self.ts[i]
            self.update_u(i, self.ts[i])
            self.xs_bar[i, :] += self.xs_bar[i + 1, :]
            self.f_bar[:] = h*self.xs_bar[i+1, :]
            self.mf.ffcn_bar(
                self.f, self.f_bar,
                t,
                self.xs[i, :], self.xs_bar[i,:],
                self.p, self.p_bar,
                self.u, self.u_bar
            )

            self.xs_bar[i + 1, :] = 0
            self.update_u_bar(i, t[0])

        self.x0_bar[:] += self.xs_bar[0, :]
        self.xs_bar[0, :] = 0.

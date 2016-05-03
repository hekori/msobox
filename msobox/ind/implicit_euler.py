# -*- coding: utf-8 -*-

"""Implementation of implicit Euler integration scheme."""

# system imports
import numpy as np
import scipy.optimize as opt

from msobox.ind.base import IND


class ImplicitEuler(IND):

    def zo_forward(self, ts, x0, p, q):

        """
        Solve nominal differential equation using an implicit Euler scheme.

        Parameters
        ----------
        ts : array-like (NTS,)
            time grid for integration
        x0 : array-like (NX,)
            initial value of the problem
        p : array-like (NP)
            current parameters of the system
        q : array-like (NU) # Q: NQ instead of NU?
            current control discretization of the system
        """

        # check if dimensions fit
        self.zo_check(ts, x0, p, q)

        # store initial value
        self.xs[0, :] = x0

        # integrate forward
        for i in range(self.NTS - 1):

            # update control discretization
            self.update_u(i, self.ts[i])

            # calculate step size
            h = self.ts[i + 1] - self.ts[i]

            # define function zo_implicit(x) = x - x_i - h * f(t_i+1, x)
            def zo_implicit(x):

                # evaluate model
                self.mf.ffcn(self.f,
                                self.ts[i + 1:i + 2],
                                x,
                                self.p,
                                self.u)

                return x - self.xs[i, :] - h * self.f

            # compute next step by finding root of zo_implicit
            res               = opt.root(fun=zo_implicit, x0=self.xs[i, :], jac=False)
            self.xs[i + 1, :] = res.x

        return self.xs[-1, ...]


    def fo_forward(self, ts, x0, x0_dot, p, p_dot, q, q_dot):

        """
        Solve nominal differential equation and evaluate first-order forward
        sensitivities of the differential states using an implicit Euler scheme.

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

        # check if dimensions fit
        self.fo_check(ts, x0, x0_dot, p, p_dot, q, q_dot)

        # store initial value
        self.xs[0, :]        = x0
        self.xs_dot[0, :, :] = x0_dot

        # integrate forward
        for i in range(self.NTS - 1):

            # update control discretization
            self.update_u_dot(i, self.ts[i])

            # calculate step size
            h = self.ts[i + 1] - self.ts[i]

            # define function fo_implicit(x, J) = [x - x_i - h * f,
            #                                      J - dx_i/dv - h * df/dv]
            def fo_implicit(var):

                # rebuild arrays from input for readability
                x = var[:self.NX]
                J = var[self.NX:]
                J = J.reshape((self.NX, self.P))

                # evaluate model
                self.mf.ffcn_dot(self.f, self.f_dot,
                                    self.ts[i + 1:i + 2],
                                    x, J,
                                    self.p, self.p_dot,
                                    self.u, self.u_dot)

                zo = x - self.xs[i, :] - h * self.f
                fo = J - self.xs_dot[i, :, :] - h * self.f_dot

                return np.concatenate((zo, fo.ravel()), axis=0)

            # compute next step by finding root of fo_implicit
            x_start                  = np.concatenate((self.xs[i, :], self.xs_dot[i, :, :].ravel()), axis=0)
            res                      = opt.root(fun=fo_implicit, x0=x_start, jac=False)
            self.xs[i + 1, :]        = res.x[:self.NX]
            self.xs_dot[i + 1, :, :] = res.x[self.NX:].reshape((self.NX, self.P))

        return self.xs[-1, ...], self.xs_dot[-1, ...]


    def so_forward(self, ts,
                           x0, x0_dot2, x0_dot1, x0_ddot,
                           p,   p_dot2,  p_dot1, p_ddot,
                           q,   q_dot2,  q_dot1, q_ddot):

        """
        Solve nominal differential equation and evaluate first-order as well as
        second-order forward sensitivities of the differential states using an
        implicit Euler scheme.

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

        # check if dimensions fit
        self.so_check(ts,
                      x0, x0_dot2, x0_dot1, x0_ddot,
                      p,   p_dot2,  p_dot1, p_ddot,
                      q,   q_dot2,  q_dot1, q_ddot)

        # store initial values
        self.xs[0, :]            = x0
        self.xs_dot1[0, :, :]    = x0_dot1
        self.xs_dot2[0, :, :]    = x0_dot2
        self.xs_ddot[0, :, :, :] = x0_ddot

        # integrate forward
        for i in range(self.NTS-1):

            # update control discretization
            self.update_u_ddot(i, self.ts[i])

            # calculate step size
            h = self.ts[i+1] - self.ts[i]

            # define function so_implicit(x, J, H) = [x- x_i - h * f,
            #                                         J - dx_i/dv - h * df/dv,
            #                                         H - d^2x_i/dv^2 - h * d^2f/dv^2]
            def so_implicit(var):

                # rebuild arrays from input for readability
                x  = var[:self.NX]
                J1 = var[self.NX:self.NX + self.NX * self.P1]
                J1 = J1.reshape((self.NX, self.P1))
                J2 = var[self.NX + self.NX * self.P1:self.NX + self.NX * self.P1 + self.NX * self.P2]
                J2 = J2.reshape((self.NX, self.P2))
                H  = var[self.NX + self.NX * self.P1 + self.NX * self.P2:]
                H  = H.reshape((self.NX, self.P1, self.P2))

                # evaluate model
                self.mf.ffcn_ddot(self.f, self.f_dot2, self.f_dot1, self.f_ddot,
                                     self.ts[i + 1:i + 2],
                                     x, J2, J1, H,
                                     self.p, self.p_dot2, self.p_dot1, self.p_ddot,
                                     self.u, self.u_dot2, self.u_dot1, self.u_ddot)

                zo  = x - self.xs[i, :] - h * self.f
                fo1 = J1 - self.xs_dot1[i, :, :] - h * self.f_dot1
                fo2 = J2 - self.xs_dot2[i, :, :] - h * self.f_dot2
                so  = H - self.xs_ddot[i, :, :] - h * self.f_ddot

                # reshape results to one-dimensional output
                return np.concatenate((zo, fo1.ravel(), fo2.ravel(), so.ravel()), axis=0)

            # compute next step by finding root of so_implicit
            x_start                      = np.concatenate((self.xs[i, :],
                                                          self.xs_dot1[i, :, :].ravel(), self.xs_dot2[i, :, :].ravel(),
                                                          self.xs_ddot[i, :, :, :].ravel()), axis=0)
            res                          = opt.root(fun=so_implicit, x0=x_start, jac=False)
            self.xs[i + 1, :]            = res.x[:self.NX]
            self.xs_dot1[i + 1, :, :]    = res.x[self.NX:self.NX + self.NX * self.P1].reshape((self.NX, self.P1))
            self.xs_dot2[i + 1, :, :]    = res.x[self.NX + self.NX * self.P1:
                                                 self.NX + self.NX * self.P1 + self.NX * self.P2].reshape((self.NX, self.P2))
            self.xs_ddot[i + 1, :, :, :] = res.x[self.NX + self.NX * self.P1 + self.NX * self.P2:].reshape((self.NX, self.P1, self.P2))

        return self.xs[-1, ...], self.xs_dot1[-1, ...], self.xs_dot2[-1, ...], self.xs_ddot[-1, ...]


    def fo_reverse(self, xs_bar):

        """
        Solve nominal differential equation and evaluate first-order forward
        sensitivities of the differential states using an implicit Euler scheme.

        .. note:: Assumes zo_forward before calling fo_reverse!

        Parameters
        ----------
        xs_bar : array-like (NX, Q)
            backward directions for evaluation of derivatives wrt. x0, p, q.
        """

        raise NotImplementedError

# -*- coding: utf-8 -*-

"""Implementation of implicit Euler integration scheme."""

# system imports
import numpy as np
import scipy.optimize as opt

class ImplicitEuler(object):

    """Implementation of implicit Euler integration scheme."""

    def __init__(self, model):

        """
        Initialize Euler integration scheme from model.

        Parameters
        ----------
        model : msobox Model instance
            Model object providing right-hand side and its derivatives
        """

        self.printlevel = 0

        # TODO: Get dimensions from model instance
        self.NY  = 0  # number of differential variables y
        self.NZ  = 0  # number of algebraic variables z
        self.NX  = 0  # number of variables x = (y,z)
        self.NP  = 0  # number of parameters
        self.NU  = 0  # number of control functions
        self.NQI = 0  # number of q in one control interval

        # instance of model function and derivatives
        # TODO: add check on instance
        # err_str = "model is not an instance of MSOBox Model object."
        # assert isinstance(model, Model), err_str
        self.model = model

    def zo_check(self, ts, x0, p, q):

        """Check for dimensions and allocate memory."""

        # set dimensions
        self.NTS = ts.size                   # number of time steps
        self.NQ  = self.NU * self.NTS * 2    # number of control variables # Q: why 2? parametrization of controls?

        # assert that the dimensions match
        self.NX = x0.size
        self.NP = p.size

        self.NU  = q.shape[0]
        self.NTS = q.shape[1]

        # assign variables
        self.ts = ts
        self.x0 = x0
        self.p  = p
        self.q  = q

        # allocate memory
        self.xs = np.zeros((self.NTS, self.NX))
        self.f  = np.zeros(self.NX)
        self.u  = np.zeros(self.NU)

    def fo_check(self, ts, x0, x0_dot, p, p_dot, q, q_dot):

        """Check for dimensions and allocate memory."""

        self.zo_check(ts, x0, p, q)

        self.P = x0_dot.shape[1]

        assert self.NP == p_dot.shape[0]
        assert self.P == p_dot.shape[1]
        assert self.P == q_dot.shape[3]

        # assign variables
        self.x0_dot = x0_dot
        self.p_dot  = p_dot
        self.q_dot  = q_dot

        # allocate memory
        self.xs_dot = np.zeros((self.NTS, self.NX, self.P))
        self.f_dot  = np.zeros((self.NX, self.P))
        self.u_dot  = np.zeros((self.NU, self.P))

    def so_check(self, ts,
                 x0, x0_dot2, x0_dot1, x0_ddot,
                 p, p_dot2, p_dot1, p_ddot,
                 q, q_dot2, q_dot1, q_ddot):

        """Check for dimensions and allocate memory."""

        self.zo_check(ts, x0, p, q)

        self.P1 = x0_dot1.shape[1]
        self.P2 = x0_dot2.shape[1]

        assert self.NP == p_dot1.shape[0]
        assert self.NP == p_dot2.shape[0]
        assert self.NP == p_ddot.shape[0]

        assert self.P1 == p_dot1.shape[1]
        assert self.P1 == q_dot1.shape[3]

        assert self.P2 == p_dot2.shape[1]
        assert self.P2 == q_dot2.shape[3]

        assert self.P1 == x0_ddot.shape[1]
        assert self.P1 == p_ddot.shape[1]
        assert self.P1 == q_ddot.shape[3]

        assert self.P2 == x0_ddot.shape[2]
        assert self.P2 == p_ddot.shape[2]
        assert self.P2 == q_ddot.shape[4]

        # assign variables
        self.x0_dot1 = x0_dot1
        self.p_dot1  = p_dot1
        self.q_dot1  = q_dot1

        self.x0_dot2 = x0_dot2
        self.p_dot2  = p_dot2
        self.q_dot2  = q_dot2

        self.x0_ddot = x0_ddot
        self.p_ddot  = p_ddot
        self.q_ddot  = q_ddot

        # allocate memory
        self.xs_dot1 = np.zeros((self.NTS, self.NX, self.P1))
        self.xs_dot2 = np.zeros((self.NTS, self.NX, self.P2))
        self.xs_ddot = np.zeros((self.NTS, self.NX, self.P1, self.P2))

        self.f_dot1 = np.zeros((self.NX, self.P1))
        self.f_dot2 = np.zeros((self.NX, self.P2))
        self.f_ddot = np.zeros((self.NX, self.P1, self.P2))

        self.u_dot1 = np.zeros((self.NU, self.P1))
        self.u_dot2 = np.zeros((self.NU, self.P2))
        self.u_ddot = np.zeros((self.NU, self.P1, self.P2))

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
                self.model.ffcn(self.f,
                                self.ts[i + 1:i + 2],
                                x,
                                self.p,
                                self.u)

                return x - self.xs[i, :] - h * self.f

            # compute next step by finding root of zo_implicit
            res               = opt.root(fun=zo_implicit, x0=self.xs[i, :], jac=False)
            self.xs[i + 1, :] = res.x

    def fo_forward_xpq(self, ts, x0, x0_dot, p, p_dot, q, q_dot):

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
                self.model.ffcn_dot(self.f, self.f_dot,
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

    def so_forward_xpq_xpq(self, ts,
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
                self.model.ffcn_ddot(self.f, self.f_dot2, self.f_dot1, self.f_ddot,
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

    def update_u(self, i, t):

        """Update control discretization for step i."""

        self.u[:] = self.q[:, i, 0]

    def update_u_dot(self, i, t):

        """Update control discretization for step i."""

        self.u[:]        = self.q[:, i, 0]
        self.u_dot[:, :] = self.q_dot[:, i, 0, :]

    def update_u_bar(self, i, t):

        """Update control discretization for step i."""

        self.q_bar[:, i, 0] = self.q_bar[:, i, 0] + self.u_bar[:]
        self.u_bar[:]       = 0

    def update_u_ddot(self, i, t):

        """Update control discretization for step i."""

        self.u[:]         = self.q[:, i, 0]
        self.u_dot1[:, :] = self.q_dot1[:, i, 0, :]
        self.u_dot2[:, :] = self.q_dot2[:, i, 0, :]
        self.u_ddot[:, :] = self.q_ddot[:, i, 0, :, :]
# -*- coding: utf-8 -*-

import numpy

class IND(object):

    """Base class for ind schemes"""

    def __init__(self, model):
        """

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
        self.mf = model


    def zo_check(self, ts, x0, p, q):
        """Check for dimensions and allocate memory."""
        # set dimeions
        self.NTS = ts.size              # number of time steps
        self.NQ = self.NU*self.NTS*2     # number of control variables

        # assert that the dimensions match
        self.NX = x0.size
        self.NP = p.size

        self.NU = q.shape[0]

        # assign variables
        self.ts = ts
        self.x0 = x0
        self.p = p
        self.q = q

        # allocate memory
        self.xs = numpy.zeros((self.NTS, self.NX))
        self.f = numpy.zeros(self.NX)
        self.u = numpy.zeros(self.NU)


    def fo_check(self, ts, x0, x0_dot, p, p_dot, q, q_dot):
        """Check for dimensions and allocate memory."""
        self.zo_check(ts, x0, p, q)

        self.P = x0_dot.shape[1]

        assert self.NP == p_dot.shape[0]

        assert self.P == p_dot.shape[1]
        assert self.P == q_dot.shape[1]

        # assign variables
        self.x0_dot = x0_dot
        self.p_dot = p_dot
        self.q_dot = q_dot

        # allocate memory
        self.xs_dot = numpy.zeros((self.NTS, self.NX, self.P))
        self.f_dot = numpy.zeros((self.NX, self.P))
        self.u_dot = numpy.zeros((self.NU, self.P))


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
        assert self.P1 == q_dot1.shape[1], "%d != %d" %(self.P1, q_dot1.shape[1])

        assert self.P2 == p_dot2.shape[1]
        assert self.P2 == q_dot2.shape[1]

        assert self.P1 == x0_ddot.shape[1]
        assert self.P1 == p_ddot.shape[1]
        assert self.P1 == q_ddot.shape[1]

        assert self.P2 == x0_ddot.shape[2]
        assert self.P2 == p_ddot.shape[2]
        assert self.P2 == q_ddot.shape[2]

        # assign variables
        self.x0_dot1 = x0_dot1
        self.p_dot1 = p_dot1
        self.q_dot1 = q_dot1

        self.x0_dot2 = x0_dot2
        self.p_dot2 = p_dot2
        self.q_dot2 = q_dot2

        self.x0_ddot = x0_ddot
        self.p_ddot = p_ddot
        self.q_ddot = q_ddot

        # allocate memory
        self.xs_dot1 = numpy.zeros((self.NTS, self.NX, self.P1))
        self.xs_dot2 = numpy.zeros((self.NTS, self.NX, self.P2))
        self.xs_ddot = numpy.zeros((self.NTS, self.NX, self.P1, self.P2))

        self.f_dot1 = numpy.zeros((self.NX, self.P1))
        self.f_dot2 = numpy.zeros((self.NX, self.P2))
        self.f_ddot = numpy.zeros((self.NX, self.P1, self.P2))

        self.u_dot1 = numpy.zeros((self.NU, self.P1))
        self.u_dot2 = numpy.zeros((self.NU, self.P2))
        self.u_ddot = numpy.zeros((self.NU, self.P1, self.P2))


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
        raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError


    def update_u(self, i, t):
        """Update control discretization for step i."""
        self.u[:] = self.q[:]

    def update_u_dot(self, i, t):
        """Update control discretization for step i."""
        self.u[:]        = self.q[:]
        self.u_dot[:, :] = self.q_dot[:, :]

    def update_u_bar(self, i, t):
        """Update control discretization for step i."""
        self.q_bar[:] += self.u_bar[:]
        self.u_bar[:]     = 0.

    def update_u_ddot(self, i, t):
        """Update control discretization for step i."""
        self.u[:]         = self.q[:]
        self.u_dot1[:, :] = self.q_dot1[:, :]
        self.u_dot2[:, :] = self.q_dot2[:, :]
        self.u_ddot[:, :] = self.q_ddot[:, :, :]

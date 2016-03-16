"""Python back-end of indegrator using algopy."""

import os
import sys
import numpy
import algopy


class BackendAlgopy(object):

    """
    Back-end of indegrator utilizing algopy to evaluate derivatives.
    """

    def __init__(self, ffcn):
        """
        Load the provided pure Python back-end.

        Where ffcn is either:

        1)  a callable  of the form ``def ffcn(f, t, x, p, u)``
        2) or a path to a directory containing a file ffcn.py
        """
        if hasattr(ffcn, '__call__'):
            self._ffcn = ffcn
        else:
            self.path = os.path.abspath(ffcn)
            self.dir = os.path.dirname(self.path)

            sys.path.insert(0, self.dir)
            try:
                import ffcn
            except ImportError:
                err_str = "Could not load 'ffcn' from module {}".format(ffcn)
                raise ImportError(err_str)

            self._ffcn = ffcn.ffcn

        self.traced = False

    def trace(self, dims):
        """Get computational graph of right-hand side function."""
        # initializes values
        t = numpy.zeros((1,))
        # utpm_t = algopy.UTPM(numpy.zeros([1, 1, dims['t']]))
        utpm_x = algopy.UTPM(numpy.zeros([1, 1, dims['x']]))
        utpm_p = algopy.UTPM(numpy.zeros([1, 1, dims['p']]))
        utpm_u = algopy.UTPM(numpy.zeros([1, 1, dims['u']]))

        # TRACE ON
        cg = algopy.CGraph()
        cg.trace_on()

        # evaluate function
        # func_t = algopy.Function(utpm_t)
        func_x = algopy.Function(utpm_x)
        func_p = algopy.Function(utpm_p)
        func_u = algopy.Function(utpm_u)

        func_f = algopy.zeros(dims['x'], dtype=func_x)

        # TODO: What about time t?
        # self.ffcn(func_f, func_t, func_x, func_p, func_u)
        self._ffcn(func_f, t, func_x, func_p, func_u)

        # TRACE OFF
        cg.trace_off()

        # cg.independentFunctionList = [func_t, func_x, func_p, func_u]
        cg.independentFunctionList = [func_x, func_p, func_u]
        cg.dependentFunctionList = [func_f]

        # enable evaluation of derivatives now
        self.cg = cg
        self.traced = True

    def ffcn(self, f, t, x, p, u):
        """
        Right-hand side function.

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
        if self.traced is False:
            dims = {'x': x.size, 'p': p.size, 'u': u.size}
            self.trace(dims)

        # initialize with values
        P = 1

        # utpm_t = algopy.UTPM(numpy.zeros([1, P, t.shape[0]]))
        utpm_f = algopy.UTPM(numpy.zeros([1, P, x.shape[0]]))
        utpm_x = algopy.UTPM(numpy.zeros([1, P, x.shape[0]]))
        utpm_p = algopy.UTPM(numpy.zeros([1, P, p.shape[0]]))
        utpm_u = algopy.UTPM(numpy.zeros([1, P, u.shape[0]]))

        # assign directions
        # utpm_t.data[0] = t[numpy.newaxis, :]
        utpm_x.data[0] = x[numpy.newaxis, :]
        utpm_p.data[0] = p[numpy.newaxis, :]
        utpm_u.data[0] = u[numpy.newaxis, :]

        # evaluate function
        # TODO what about time t?
        # self.cg.pushforward([utpm_t, utpm_x, utpm_p, utpm_u])
        self.cg.pushforward([utpm_x, utpm_p, utpm_u])

        # extract derivatives
        utpm_f = self.cg.dependentFunctionList[0]
        f[...] = utpm_f.x.data[0, 0, :]

    def ffcn_dot(self, f, f_dot, t, x, x_dot, p, p_dot, u, u_dot):
        """
        Nominal evaluation and first-order forward derivative of right-hand side function.

        Parameters
        ----------
        t : scalar
            current time for evaluation
        x : array-like (NX)
            current differential states of the system
        x_dot : array-like (NX, P)
            forward directions for derivative evaluation
        p : array-like (NP)
            current parameters of the system
        p_dot : array-like (NP, P)
            forward directions for derivative evaluation
        u : array-like (NU)
            current control input of the system
        u_dot : array-like (NU, P)
            forward directions for derivative evaluation

        Returns
        -------
        f : array-like (NX)
            evaluated right-hand side function
        f_dot : array-like (NX, P)
            directional forward derivative w.r.t. states, controls and parameters
        """
        if self.traced is False:
            dims = {'x': x.size, 'p': p.size, 'u': u.size}
            self.trace(dims)

        # initialize with values
        P = f_dot.shape[1]

        # utpm_t = algopy.UTPM(numpy.zeros([2, P, t.shape[0]]))
        utpm_f = algopy.UTPM(numpy.zeros([2, P, x.shape[0]]))
        utpm_x = algopy.UTPM(numpy.zeros([2, P, x.shape[0]]))
        utpm_p = algopy.UTPM(numpy.zeros([2, P, p.shape[0]]))
        utpm_u = algopy.UTPM(numpy.zeros([2, P, u.shape[0]]))

        # assign directions
        # utpm_t.data[0] = t[numpy.newaxis, :]
        utpm_x.data[0] = x[numpy.newaxis, :]
        utpm_p.data[0] = p[numpy.newaxis, :]
        utpm_u.data[0] = u[numpy.newaxis, :]

        # utpm_t.data[1, :, :] = t_dot.transpose()
        utpm_x.data[1, :, :] = x_dot.transpose()
        utpm_p.data[1, :, :] = p_dot.transpose()
        utpm_u.data[1, :, :] = u_dot.transpose()

        # evaluate function
        # TODO what about time t?
        # self.cg.pushforward([utpm_t, utpm_x, utpm_p, utpm_u])
        self.cg.pushforward([utpm_x, utpm_p, utpm_u])

        # extract derivatives
        utpm_f = self.cg.dependentFunctionList[0]
        f[...] = utpm_f.x.data[0, 0, :]
        f_dot[...] = utpm_f.x.data[1, :, :].transpose()

    def ffcn_bar(self, f, f_bar, t, x, x_bar, p, p_bar, u, u_bar):
        """
        Nominal evaluation and first-order reverse derivative of right-hand side function.

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
        f_bar : array-like (NX, Q)
            adjoint directions of right-hand side f

        Returns
        -------
        f : array-like (NX)
            evaluated right-hand side function
        x_bar : array-like (NX, Q)
            adjoint derivatives w.r.t. x
        p_bar : array-like (NP, P)
            adjoint derivatives w.r.t. x
        u_bar : array-like (NU, Q)
            adjoint derivatives w.r.t. u
        """
        # get computational graph
        if self.traced is False:
            dims = {'x': x.size, 'p': p.size, 'u': u.size}
            self.trace(dims)

        # initialize with values
        P = 1  # forward propagation of nominal values

        # utpm_t = algopy.UTPM(numpy.zeros([2, P, t.shape[0]]))
        utpm_f = algopy.UTPM(numpy.zeros([1, 1, x.shape[0]]))
        utpm_x = algopy.UTPM(numpy.zeros([1, 1, x.shape[0]]))
        utpm_p = algopy.UTPM(numpy.zeros([1, 1, p.shape[0]]))
        utpm_u = algopy.UTPM(numpy.zeros([1, 1, u.shape[0]]))

        # assign directions
        # utpm_t.data[0] = t[numpy.newaxis, :]
        utpm_x.data[0] = x[numpy.newaxis, :]
        utpm_p.data[0] = p[numpy.newaxis, :]
        utpm_u.data[0] = u[numpy.newaxis, :]

        # evaluate function
        self.cg.pushforward([utpm_x, utpm_p, utpm_u])

        # extract derivatives
        utpm_f = self.cg.dependentFunctionList[0]
        f[...] = utpm_f.x.data[0, 0, :]

        # reverse derivative evaluation
        # print f_bar.shape
        # Q = f_bar.shape[1]  # reverse propagation of adjoint direction
        Q = 1
        err_str = 'propagation of only one adjoint direction is implemented!'
        assert Q == 1, err_str

        # initialize with values
        utpm_f_bar = algopy.UTPM(numpy.zeros([1, 1, f.shape[0]]))

        # assign directions
        utpm_f_bar.data[0] = f_bar.transpose()

        # evaluate function
        # TODO what about time t?
        self.cg.pullback([utpm_f_bar])

        # extract derivatives
        [utpm_x, utpm_p, utpm_u] = self.cg.independentFunctionList

        x_bar[...] = utpm_x.xbar.data[0, :, :].transpose()
        p_bar[...] = utpm_p.xbar.data[0, :, :].transpose()
        u_bar[...] = utpm_u.xbar.data[0, :, :].transpose()

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
        p : array-like ()
            current parameters of the system
        p_dot2 : array-like (,)
        p_dot1 : array-like (,)
        p_ddot : array-like (,)
        u : array-like ()
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

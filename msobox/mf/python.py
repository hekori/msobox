"""Pure Python back-end of indegrator."""

import os
import sys
import numpy


class BackendPython(object):

    """
    Pure Python back-end of indegrator.

    Requires the manual implementation of all wanted derivative, i.e:

    zo_forward: ffcn
    fo_forward: ffcn, ffcn_dot
    fo_reverse: ffcn, ffcn_bar
    so_reverse: ffcn, ffcn_dot, ffcn_ddot
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

        1) a class or an object
        2) or a path to a directory containing a python file '*.py'

        providing methods of the form:

        * ffcn     (self, f,        t, x,        p,        u)
        * ffcn_dot (self, f, f_dot, t, x, x_dot, p, p_dot, u, u_dot)
        * ffcn_bar (self, f, f_bar, t, x, x_bar, p, p_bar, u, u_bar)
        * ffcn_ddot(
            f, f_dot2, f_dot1, f_ddot,
            self, t,
            x, x_dot2, x_dot1, x_ddot,
            p, p_dot2, p_dot1, p_ddot,
            u, u_dot2, u_dot1, u_ddot
          )
        """
        # check if object has respective members
        check_members = [hasattr(ffcn, method) for method in self._methods]
        check_members = numpy.asarray(check_members, dtype=bool)

        if check_members.any():
            # avoid rechecking by iterating over check_members
            for i, check in enumerate(check_members):
                if check:
                    setattr(self,
                            self._methods[i],
                            getattr(ffcn, self._methods[i],))
                else:
                    # implemented abstract methods are taken
                    err_str = (
                        'could not register method {method},'
                        .format(method=self._methods[i])
                        + ' dummy is taken instead'
                        )
                    sys.stdout.write(err_str + '\n')
                    pass
        else:
            self.path = os.path.abspath(ffcn)
            self.dir = os.path.dirname(self.path)

            sys.path.insert(0, self.dir)
            try:
                import ffcn
            except ImportError:
                err_str = "Could not load 'ffcn' from module {}".format(ffcn)
                raise ImportError(err_str)

            for method in self._methods:
                # check if module has respective members
                if hasattr(ffcn, method):
                    setattr(self, method, getattr(ffcn, method))
                else:
                    # implemented abstract methods are taken
                    err_str = (
                        'WARNING: could not register "{method}",'
                        .format(method=method)
                        + ' dummy is taken instead'
                    )
                    sys.stdout.write(err_str + '\n')
                    pass

    def ffcn(self, f, t, x, p, u):
        """
        Dummy right-hand side function.

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
        raise NotImplementedError

    def ffcn_dot(self, f, f_dot, t, x, x_dot, p, p_dot, u, u_dot):
        """
        Dummy forward derivative of right-hand side function.

        Parameters
        ----------
        t : scalar
            current time for evaluation
        x : array-like (NX)
            current differential states of the system
        x_dot : array-like (NX, ndirs)
        p : array-like (NP)
            current parameters of the system
        p_dot : array-like (NP, ndirs)
        u : array-like (NU)
            current control input of the system
        u_dot : array-like (NU, ndirs)

        Returns
        -------
        f : array-like (NX)
            evaluated right-hand side function
        f_dot : array-like (,)
            evaluated right-hand side function
        """
        raise NotImplementedError

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
        raise NotImplementedError

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

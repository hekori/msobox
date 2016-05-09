"""Implementation of a reverse communication explicit Euler integration scheme."""

import numpy
from matplotlib import pyplot

class RcExplicitEuler(object):

    def init_zo_forward(self,  NTS, NX, NP, NU):
        """Initialize memory for zero order forward integration"""

        self.NTS = NTS
        self.NX = NX
        self.NP = NP
        self.NU = NU

        self.xpu   = numpy.zeros(NX + NP + NU)
        self.x     = self.xpu[:NX]
        self.p     = self.xpu[NX:NX+NP]
        self.u     = self.xpu[NX+NP:]
        self.f     = numpy.zeros(NX)
        self.ts    = numpy.zeros(NTS)
        self.t     = numpy.zeros(1)

        self.STATE = 'plot'
        self.j     = 0


    def step_zo_forward(self):
        self.t[0] = self.ts[self.j]

        if self.j == self.NTS -1:
            self.STATE = 'finished'

        elif self.STATE == 'plot':
            self.STATE = 'provide_f'

        elif self.STATE == 'provide_f':
            self.STATE = 'advance'

        elif self.STATE == 'advance':
            print self.x
            h = self.ts[self.j+1] - self.ts[self.j]
            self.x[:] = self.x + h*self.f
            self.STATE = 'plot'
            self.j += 1

if __name__ == '__main__':

    def ffcn(f, t, x, p, u):
        f[0] = - p[0]*(x[0] - u[0])

    ind = RcExplicitEuler()
    ind.init_zo_forward(100, 1, 1, 1)
    ind.ts[:] = numpy.linspace(0,1,ind.NTS)
    ind.x[0] = 2.
    ind.p[0] = 1.

    xs = numpy.zeros((ind.NTS, ind.NX))

    while True:

        if ind.STATE == 'provide_f':
            ind.u[0] = 1.
            ffcn(ind.f, ind.t, ind.x, ind.p, ind.u)

        if ind.STATE == 'plot':
            xs[ind.j, :] = ind.x

        elif ind.STATE == 'finished':
            print 'done'
            break

        ind.step_zo_forward()


pyplot.plot(ind.ts, xs, '.k')
pyplot.show()

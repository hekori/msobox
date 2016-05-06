import numpy
from msobox.ind.explicit_euler import ExplicitEuler
from msobox.ind.rk4classic import RK4Classic

from msobox.mf.tapenade import Differentiator
from msobox.mf.fortran import BackendFortran

d = Differentiator('./mf/ffcn.f')
backend_fortran = BackendFortran('./mf/libproblem.so')
rk4 = RK4Classic(backend_fortran)

# =============================================
# first order forward w.r.t. p

ts          = numpy.linspace(0,2,500)
x0          = numpy.ones(5)
p           = numpy.ones(5)
q           = numpy.zeros((4, ts.size, 2))
q[0, :, 0]  = 90.
q[1:, :, 0] = 1.

P           = p.size
x0_dot      = numpy.zeros((x0.size, P))
p_dot       = numpy.zeros((p.size, P))
q_dot       = numpy.zeros(q.shape + (P,))

p_dot[:, :] = numpy.eye(P) 

rk4.fo_forward(ts, x0, x0_dot, p, p_dot, q, q_dot)

from matplotlib import pyplot
pyplot.figure(figsize=(3, 2))
pyplot.title('d/dp x2(t)')
pyplot.plot(rk4.ts, rk4.xs_dot[:,1,:])
pyplot.xlabel('t')
pyplot.tight_layout()
pyplot.savefig('bimolkat_fo_forward_p.png')

# =============================================
# first order forward w.r.t. q

ts          = numpy.linspace(0,2,20)
x0          = numpy.ones(5)
p           = numpy.ones(5)
q           = numpy.zeros((4, ts.size, 1))
q[0, :]     = 90.
q[1:, :]    = 1.

P           = q.size
x0_dot      = numpy.zeros((x0.size, P))
p_dot       = numpy.zeros((p.size, P))
q_dot       = numpy.zeros(q.shape + (P,))

q_dot.reshape((P, P))[:, :] = numpy.eye(P) 

rk4.fo_forward(ts, x0, x0_dot, p, p_dot, q, q_dot)

from matplotlib import pyplot
pyplot.figure(figsize=(3, 2))
pyplot.title('d/dq1 x2(t)')
pyplot.plot(rk4.ts, rk4.xs_dot[:,1,:20])
pyplot.tight_layout()
pyplot.savefig('bimolkat_fo_forward_q.png')


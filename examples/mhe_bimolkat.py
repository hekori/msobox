import os
import sys
import json
import time
import numpy as np

from msobox.mhe.mhe import MHE
from msobox.mhe import RealtimePlot

from msobox.mf.model import Model
from msobox.mf.tapenade import Differentiator
# from msobox.mf.fortran import BackendFortran
from msobox.ind.rk4classic import RK4Classic

# setting print options to print all array elements
np.set_printoptions(threshold=np.nan)

def get_dir_path():
    """return script directory"""
    import inspect, os
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

with open(get_dir_path() + "/fortran/bimolkat/json.txt", "r") as f:
    ds = json.load(f)

# differentiate model functions
# Differentiator(get_dir_path() + "/fortran/bimolkat", ds=ds)
mf = Model(get_dir_path() + "/fortran/bimolkat/gen/libproblem.so", ds)

# brief check to show everything works
assert hasattr(mf, "ffcn")
assert hasattr(mf, "ffcn_dot")
assert hasattr(mf, "ffcn_d_xpu_v")
assert hasattr(mf, "hfcn")
assert hasattr(mf, "hfcn_dot")
assert hasattr(mf, "hfcn_d_xpu_v")

mf.NX = ds['dims']['x']
mf.NY = ds['dims']['f']
mf.NZ = ds['dims']['x'] - ds['dims']['f']
mf.NP = ds['dims']['p']
mf.NU = ds['dims']['u']
mf.NH = ds['dims']['h']

# choose an integrator
ind = RK4Classic(mf)
mhe = MHE(mf, ind, M=20, dt=0.1, major=20)
mhe.plot_data.p_options[:] = True

# parameters
mhe.p[:]      = np.ones(mf.NP)
mhe.pbar[:]   = mhe.p[:]
mhe.p_ref[:]  = mhe.p[:]

# states
mhe.s[:, :]     = np.array([1., 1., 0., 3., 0.0])[np.newaxis, :]
mhe.xbar[:]     = mhe.s[0, :]
mhe.s_ref[:, :] = mhe.s

# arrival costs
mhe.ACSBI[...] = np.eye(mhe.NX + mf.NP)
mhe.ACSBI[:mhe.NX, :mhe.NX] *= 1  # states
mhe.ACSBI[mhe.NX:, mhe.NX:] *= 1  # parameter

# initial weight matrix
mhe.CovPenalty[...] = np.eye(mhe.NP + mhe.NX)
mhe.CovPenalty[:mhe.NX, :mhe.NX] *= 1   # states
mhe.CovPenalty[mhe.NX:, mhe.NX:] *= 1   # parameter

mhe.q[:, 0] =   np.linspace(10, 90, mhe.M)     # temperature val
mhe.q[:, 1] =   np.linspace(0, 1, mhe.M)**3    # Ckat feed val
mhe.q[:, 2] =   np.linspace(0, 1, mhe.M)       # feed A val
mhe.q[:, 3] =   np.linspace(0, 1, mhe.M)       # feed B val

# control function discretization
mhe.simulate_s()
mhe.simulate_measurement()

mhe.plot_data.p_options[:] = True
mhe.plot_data.h_options[:] = True
mhe.plot_data.x_options[:] = True
mhe.plot_data.u_options[:] = True

plotter = RealtimePlot(mhe, show_canvas=True,
                            save2file=False,
                            path='/tmp',
                            fname='out')

# plotter.state_ylim = (-0.1, 2)

mhe.VERBOSE = False
mhe.simulate_measurement(simulate_error=False)


mhe.preparation_phase()
mhe.print_parameter_correlations()

# EVENT LOOP

# mhe.p[0] = 2.5
# mhe.save_values()

t = 0.
for i in range(100):
    print '======== Iteration %4d ========'%i

    # if i % 20 == 0:
    #     mhe.p_ref[1] += 1

    mhe.update_arrival_cost()
    mhe.shift()
    q = np.zeros(mhe.NU)
    # q[...] = 0.
    q[0] = 100 * np.sin(0.8*t)**2
    q[1] = np.sin(t + 0.2)**2
    q[2] = np.sin(t + 0.5)**2
    q[3] = np.sin(1.5*t)**2
    t += mhe.dt
    # q[1:,0] = 1.
    mhe.set_control(q)
    eta, sigma = mhe.simulate_measurement(simulate_error=True)
    mhe.set_measurement(eta, sigma)
    mhe.plot_data.integrate_nodewise();
    st = time.time()
    plotter.draw();  #raw_input("press enter to continue")
    # print 'plotting time=', time.time() - st

    # mhe.ub[:mhe.NX] = 5.5
    # mhe.lb[:mhe.NX] = -5.5

    # mhe.ub[mhe.NX:] = 5
    # mhe.lb[mhe.NX:] = 0.01

    for k in range(3):
        mhe.preparation_phase()
        mhe.optimize()

    # print mhe.xbar
    print mhe.pbar
    print mhe.p
    print mhe.p_ref

    if np.any(mhe.p <= 0.):
        raise Exception("parameter negative")


raw_input("press enter to continue")

# mhe.plot_data.save_data('/tmp/out.txt')

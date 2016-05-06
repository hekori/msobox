import numpy
import vplan

class MHEPlugin(vplan.daesol2.StandardPlugin):
    def __init__(self, mhe):
        super(MHEPlugin, self).__init__()
        self.mhe = mhe
        self.dosave = True

    def set_interval(self, nci):
        self.nci = nci
        self.nis = 0

    def __call__(self, t, t_type, xd, xa, q, p, u,
                 Vxd=None, Vxa=None, Vq=None, Vp=None, Vu=None):

        if self.dosave == False:
            return

        # store time instants
        self.mhe.plot_data.t[self.nci, self.nis] = t

        # store state x_{L+1}
        self.mhe.plot_data.x[self.nci, self.nis, :self.mhe.NY] = xd
        self.mhe.plot_data.x[self.nci, self.nis, self.mhe.NY:] = xa

        if not Vxd == None and not Vxa == None and (Vq == 0.).all():
            # store Xx
            self.mhe.plot_data.Xx[self.nci, self.nis, :self.mhe.NY, :] = Vxd[:self.mhe.NX, 0, :].T
            self.mhe.plot_data.Xx[self.nci, self.nis, self.mhe.NY:, :] = Vxa[:self.mhe.NX, 0, :].T

            # store Xp
            self.mhe.plot_data.Xp[self.nci, self.nis, :self.mhe.NY, :] = Vxd[self.mhe.NX:, 0, :].T
            self.mhe.plot_data.Xp[self.nci, self.nis, self.mhe.NY:, :] = Vxa[self.mhe.NX:, 0, :].T

        self.nis += 1
        self.tabula_rasa()


class PlotData(object):
    """
    store information for plotting and analytics

    Use t[:,0] and x[:,0] to obtain values on the
    multiple shooting grid.

    """

    def __init__(self, mhe, NIS = 20,):
        self.mhe = mhe
        self.NIS = NIS

        # fine grid, NIS*M values on time horizon
        self.t          = numpy.zeros((mhe.M, self.NIS))
        self.x          = numpy.zeros((mhe.M, self.NIS, mhe.NX))
        self.u          = numpy.zeros((mhe.M, self.NIS, mhe.NU))
        self.Xx         = numpy.zeros((mhe.M, self.NIS, mhe.NX, mhe.NX))
        self.Xp         = numpy.zeros((mhe.M, self.NIS, mhe.NX, mhe.NP))

        # iterations
        self.p          = numpy.nan * numpy.ones((mhe.M, mhe.NP))
        self.C          = numpy.nan * numpy.ones((mhe.M, mhe.NX + mhe.NP, mhe.NX + mhe.NP))
        self.JC         = numpy.nan * numpy.ones((mhe.M,) + mhe.JC.shape)


        # # shift stored residual values
        # self.F1_list[:mhe.M-1]   = self.F1_list[1:mhe.M]
        # self.F1_list[mhe.M-1]    = numpy.nan
        # self.F1AC_list[:mhe.M-1] = self.F1AC_list[1:mhe.M]
        # self.F1AC_list[mhe.M-1]  = numpy.nan
        # self.F2_list[:mhe.M-1]   = self.F2_list[1:mhe.M]
        # self.F2_list[mhe.M-1]    = numpy.nan
        # # shift stored parameter values
        # self.p[:mhe.M-1, :] = self.p[1:mhe.M, :]
        # self.p[mhe.M-1, :]  = numpy.nan

        # # shift stored covariance matrices
        # self.C[:mhe.M-1, :, :] = self.C[1:mhe.M, : ,:]
        # self.C[mhe.M-1, :, :]  = numpy.nan

        # # shift stored condensed Jacobians
        # self.JC[:mhe.M-1, :, :] = self.JC[1:mhe.M, : ,:]
        # self.JC[mhe.M-1, :, :]  = numpy.nan


        # self.t_list[:, :]     = numpy.nan
        # self.x_list[:, :, :]  = numpy.nan

        # # model trajectory
        # self.NIS = 20 # number of intermediate steps
        # self.x_list     = numpy.zeros((mhe.M, self.NIS, mhe.NX))
        # self.x_list_pe  = self.x_list[:mhe.M-1]



        # self.t_list     = numpy.nan * numpy.ones((mhe.M, self.NIS))
        # self.t_list_pe  = self.t_list[:mhe.M-1]

        # # to store the time evolution of model match
        # self.F1_list     = numpy.nan * numpy.ones((mhe.M, mhe.M * self.NH))
        # self.F1AC_list      = numpy.nan * numpy.ones((mhe.M, mhe.NX + mhe.NP))
        # self.F2_list     = numpy.nan * numpy.ones((mhe.M, (mhe.M-1) * mhe.NX))

        # # to store the time evolution of the parameters (e.g. for plotting later)
        # self.p       = numpy.nan * numpy.ones((mhe.M, mhe.NP))

        # # to store the time evolution of the covariance matrix (e.g. for plotting later)
        # self.C       = numpy.nan * numpy.ones((mhe.M, mhe.NX + mhe.NP, mhe.NX + mhe.NP))

        # # to store the time evolution of the condensed Jacobian (e.g. for plotting later)
        # self.JC      = numpy.nan * numpy.ones((mhe.M,) + self.JC.shape)


        self.p_label = [''] * self.mhe.NP
        self.x_label = [''] * self.mhe.NX
        self.h_label = [''] * self.mhe.NH
        self.u_label = [''] * self.mhe.NU


        self.p_options = numpy.ones(self.mhe.NP, dtype=bool)
        self.x_options = numpy.ones(self.mhe.NX, dtype=bool)
        self.h_options = numpy.ones(self.mhe.NH, dtype=bool)
        self.u_options = numpy.ones(self.mhe.NU, dtype=bool)

        # # reference label and options
        # self.json['p_label']   =  self.p_label
        # self.json['x_label']   =  self.x_label
        # self.json['h_label']   =  self.h_label
        # self.json['u_label']   =  self.u_label
        # self.json['p_options'] =  self.p_options
        # self.json['x_options'] =  self.x_options
        # self.json['h_options'] =  self.h_options
        # self.json['u_options'] =  self.u_options


    def shift(self):
        mhe = self.mhe
        self.p[:mhe.M-1, :] = self.p[1:mhe.M, :]
        self.p[mhe.M-1, :]  = numpy.nan

        self.C[:mhe.M-1, :, :] = self.C[1:mhe.M, : ,:]
        self.C[mhe.M-1, :, :]  = numpy.nan

        # shift stored condensed Jacobians
        self.JC[:mhe.M-1, :, :] = self.JC[1:mhe.M, : ,:]
        self.JC[mhe.M-1, :, :]  = numpy.nan

    def integrate(self):
        """
        stores trajectory in self.x_list
        """
        # enable save into t_list and x_list
        tmp = self.mhe.integrator.plugin.dosave
        self.mhe.integrator.plugin.dosave = True

        self.mhe._init_ms_node(0)

        # integrate over whole horizon part by part
        for j, node in enumerate(self.mhe.ts[:-1]):
            self.mhe._update_rwh_iwh(j)
            t0, tend = self.mhe.ts[j], self.mhe.ts[j + 1]
            self.mhe.xd[...], self.mhe.xa[...] = self.mhe.integrator.forward(
                numpy.linspace(t0, tend, self.NIS),
                self.mhe.xd, self.mhe.xa, self.mhe.q.ravel(), self.mhe.p)

        self.mhe.integrator.plugin.dosave = tmp

    def integrate_nodewise(self):
        """
        stores trajectory in self.x_list
        """
        #print '#'* 20
        #print 'integrate_nodewise():\n'

        tmp = self.mhe.integrator.plugin.dosave
        self.mhe.integrator.plugin.dosave = True

        # integrate over whole horizon part by part
        for j, node in enumerate(self.mhe.ts[:-1]):
            self.mhe._init_ms_node(j)
            self.mhe._update_rwh_iwh(j)
            t0, tend = self.mhe.ts[j], self.mhe.ts[j + 1]
            self.mhe.xd[...], self.mhe.xa[...] = self.mhe.integrator.forward(
                numpy.linspace(t0, tend, self.NIS),
                self.mhe.xd, self.mhe.xa, self.mhe.q.ravel(), self.mhe.p)

        self.mhe.integrator.plugin.dosave = tmp

    def get_sensitivities(self):
        """
        stores sensitivity in self.Xx and Xp
        """
        mhe = self.mhe
        # enable save into t_list and x_list
        tmp = mhe.integrator.plugin.dosave
        mhe.integrator.plugin.dosave = True

        # integrate over whole horizon part by part
        mhe._init_ms_node(0)

        for j, node in enumerate(mhe.ts[:-1]):
            mhe._update_rwh_iwh(j)
            t0, tend = mhe.ts[j], mhe.ts[j + 1]
            mhe.xd[:], mhe.xa[:], mhe.Vxd[...], mhe.Vxa[...] = \
                 mhe.integrator.forward(
                                numpy.linspace(t0, tend, self.NIS),
                                mhe.xd, mhe.xa, mhe.q.ravel(), mhe.p,
                                mhe.Vxd, mhe.Vxa, mhe.Vq, mhe.Vp)

        mhe.integrator.plugin.dosave = tmp

    def get_sensitivities_nodewise(self):
        """
        stores sensitivity in mhe.Xx and Xp
        """
        mhe = self.mhe
        # enable save into t_list and x_list
        tmp = mhe.integrator.plugin.dosave
        mhe.integrator.plugin.dosave = True

        # integrate over whole horizon part by part
        for j, node in enumerate(mhe.ts[:-1]):
            mhe._init_ms_node(j)
            mhe._update_rwh_iwh(j)

            t0, tend = mhe.ts[j], mhe.ts[j + 1]
            mhe.xd[:], mhe.xa[:], mhe.Vxd[...], mhe.Vxa[...] = \
                 mhe.integrator.forward(
                                numpy.linspace(t0, tend, self.NIS),
                                mhe.xd, mhe.xa, mhe.q.ravel(), mhe.p,
                                mhe.Vxd, mhe.Vxa, mhe.Vq, mhe.Vp)

        mhe.integrator.plugin.dosave = tmp




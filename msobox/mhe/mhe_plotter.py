import os
import numpy
import vplan

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

matplotlib.rc('xtick', labelsize=6)
matplotlib.rc('ytick', labelsize=6)
matplotlib.rc('font', size=10)
matplotlib.rc('font', size=10)
matplotlib.rc('text', usetex=True)

class NoPlot(object):
    def draw(self):
        pass

class RealtimePlot(object):
    """
    Adds live plotting functionality to MHE class
    """

    picture_cnt = 0

    def __init__(self, *argc, **argv):
        return self.init(*argc, **argv)

    def init(self, mhe, show_canvas=True, save2file=False, path=None,
            fname=None):
        """
        setup plotting for MHE instance mhe

        * states
        * parameters
        * model response

        :mhe:   instance of the class MHE

        """

        self.mhe = mhe
        plot_data = mhe.plot_data

        self.figsize = (8.0, 5.0)
        self.dpi     = 300

        self.show_canvas = show_canvas
        self.save2file = save2file
        self.path = path
        self.fname = fname

        if self.save2file:
            if path == None:
                err_str = 'Please add path for pictures'
                raise ValueError(err_str)

            else:
                if not os.path.isdir(self.path):
                    os.makedirs(self.path)

            if self.fname == None:
                self.fname = 'canvas'

        # Display Options
        self.info_flag = False   # shows residual, A,D,E, M Criteria

        self.state_ylim = None
        self.p_ylim     = None
        self.h_ylim     = None
        self.u_ylim     = None

        #criteria          Res   A   D   EV
        self.criteria  = [[   ],[ ],[ ],[  ]]

        # define labels
        self.info_label = ["Residual",
                           "A criterion",
                           "SVD of J_C",
                           "Eigenvalues"]

        # calculate rows and columns of plot
        self.nrows = len( numpy.where(
                          numpy.array([self.info_flag,
                                       self.mhe.plot_data.p_options.any(),
                                       self.mhe.plot_data.x_options.any(),
                                       self.mhe.plot_data.h_options.any(),
                                       self.mhe.plot_data.u_options.any()],
                                       dtype=bool))[0])

        #+ 1 # for the residuals

        # FIXME calculate layout more nicely by KGV
        self.ncols = max(len(numpy.where(self.mhe.plot_data.p_options)[0]),
                         len(numpy.where(self.mhe.plot_data.x_options)[0]),
                         len(numpy.where(self.mhe.plot_data.h_options)[0]),
                         len(numpy.where(self.mhe.plot_data.u_options)[0]),
                         )

        if self.info_flag:
            self.ncols = max(self.ncols, 4)


        # initialize plotter
        self.fig = plt.figure(figsize=self.figsize)

        # use interactive mode for "real time" plotting
        if self.show_canvas:
            plt.interactive(True)

        # build up grid
        self.gs = gridspec.GridSpec(self.nrows, self.ncols)

        row_cnt = 0
        # insert axes into grid
        if self.info_flag:
            self.info_axes = []
            self.info_lines = []
            for i in range(len(self.criteria)):
                ax = self.fig.add_subplot(self.gs[row_cnt, i])
                ax.set_xlabel("time [s]")
                self.info_axes.append(ax)

                if i == 0: # residual
                    line = [ax.plot([], [], lw='1', marker='.', ms=4)[0] for i in range(3)]

                elif i == 1: # A criterion
                    line = [ax.plot([], [], c='k', lw='1', marker='.', ms=4)[0] for i in range(1)]

                elif i == 2: # JC
                    line = [ax.plot([], [], c='k', lw='1', marker='.', ms=4)[0] for i in range(plot_data.JC_list[0].shape[1])]

                elif i == 3: # eigenvalues
                    line = [ax.plot([], [], c='k', lw='1', marker='.', ms=4)[0] for i in range(plot_data.C[0].shape[0])]

                self.info_lines.append(line)

            row_cnt += 1

        if self.mhe.plot_data.p_options.any():
            self.p_axes = []
            self.p_lines = []
            for i,j in enumerate(numpy.where(self.mhe.plot_data.p_options)[0]):
                ax = self.fig.add_subplot(self.gs[row_cnt, i])
                ax.set_ylabel("P: {}".format(plot_data.p_label[j]))
                ax.set_xlabel("time [s]")
                self.p_axes.append(ax)

                plotline, caplines, barlinecols = ax.errorbar([0],[0],yerr=[0],label=plot_data.p_label[i],
                                        fmt='x', c='r')

                self.p_lines.append([plotline, caplines, barlinecols])

            row_cnt += 1

        if self.mhe.plot_data.x_options.any():
            self.x_axes = []
            self.x_lines = []
            for i,j in enumerate(numpy.where(self.mhe.plot_data.x_options)[0]):
                ax = self.fig.add_subplot(self.gs[row_cnt, i])
                ax.set_ylabel("X: {}".format(plot_data.x_label[j]))
                ax.set_xlabel("time [s]")
                self.x_axes.append(ax)


                dot, = ax.plot([], [], lw='1',
                                marker='.', ms=4, ls='', c='r', label=plot_data.x_label[j])

                line, = ax.plot([], [], lw='1',
                                ls='-',  marker='', ms=4, c='r', label=plot_data.x_label[j])

                line_blo, = ax.plot([], [], lw='1',
                                ls='--', c='b', label="")
                line_bup, = ax.plot([], [], lw='1',
                                ls='--', c='b', label="")


                self.x_lines.append([dot, line, line_blo, line_bup])

            row_cnt += 1

        if self.mhe.plot_data.h_options.any():
            self.h_axes = []
            self.h_lines = []
            for i,j in enumerate(numpy.where(self.mhe.plot_data.h_options)[0]):
                ax = self.fig.add_subplot(self.gs[row_cnt, i])
                ax.set_ylabel("H: {}".format(plot_data.h_label[j]))
                ax.set_xlabel("time [s]")
                self.h_axes.append(ax)
                plotline, caplines, barlinecols = ax.errorbar([0],
                                        [0],
                                        yerr=[0],
                                        label=plot_data.h_label[j],
                                        fmt='.', c='r')

                line, = ax.plot([0],[0],
                                        label=plot_data.h_label[j],
                                        marker='x', c='r')


                self.h_lines.append([plotline, caplines, barlinecols, line])

            row_cnt += 1

        if self.mhe.plot_data.u_options.any():
            self.u_axes = []
            self.u_lines = []
            for i,j in enumerate(numpy.where(self.mhe.plot_data.u_options)[0]):
                ax = self.fig.add_subplot(self.gs[row_cnt, i])
                ax.set_ylabel("U: {}".format(plot_data.u_label[j]))
                ax.set_xlabel("time [s]")
                self.u_axes.append(ax)

                line_pe, = ax.plot([], [], lw='1',
                                ls='-',  marker='.', ms=4, c='r', label=plot_data.u_label[j])
                self.u_lines.append(line_pe)
            row_cnt += 1

        # show plot canvas with tght layout
        self.fig.tight_layout()
        if self.show_canvas:
            self.fig.show()
            plt.pause(1e-8)

    def draw(self):
        """update plots"""
        plot_data = self.mhe.plot_data

        if self.show_canvas:
            plt.ion()

        # update info plot
        if self.info_flag:
            for i in range(len(self.criteria)):
                ax = self.info_axes[i]
                line = self.info_lines[i]

                # self.info_axes[i].cla()


                label = None
                vals = numpy.nan * numpy.ones(self.mhe.M)

                if i == 0: # residual
                    vals = numpy.nan * numpy.ones((self.mhe.M,
                                                   3))
                    for ci in range(self.mhe.M):
                        res = numpy.linalg.norm(self.mhe.res_list[ci, :], 2)
                        ac  = numpy.linalg.norm(self.mhe.ac_list[ci, :], 2)

                        vals[ci, 0] = res
                        vals[ci, 1] = ac
                        vals[ci, 2] = res + ac


                    # self.info_axes[i].plot(plot_data.t,
                    #                        vals, label=label,
                    #                        ls='-', marker='.', ms=4)

                    for l, y in zip(line, vals.T):
                        l.set_data(plot_data.t, y)


                elif i == 1: # A criterion
                    for ci in range(self.mhe.M):
                        vals[ci] = numpy.trace(plot_data.C[ci])

                    if numpy.sum(numpy.isnan(vals)) < vals.size: # pyplot raises Exception when all values are nan
                        # self.info_axes[i].semilogy(plot_data.t,
                        #                        vals, label=label,
                        #                        ls='-', marker='.', ms=4)

                        line[0].set_data(plot_data.t, vals)


                elif i == 2: # JC
                    vals = numpy.nan * numpy.ones((self.mhe.M,
                                                   plot_data.JC_list[0].shape[1]))

                    for ci in range(self.mhe.M):
                        if  numpy.isnan(numpy.sum(plot_data.JC_list[ci])):
                            vals[ci, :] = numpy.nan
                        else:
                            vals[ci, :] = numpy.linalg.svd(plot_data.JC_list[ci, :-self.mhe.ACSBI.shape[0], :])[1]


                    if numpy.sum(numpy.isnan(vals)) < vals.size: # pyplot raises Exception when all values are nan
                        # self.info_axes[i].semilogy(plot_data.t,
                        #                    vals, label=label,
                        #                    ls='-', marker='.', ms=4)
                        for l, y in zip(line, vals.T):
                            l.set_data(plot_data.t, y)

                elif i == 3: # eigenvalues
                    vals = numpy.nan * numpy.ones((self.mhe.M,
                                                   plot_data.C[0].shape[0]))
                    for ci in range(self.mhe.M):
                        if  numpy.isnan(numpy.sum(plot_data.C[ci])):
                            vals[ci, :] = numpy.nan
                        else:
                            vals[ci, :] = numpy.linalg.eigh(plot_data.C[ci])[0]

                    if numpy.sum(numpy.isnan(vals)) < vals.size: # pyplot raises Exception when all values are nan
                        # self.info_axes[i].semilogy(plot_data.t,
                        #                        vals, label=label,
                        #                        ls='-', marker='.', ms=4)
                        for l, y in zip(line, vals.T):
                            l.set_data(plot_data.t, y)


                # set ylim
                ax.relim()
                ax.autoscale_view()

                # set xlim
                a = self.mhe.ts[0]-self.mhe.dt
                b = self.mhe.ts[-1]+self.mhe.dt
                self.info_axes[i].set_xlim(a,b)


        stds = numpy.zeros((self.mhe.M, self.mhe.NP))
        for ci in range(self.mhe.M):
            stds[ci,:] = numpy.diag(plot_data.C[ci])[self.mhe.NX:]**0.5

        # update paramter plot
        if self.mhe.plot_data.p_options.any():
            for i,j in enumerate(numpy.where(self.mhe.plot_data.p_options)[0]):

                ax = self.p_axes[i]
                plotline, caplines, barlinecols = self.p_lines[i]

                if self.p_ylim != None:
                    self.p_axes[i].set_ylim(self.p_ylim[i])


                # self.p_axes[i].plot(self.mhe.json['ts'],
                #                     numpy.array(self.mhe.json['p'])[:, j],
                #                     ls='-', marker='.', ms=4, label=plot_data.p_label[i])

                # self.p_axes[i].errorbar(plot_data.t,
                #                         numpy.array(plot_data.p)[:, j],
                #                         yerr=stds[:,j],
                #                         label=plot_data.p_label[i],
                #                         fmt='x', c='r')


                set_errorbar(plotline, caplines, barlinecols,
                             self.mhe.ts,
                             plot_data.p[:, j],
                             stds[:,j]
                             )

                # set ylim
                ax.relim()
                ax.autoscale_view()

                # set xlim
                a = self.mhe.ts[0]-self.mhe.dt
                b = self.mhe.ts[-1]+self.mhe.dt
                self.p_axes[i].set_xlim(a,b)

        # update states and shooting variables
        if self.mhe.plot_data.x_options.any():
            for i,j in enumerate(numpy.where(self.mhe.plot_data.x_options)[0]):
                dot, line, line_blo, line_bup = self.x_lines[i]
                ax = self.x_axes[i]
                # plot multiple shooting variables
                # self.x_axes[i].plot(plot_data.t,
                #                     self.mhe.s_pe[:,j],
                #                     ls='', c='r', marker=".")

                dot.set_xdata(self.mhe.ts)
                dot.set_ydata(self.mhe.s[:,j])

                # plot states between node
                # self.x_axes[i].plot(plot_data.t.ravel(),
                #                     plot_data.x[:,:, j].ravel(),
                #                     ls='-', c='r', label=plot_data.x_label[j])

                line.set_xdata(plot_data.t[:self.mhe.M-1].ravel())
                line.set_ydata(plot_data.x[:self.mhe.M-1,:, j].ravel())


                if hasattr(self.mhe, 'b_lo'):
                    tmp = self.mhe.b_lo[:self.mhe.M, j].copy()
                    tmp[tmp<=-1e08] = numpy.nan
                    ax.plot(plot_data.t,
                                        tmp,
                                        ls='--', c='r', label="")


                if hasattr(self.mhe, 'b_up'):
                    tmp = self.mhe.b_up[:self.mhe.M, j].copy()
                    tmp[tmp>=1e08] = numpy.nan
                    ax.plot(plot_data.t,
                                        tmp,
                                        ls='--', c='r', label="")

                # set xlim
                a = self.mhe.ts[0]-self.mhe.dt
                b = self.mhe.ts[-1]+self.mhe.dt
                ax.set_xlim(a,b)

                # set ylim
                ax.relim()
                ax.autoscale_view()

                if self.state_ylim != None:
                    ax.set_ylim(self.state_ylim[i])


        if self.mhe.plot_data.h_options.any():

            # update measurements and model response
            for i,j in enumerate(numpy.where(self.mhe.plot_data.h_options)[0]):

                ax = self.h_axes[i]
                plotline, caplines, barlinecols, line = self.h_lines[i]

                # parameter estimation horizon
                # plot measurements with error bars
                # ax.errorbar(plot_data.t,
                #                         self.mhe.etas[:,j],
                #                         yerr=self.mhe.sigmas[:,j],
                #                         label=plot_data.h_label[j],
                #                         fmt='.', c='r')

                set_errorbar(plotline, caplines, barlinecols,
                             self.mhe.ts,
                             self.mhe.etas[:,j],
                             self.mhe.sigmas[:,j])

                # plot model response without error bars
                # ax.errorbar(plot_data.t,
                #                         plot_data.h[:,j],
                #                         label=plot_data.h_label[j],
                #                         fmt='x', c='r')
                line.set_data(self.mhe.ts,
                              self.mhe.hs[:,j])


                # set xlim
                a = self.mhe.ts[0]-self.mhe.dt
                b = self.mhe.ts[-1]+self.mhe.dt
                ax.set_xlim(a,b)

                # set ylim
                ax.relim()
                ax.autoscale_view()
                if self.h_ylim != None:
                    ax.set_ylim(self.h_ylim[i])


        # update control functions
        if self.mhe.plot_data.u_options.any():
            for i,j in enumerate(numpy.where(self.mhe.plot_data.u_options)[0]):
                u_list = numpy.nan * numpy.ones((self.mhe.ts.size, 3))
                t_list = numpy.nan * numpy.ones((self.mhe.ts.size, 3))

                for ci in range(self.mhe.ts.size - 1):
                    t0 = self.mhe.ts[ci]
                    t1 = self.mhe.ts[ci+1]
                    q0 = self.mhe.q[j, ci, 0]
                    q1 = self.mhe.q[j, ci, 1]

                    t_list[ci, 0] = t0
                    t_list[ci, 1] = t1

                    u_list[ci, 0] = q0
                    u_list[ci, 1] = q0 + q1 * (t1 - t0)

                # last node
                ci = self.mhe.ts.size - 1
                t0 = self.mhe.ts[ci]
                q0 = self.mhe.q[j, ci, 0]
                t_list[ci, 0] = t0
                u_list[ci, 0] = q0

                ts1 = t_list[:self.mhe.M-1].ravel()
                us1 = u_list[:self.mhe.M-1].ravel()

                line = self.u_lines[i]
                line.set_xdata(ts1)
                line.set_ydata(us1)

                # set xlim
                a = self.mhe.ts[0]-self.mhe.dt
                b = self.mhe.ts[-1]+self.mhe.dt
                self.u_axes[i].set_xlim(a,b)

                # set ylim
                self.u_axes[i].set_ylim(min(us1), max(us1))
                if self.u_ylim != None:
                    self.u_axes[i].set_ylim(self.u_ylim[i])

        if self.show_canvas:
        #     # self.fig.canvas.draw()
            plt.pause(1e-8)

        if self.save2file:
            f_path = os.path.join(self.path,
                    self.fname + '{cnt:04}.png'.format(cnt=self.picture_cnt)
                    )
            self.fig.savefig(f_path, dpi=self.dpi)
            self.picture_cnt += 1

class VplanPlotPlugin(vplan.daesol2.StandardPlugin):
    """"
    Call the plot.f function from vplan
    """

    def __init__(self, db, rwh, iwh):
        super(VplanPlotPlugin, self).__init__()
        self.rwh = rwh
        self.iwh = iwh
        self.db  = db

        self.unit = 10
        filename = "plot/integ.plt.1"
        vplan.fortopen( self.unit, filename)

    def __call__(self, t, t_type, xd, xa, q, p, u,
                 Vxd=None, Vxa=None, Vq=None, Vp=None, Vu=None):
        super(VplanPlotPlugin, self).__call__(\
              t, t_type, xd, xa, q, p, u,
              Vxd, Vxa, Vq, Vp, Vu)
        x = numpy.hstack((xd, xa))
        int_dummy = numpy.zeros(1, dtype=numpy.int32)
        double_dummy = numpy.zeros(1, dtype=float)

        self.db.V.PLOT(0, numpy.array(t), x, p, q,
                          int_dummy, int_dummy, double_dummy,
                          int_dummy, double_dummy,
                          int_dummy, int_dummy, double_dummy,
                          self.rwh, self.iwh)

    def __del__(self):
        vplan.fortclose(self.unit)

def set_errorbar(plotline, caplines, barlinecols, x, y, yerr):
    """
    helper function to update the pyplot error bars in RealtimePlot.draw()
    """

    # Replot the data first
    plotline.set_data(x,y)

    # Find the ending points of the errorbars
    error_positions = (x,y-yerr), (x,y+yerr)

    # Update the caplines
    for i,pos in enumerate(error_positions):
         caplines[i].set_data(pos)

    # Update the error bars
    barlinecols[0].set_segments(zip(zip(x,y-yerr), zip(x,y+yerr)))

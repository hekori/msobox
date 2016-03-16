"""Academic example from Albersmeyer's thesis."""
from numpy import (zeros, exp, cos, sin, power, sqrt)


def ffcn(f, t, x, p, u):
    """Academic example from Albersmeyer diss."""
    # --------------------------------------------------------------------------
    # Independent values
    v_m2 = x[0]
    v_m1 = x[1]
    v_0 = p[0]
    # --------------------------------------------------------------------------
    # Intermediate values
    v_1 = exp(v_m2)
    v_2 = v_m1 + v_0
    v_3 = sqrt(v_2)
    v_4 = v_1 * v_2
    v_5 = sin(v_m1)
    v_6 = v_4 + v_5
    v_7 = v_5 - v_3
    # --------------------------------------------------------------------------
    # Dependent values
    f[0] = v_6
    f[1] = v_7
    # --------------------------------------------------------------------------


def ffcn_dot(f, f_dot, t, x, x_dot, p, p_dot, u, u_dot):
    """AD forward derivative of academic example."""
    # --------------------------------------------------------------------------
    # Independent values
    v_m2_dot = x_dot[0, :]
    v_m1_dot = x_dot[1, :]
    v_0_dot = p_dot[0, :]

    v_m2 = x[0]
    v_m1 = x[1]
    v_0 = p[0]
    # --------------------------------------------------------------------------
    # Intermediate values
    v_1 = exp(v_m2)
    v_1_dot = v_1*v_m2_dot

    v_2_dot = v_m1_dot + v_0_dot
    v_2 = v_m1 + v_0

    v_3 = sqrt(v_2)
    v_3_dot = v_2_dot/(2*v_3)

    v_4_dot = v_1_dot * v_2 + v_1 * v_2_dot
    v_4 = v_1 * v_2

    v_5_dot = cos(v_m1)*v_m1_dot
    v_5 = sin(v_m1)

    v_6_dot = v_4_dot + v_5_dot
    v_6 = v_4 + v_5

    v_7_dot = v_5_dot - v_3_dot
    v_7 = v_5 - v_3
    # --------------------------------------------------------------------------
    # Dependent values
    f_dot[0] = v_6_dot
    f_dot[1] = v_7_dot
    f[0] = v_6
    f[1] = v_7
    # --------------------------------------------------------------------------


def ffcn_bar(f, f_bar, t, x, x_bar, p, p_bar, u, u_bar):
    """AD reverse derivative of academic example."""
    # --------------------------------------------------------------------------
    # Independent values
    v_m2 = x[0]
    v_m1 = x[1]
    v_0 = p[0]
    # --------------------------------------------------------------------------
    # Intermediate values
    v_1 = exp(v_m2)
    v_2 = v_m1 + v_0
    v_3 = sqrt(v_2)
    v_4 = v_1 * v_2
    v_5 = sin(v_m1)
    v_6 = v_4 + v_5
    v_7 = v_5 - v_3
    # --------------------------------------------------------------------------
    # Dependent values
    f[0] = v_6
    f[1] = v_7
    # --------------------------------------------------------------------------
    Q = f_bar.shape[1]
    v_m2_bar = zeros((Q))
    v_m1_bar = zeros((Q))
    v_0_bar = zeros((Q))
    v_1_bar = zeros((Q))
    v_2_bar = zeros((Q))
    v_3_bar = zeros((Q))
    v_4_bar = zeros((Q))
    v_5_bar = zeros((Q))
    v_6_bar = zeros((Q))
    v_7_bar = zeros((Q))
    v_6_bar = f_bar[0, :]
    v_7_bar = f_bar[1, :]
    # --------------------------------------------------------------------------
    v_5_bar += v_7_bar
    v_3_bar += -v_7_bar
    v_4_bar += v_6_bar
    v_5_bar += v_6_bar
    v_m1_bar += v_5_bar*cos(v_m1)
    v_1_bar += v_4_bar*v_2
    v_2_bar += v_4_bar*v_1
    v_2_bar += v_3_bar/(2*v_3)
    v_m1_bar += v_2_bar
    v_0_bar += v_2_bar
    v_m2_bar += v_1_bar*v_1
    # --------------------------------------------------------------------------
    x_bar[0, :] = v_m2_bar
    x_bar[1, :] = v_m1_bar
    p_bar[0, :] = v_0_bar
    # --------------------------------------------------------------------------


def ffcn_ddot(
        f, f_dot2, f_dot1, f_ddot,
        t,
        x, x_dot2, x_dot1, x_ddot,
        p, p_dot2, p_dot1, p_ddot,
        u, u_dot2, u_dot1, u_ddot
        ):
    """AD second order forward derivative of academic example."""
    # --------------------------------------------------------------------------
    # Independent values
    v_m2 = x[0]
    v_m1 = x[1]
    v_0 = p[0]

    v_m2_dot2 = x_dot2[0, :]
    v_m1_dot2 = x_dot2[1, :]
    v_0_dot2 = p_dot2[0, :]

    v_m2_dot1 = x_dot1[0, :]
    v_m1_dot1 = x_dot1[1, :]
    v_0_dot1 = p_dot1[0, :]

    v_m2_ddot = x_ddot[0, :]
    v_m1_ddot = x_ddot[1, :]
    v_0_ddot = p_ddot[0, :]
    # --------------------------------------------------------------------------
    # Intermediate values
    v_1 = exp(v_m2)  # nominal
    v_1_dot2 = v_1*v_m2_dot2  # dot2 derivative

    v_1_dot1 = v_1*v_m2_dot1  # nominal (former dot derivative)
    v_1_ddot = v_1_ddot*v_m2_dot1 + v_1*v_m2_ddot  # ddot derivative
    # --------------------------------------------------------------------------
    v_2 = v_m1 + v_0  # nominal
    v_2_dot2 = v_m1_dot2 + v_0_dot2  # dot2 derivative

    v_2_dot1 = v_m1_dot1 + v_0_dot1  # nominal (former dot derivative)
    v_2_ddot = v_m1_ddot + v_0_ddot  # ddot derivative
    # --------------------------------------------------------------------------
    v_3 = sqrt(v_2)  # nominal
    v_3_dot2 = v_2_dot2/(2*v_3)  # dot2 derivative

    v_3_dot1 = v_2_dot1/(2*v_3)  # nominal (former dot derivative)
    # ddot derivative
    v_3_ddot = v_2_ddot/(2*v_3) - v_2_dot1*v_3_ddot/(4*v_3**(3/2))
    # --------------------------------------------------------------------------
    v_4 = v_1 * v_2  # nominal
    v_4_dot2 = v_1_dot2 * v_2 + v_1 * v_2_dot2  # dot2 derivative

    # nominal (former dot derivative)
    v_4_dot1 = v_1_dot1 * v_2 + v_1 * v_2_dot1
    # ddot derivative
    v_4_ddot = v_1_ddot * v_2 + v_1_dot1 * v_2_ddot \
        + v_1_ddot * v_2_dot1 + v_1 * v_2_ddot
    # --------------------------------------------------------------------------
    v_5 = sin(v_m1)  # nominal
    v_5_dot2 = cos(v_m1)*v_m1_dot2  # dot2 derivative

    v_5_dot1 = cos(v_m1)*v_m1_dot1  # nominal (former dot derivative)
    # ddot derivative
    v_5_ddot = -sin(v_m1)*v_m1_ddot*v_m1_dot1 + cos(v_m1)*v_m1_ddot
    # --------------------------------------------------------------------------
    v_6 = v_4 + v_5  # nominal
    v_6_dot2 = v_4_dot2 + v_5_dot2  # dot2 derivative

    v_6_dot1 = v_4_dot1 + v_5_dot1  # nominal (former dot derivative)
    v_6_ddot = v_4_ddot + v_5_ddot  # ddot derivative
    # --------------------------------------------------------------------------
    v_7 = v_5 - v_3  # nominal
    v_7_dot2 = v_5_dot2 - v_3_dot2  # dot2 derivative

    v_7_dot1 = v_5_dot1 - v_3_dot1  # nominal (former dot derivative)
    v_7_ddot = v_5_ddot - v_3_ddot  # ddot derivative
    # --------------------------------------------------------------------------
    # Dependent values
    f[0] = v_6
    f[1] = v_7
    f_dot2[0] = v_6_dot2
    f_dot2[1] = v_7_dot2

    f_dot1[0] = v_6_dot1
    f_dot1[1] = v_7_dot1
    f_ddot[0] = v_6_ddot
    f_ddot[1] = v_7_ddot
    # --------------------------------------------------------------------------


def ffcn_dot_bar(
        f, f_bar, f_dot, f_dot_bar,
        t,
        x, x_bar, x_dot, x_dot_bar,
        p, p_bar, p_dot, p_dot_bar,
        u, u_bar, u_dot, u_dot_bar
        ):
    """AD second order forward derivative of academic example."""
    # --------------------------------------------------------------------------
    # Independent values
    v_m2_dot = x_dot[0, :]
    v_m1_dot = x_dot[1, :]
    v_0_dot = p_dot[0, :]

    v_m2 = x[0]
    v_m1 = x[1]
    v_0 = p[0]
    # --------------------------------------------------------------------------
    # Intermediate values
    v_1 = exp(v_m2)
    v_1_dot = v_1*v_m2_dot

    v_2_dot = v_m1_dot + v_0_dot
    v_2 = v_m1 + v_0

    v_3 = sqrt(v_2)
    v_3_dot = v_2_dot/(2*v_3)

    v_4_dot = v_1_dot * v_2 + v_1 * v_2_dot
    v_4 = v_1 * v_2

    v_5_dot = cos(v_m1)*v_m1_dot
    v_5 = sin(v_m1)

    v_6_dot = v_4_dot + v_5_dot
    v_6 = v_4 + v_5

    v_7_dot = v_5_dot - v_3_dot
    v_7 = v_5 - v_3
    # --------------------------------------------------------------------------
    # Dependent values
    f_dot[0] = v_6_dot
    f_dot[1] = v_7_dot
    f[0] = v_6
    f[1] = v_7
    # --------------------------------------------------------------------------
    Q = f_bar.shape[1]

    v_m2_dot_bar = zeros((Q))
    v_m2_bar = zeros((Q))

    v_m1_dot_bar = zeros((Q))
    v_m1_bar = zeros((Q))

    v_0_dot_bar = zeros((Q))
    v_0_bar = zeros((Q))

    v_1_dot_bar = zeros((Q))
    v_1_bar = zeros((Q))

    v_2_dot_bar = zeros((Q))
    v_2_bar = zeros((Q))

    v_3_dot_bar = zeros((Q))
    v_3_bar = zeros((Q))

    v_4_dot_bar = zeros((Q))
    v_4_bar = zeros((Q))

    v_5_dot_bar = zeros((Q))
    v_5_bar = zeros((Q))

    v_6_dot_bar = f_dot_bar[0, :]
    v_6_bar = f_bar[0, :]

    v_7_dot_bar = f_dot_bar[1, :]
    v_7_bar = f_bar[1, :]
    # --------------------------------------------------------------------------
    # adjoint derivative of nominal code

    # v_7 = v_5 - v_3
    v_5_bar += v_7_bar
    v_3_bar += -v_7_bar

    # v_6 = v_4 + v_5
    v_4_bar += v_6_bar
    v_5_bar += v_6_bar

    # v_5 = sin(v_m1)
    v_m1_bar += v_5_bar*cos(v_m1)

    # v_4 = v_1 * v_2
    v_1_bar += v_4_bar*v_2
    v_2_bar += v_4_bar*v_1

    # v_3 = sqrt(v_2)
    v_2_bar += v_3_bar/(2*v_3)

    # v_2 = v_m1 + v_0
    v_m1_bar += v_2_bar
    v_0_bar += v_2_bar

    # v_1 = exp(v_m2)
    v_m2_bar += v_1_bar*v_1
    # --------------------------------------------------------------------------
    # adjoint derivative of dot derivative code
    # v_7_dot = v_5_dot - v_3_dot
    v_5_dot_bar += v_7_dot_bar
    v_3_dot_bar += -v_7_dot_bar

    # v_6_dot = v_4_dot + v_5_dot
    v_4_dot_bar += v_6_dot_bar
    v_5_dot_bar += v_6_dot_bar

    # v_5_dot = cos(v_m1)*v_m1_dot
    v_m1_dot_bar += -v_5_dot_bar*sin(v_m1)*v_m1_dot
    v_m1_dot_bar += cos(v_m1)*v_m1_dot_bar

    # v_4_dot = v_1_dot * v_2 + v_1 * v_2_dot
    v_1_dot_bar += v_1_dot_bar * v_2

    # v_2_dot = v_m1_dot + v_0_dot

    # v_3_dot = v_2_dot/(2*v_3)

    # v_1_dot = v_1*v_m2_dot
    v_1_bar += 0 * v1_dot_bar
    v_m2_dot_bar += 0 * v1_dot_bar

    # --------------------------------------------------------------------------
    x_bar[0, :] = v_m2_bar
    x_bar[1, :] = v_m1_bar
    p_bar[0, :] = v_0_bar
    # --------------------------------------------------------------------------

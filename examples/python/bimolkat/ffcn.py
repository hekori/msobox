from numpy import (zeros, exp, power)

def ffcn(f, t, x, p, u):
    # states
    n1 = x[0]
    n2 = x[1]
    n3 = x[2]
    n4 = x[3]
    Ckat = x[4]

    # constants
    na1 = 1.0
    na2 = 1.0
    na4 = 2.0

    # control functions
    Tc = u[0]
    Ckat_feed = u[1]
    a_feed = u[2]
    b_feed = u[3]

    # parameters (nature-given)
    kr1 = p[0] * 1.0e-2
    E = p[1] * 60000.0
    k1 = p[2] * 0.1
    Ekat = p[3] * 40000.0
    lam = p[4] * 0.5

    # molar masses (unit: kg/mol)
    M1 = 0.1362
    M2 = 0.09806
    M3 = M1 + M2
    M4 = 0.236

    Temp = Tc + 273.0
    Rg = 8.314
    T1 = 293.0

    # computation of reaction rates
    mR = n1*M1 + n2*M2 + n3*M3 + n4*M4

    kkat = kr1 * exp(-E/Rg * (1.0/Temp - 1.0/T1)) \
        + k1 * exp(-Ekat/Rg *(1.0/Temp - 1.0/T1)) * Ckat

    r1 = kkat * n1 * n2 / mR

    f[0] = -r1 + a_feed
    f[1] = -r1 + b_feed
    f[2] = r1
    f[3] = 0.0
    f[4] = -lam*Ckat + Ckat_feed


def ffcn_dot(f, f_dot, t, x, x_dot, p, p_dot, u, u_dot):
    # states
    n1 = x[0]
    n2 = x[1]
    n3 = x[2]
    n4 = x[3]
    Ckat = x[4]

    n1_dot = x_dot[0]
    n2_dot = x_dot[1]
    n3_dot = x_dot[2]
    n4_dot = x_dot[3]
    Ckat_dot = x_dot[4]

    # constants
    na1 = 1.0
    na2 = 1.0
    na4 = 2.0

    # control functions
    Tc = u[0]
    Ckat_feed = u[1]
    a_feed = u[2]
    b_feed = u[3]

    Tc_dot = u_dot[0]
    Ckat_feed_dot = u_dot[1]
    a_feed_dot = u_dot[2]
    b_feed_dot = u_dot[3]

    # parameters (nature-given)
    kr1 = p[0] * 1.0e-2
    E = p[1] * 60000.0
    k1 = p[2] * 0.1
    Ekat = p[3] * 40000.0
    lam = p[4] * 0.5

    kr1_dot = p_dot[0] * 1.0e-2
    E_dot = p_dot[1] * 60000.0
    k1_dot = p_dot[2] * 0.1
    Ekat_dot = p_dot[3] * 40000.0
    lam_dot = p_dot[4] * 0.5

    # molar masses (unit: kg/mol)
    M1 = 0.1362
    M2 = 0.09806
    M3 = M1 + M2
    M4 = 0.236

    Temp_dot = Tc_dot  # derivative code
    Temp = Tc + 273.0  # nominal code
    Rg = 8.314
    T1 = 293.0

    # computation of reaction rates
    mR_dot = n1_dot*M1 + n2_dot*M2 + n3_dot*M3 + n4_dot*M4  # derivative coe
    mR = n1*M1 + n2*M2 + n3*M3 + n4*M4  # nominal code

    # derivative code
    kkat_dot = kr1_dot * exp(-(1.0/Temp - 1.0/T1)*E/Rg) \
        + kr1 * exp(-(1.0/Temp - 1.0/T1)*E/Rg)* (-(1.0/Temp - 1.0/T1) * E_dot/Rg + 1.0*E*Temp_dot/(Rg*power(Temp, 2))) \
        + k1_dot * exp(-(1.0/Temp - 1.0/T1)*Ekat/Rg) * Ckat \
        + k1 * exp(-(1.0/Temp - 1.0/T1)*Ekat/Rg) * (-(1.0/Temp - 1.0/T1)*Ekat_dot/Rg + 1.0*Ekat*Temp_dot/(Rg*power(Temp,2))) * Ckat \
        + k1 * exp(-(1.0/Temp - 1.0/T1)*Ekat/Rg) * Ckat_dot
    # nominal code
    kkat = kr1 * exp(-E/Rg * (1.0/Temp - 1.0/T1)) \
        + k1 * exp(-Ekat/Rg * (1.0/Temp - 1.0/T1)) * Ckat

    # derivative code
    r1_dot = kkat_dot*n1*n2/mR \
        + kkat*n1_dot*n2/mR \
        + kkat*n1*n2_dot/mR \
        - kkat*n1*n2*mR_dot/power(mR, 2)
    # nominal code
    r1 = kkat * n1 * n2 / mR

    # derivative code
    f_dot[0] = -r1_dot + a_feed_dot
    f_dot[1] = -r1_dot + b_feed_dot
    f_dot[2] = r1_dot
    f_dot[3] = 0.0
    f_dot[4] = -lam_dot*Ckat - lam*Ckat_dot + Ckat_feed_dot

    # nominal code
    f[0] = -r1 + a_feed
    f[1] = -r1 + b_feed
    f[2] = r1
    f[3] = 0.0
    f[4] = -lam*Ckat + Ckat_feed


def ffcn_bar(f, f_bar, t, x, x_bar, p, p_bar, u, u_bar):
    # states
    n1 = x[0]
    n2 = x[1]
    n3 = x[2]
    n4 = x[3]
    Ckat = x[4]

    # constants
    na1 = 1.0
    na2 = 1.0
    na4 = 2.0

    # control functions
    Tc = u[0]
    Ckat_feed = u[1]
    a_feed = u[2]
    b_feed = u[3]

    # parameters (nature-given)
    kr1 = p[0] * 1.0e-2
    E = p[1] * 60000.0
    k1 = p[2] * 0.1
    Ekat = p[3] * 40000.0
    lam = p[4] * 0.5

    # molar masses (unit: kg/mol)
    M1 = 0.1362
    M2 = 0.09806
    M3 = M1 + M2
    M4 = 0.236

    Temp = Tc + 273.0
    Rg = 8.314
    T1 = 293.0

    # computation of reaction rates
    mR = n1*M1 + n2*M2 + n3*M3 + n4*M4

    kkat = kr1 * exp(-E/Rg * (1.0/Temp - 1.0/T1)) \
        + k1 * exp(-Ekat/Rg * (1.0/Temp - 1.0/T1)) * Ckat

    r1 = kkat * n1 * n2 / mR

    f[0] = -r1 + a_feed
    f[1] = -r1 + b_feed
    f[2] = r1
    f[3] = 0.0
    f[4] = -lam*Ckat + Ckat_feed

    # reverse sweep
    P = 1
    r1_bar = zeros((P,))
    a_feed_bar = zeros((P,))
    b_feed_bar = zeros((P,))
    lam_bar = zeros((P,))
    Ckat_bar = zeros((P,))
    Ckat_feed_bar = zeros((P,))
    n1_bar = zeros((P,))
    n2_bar = zeros((P,))
    n3_bar = zeros((P,))
    n4_bar = zeros((P,))
    kkat_bar = zeros((P,))
    mR_bar = zeros((P,))
    kr1_bar = zeros((P,))
    k1_bar = zeros((P,))
    Tc_bar = zeros((P,))
    E_bar = zeros((P,))
    Ekat_bar = zeros((P,))
    Temp_bar = zeros((P,))

    # f[0] = -r1 + a_feed
    r1_bar += -f_bar[0]
    a_feed_bar += f_bar[0]

    # f[1] = -r1 + b_feed
    r1_bar += -f_bar[1]
    b_feed_bar += f_bar[1]

    # f[2] = r1
    r1_bar += f_bar[2]

    # f[4] = -lam*Ckat + Ckat_feed
    lam_bar += -f_bar[4]*Ckat
    Ckat_bar += -lam*f_bar[4]
    Ckat_feed_bar += f_bar[4]

    # r1 = kkat * n1 * n2 / mR
    kkat_bar += r1_bar * n1 * n2 / mR
    n1_bar += kkat * r1_bar * n2 / mR
    n2_bar += kkat * n1 * r1_bar / mR
    mR_bar += -r1_bar * kkat * n1 * n2 / power(mR, 2)

    # kkat = kr1 * exp(-E/Rg * (1.0/Temp - 1.0/T1)) \
    #      + k1 * exp(-Ekat/Rg * (1.0/Temp - 1.0/T1)) * Ckat
    kr1_bar += kkat_bar * exp(-E/Rg * (1.0/Temp - 1.0/T1))
    E_bar += kr1 * exp(-E/Rg * (1.0/Temp - 1.0/T1)) * (-kkat_bar/Rg * (1.0/Temp - 1.0/T1))

    k1_bar += kkat_bar * exp(-Ekat/Rg * (1.0/Temp - 1.0/T1)) * Ckat
    Ekat_bar += kkat * exp(-Ekat/Rg * (1.0/Temp - 1.0/T1)) * (-kkat_bar/Rg * (1.0/Temp - 1.0/T1)) * Ckat
    Ckat_bar += kkat * exp(-Ekat/Rg * (1.0/Temp - 1.0/T1)) * kkat_bar

    Temp_bar += kr1 * exp(-E/Rg * (1.0/Temp - 1.0/T1)) * (-E/Rg * (-kkat_bar/power(Temp, 2))) \
        + kkat * exp(-Ekat/Rg * (1.0/Temp - 1.0/T1)) * (-Ekat/Rg * (-kkat_bar/power(Temp, 2))) * Ckat

    # mR = n1*M1 + n2*M2 + n3*M3 + n4*M4
    n1_bar += mR_bar*M1
    n2_bar += mR_bar*M2
    n3_bar += mR_bar*M3
    n4_bar += mR_bar*M4

    # Temp = Tc + 273.0
    Tc_bar += Temp_bar

    # assign return values
    x_bar[0] = n1_bar
    x_bar[1] = n2_bar
    x_bar[2] = n3_bar
    x_bar[3] = n4_bar
    x_bar[4] = Ckat_bar

    u_bar[0] = Tc_bar
    u_bar[1] = Ckat_feed_bar
    u_bar[2] = a_feed_bar
    u_bar[3] = b_feed_bar

    p_bar[0] = kr1_bar
    p_bar[1] = E_bar
    p_bar[2] = k1_bar
    p_bar[3] = Ekat_bar
    p_bar[4] = lam_bar


def ffcn_ddot(
    f, f_dot2, f_dot1, f_ddot,
    t,
    x, x_dot2, x_dot1, x_ddot,
    p, p_dot2, p_dot1, p_ddot,
    u, u_dot2, u_dot1, u_ddot
):
    # states
    n1 = x[0]
    n2 = x[1]
    n3 = x[2]
    n4 = x[3]
    Ckat = x[4]

    n1_dot2 = x_dot2[0]
    n2_dot2 = x_dot2[1]
    n3_dot2 = x_dot2[2]
    n4_dot2 = x_dot2[3]
    Ckat_dot2 = x_dot2[4]

    n1_dot1 = x_dot1[0]
    n2_dot1 = x_dot1[1]
    n3_dot1 = x_dot1[2]
    n4_dot1 = x_dot1[3]
    Ckat_dot1 = x_dot1[4]

    n1_ddot = x_ddot[0]
    n2_ddot = x_ddot[1]
    n3_ddot = x_ddot[2]
    n4_ddot = x_ddot[3]
    Ckat_ddot = x_ddot[4]

    # constants
    na1 = 1.0
    na2 = 1.0
    na4 = 2.0

    # control functions
    Tc = u[0]
    Ckat_feed = u[1]
    a_feed = u[2]
    b_feed = u[3]

    Tc_dot2 = u_dot2[0]
    Ckat_feed_dot2 = u_dot2[1]
    a_feed_dot2 = u_dot2[2]
    b_feed_dot2 = u_dot2[3]

    Tc_dot1 = u_dot1[0]
    Ckat_feed_dot1 = u_dot1[1]
    a_feed_dot1 = u_dot1[2]
    b_feed_dot1 = u_dot1[3]

    Tc_ddot = u_ddot[0]
    Ckat_feed_ddot = u_ddot[1]
    a_feed_ddot = u_ddot[2]
    b_feed_ddot = u_ddot[3]

    # parameters (nature-given)
    kr1 = p[0] * 1.0e-2
    E = p[1] * 60000.0
    k1 = p[2] * 0.1
    Ekat = p[3] * 40000.0
    lam = p[4] * 0.5

    kr1_dot2 = p_dot2[0] * 1.0e-2
    E_dot2 = p_dot2[1] * 60000.0
    k1_dot2 = p_dot2[2] * 0.1
    Ekat_dot2 = p_dot2[3] * 40000.0
    lam_dot2 = p_dot2[4] * 0.5

    kr1_dot1 = p_dot1[0] * 1.0e-2
    E_dot1 = p_dot1[1] * 60000.0
    k1_dot1 = p_dot1[2] * 0.1
    Ekat_dot1 = p_dot1[3] * 40000.0
    lam_dot1 = p_dot1[4] * 0.5

    kr1_ddot = p_ddot[0] * 1.0e-2
    E_ddot = p_ddot[1] * 60000.0
    k1_ddot = p_ddot[2] * 0.1
    Ekat_ddot = p_ddot[3] * 40000.0
    lam_ddot = p_ddot[4] * 0.5

    # molar masses (unit: kg/mol)
    M1 = 0.1362
    M2 = 0.09806
    M3 = M1 + M2
    M4 = 0.236

    # TODO how to second order derivative?
    Temp_ddot = Tc_ddot  # derivative code
    Temp_dot2 = Tc_dot2  # derivative code
    Temp_dot1 = Tc_dot1  # derivative code
    Temp = Tc + 273.0  # nominal code
    Rg = 8.314
    T1 = 293.0

    # computation of reaction rates
    mR_ddot = n1_ddot*M1 + n2_ddot*M2 + n3_ddot*M3 + n4_ddot*M4  # derivative code
    mR_dot2 = n1_dot2*M1 + n2_dot2*M2 + n3_dot2*M3 + n4_dot2*M4  # derivative code
    mR_dot1 = n1_dot1*M1 + n2_dot1*M2 + n3_dot1*M3 + n4_dot1*M4  # derivative code
    mR = n1*M1 + n2*M2 + n3*M3 + n4*M4  # nominal code

    # derivative code
    # TODO how to second order derivative?
    kkat_ddot = kr1_ddot * exp(-(1.0/Temp - 1.0/T1)*E/Rg) \
        + kr1 * exp(-(1.0/Temp - 1.0/T1)*E/Rg)* (-(1.0/Temp - 1.0/T1) * E_ddot/Rg + 1.0*E*Temp_ddot/(Rg*power(Temp, 2))) \
        + k1_ddot * exp(-(1.0/Temp - 1.0/T1)*Ekat/Rg) * Ckat \
        + k1 * exp(-(1.0/Temp - 1.0/T1)*Ekat/Rg) * (-(1.0/Temp - 1.0/T1)*Ekat_ddot/Rg + 1.0*Ekat*Temp_ddot/(Rg*power(Temp,2))) * Ckat \
        + k1 * exp(-(1.0/Temp - 1.0/T1)*Ekat/Rg) * Ckat_ddot

    # derivative code
    kkat_dot2 = kr1_dot2 * exp(-(1.0/Temp - 1.0/T1)*E/Rg) \
        + kr1 * exp(-(1.0/Temp - 1.0/T1)*E/Rg)* (-(1.0/Temp - 1.0/T1) * E_dot2/Rg + 1.0*E*Temp_dot2/(Rg*power(Temp, 2))) \
        + k1_dot2 * exp(-(1.0/Temp - 1.0/T1)*Ekat/Rg) * Ckat \
        + k1 * exp(-(1.0/Temp - 1.0/T1)*Ekat/Rg) * (-(1.0/Temp - 1.0/T1)*Ekat_dot2/Rg + 1.0*Ekat*Temp_dot2/(Rg*power(Temp,2))) * Ckat \
        + k1 * exp(-(1.0/Temp - 1.0/T1)*Ekat/Rg) * Ckat_dot2

    # derivative code
    kkat_dot1 = kr1_dot1 * exp(-(1.0/Temp - 1.0/T1)*E/Rg) \
        + kr1 * exp(-(1.0/Temp - 1.0/T1)*E/Rg)* (-(1.0/Temp - 1.0/T1) * E_dot1/Rg + 1.0*E*Temp_dot1/(Rg*power(Temp, 2))) \
        + k1_dot1 * exp(-(1.0/Temp - 1.0/T1)*Ekat/Rg) * Ckat \
        + k1 * exp(-(1.0/Temp - 1.0/T1)*Ekat/Rg) * (-(1.0/Temp - 1.0/T1)*Ekat_dot1/Rg + 1.0*Ekat*Temp_dot1/(Rg*power(Temp,2))) * Ckat \
        + k1 * exp(-(1.0/Temp - 1.0/T1)*Ekat/Rg) * Ckat_dot1
    # nominal code
    kkat = kr1 * exp(-E/Rg * (1.0/Temp - 1.0/T1)) \
        + k1 * exp(-Ekat/Rg * (1.0/Temp - 1.0/T1)) * Ckat

    # derivative code
    # TODO how to second order derivative?
    r1_ddot = kkat_ddot*n1*n2/mR \
        + kkat*n1_ddot*n2/mR \
        + kkat*n1*n2_ddot/mR \
        - kkat*n1*n2*mR_ddot/power(mR, 2)

    # derivative code
    r1_dot2 = kkat_dot2*n1*n2/mR \
        + kkat*n1_dot2*n2/mR \
        + kkat*n1*n2_dot2/mR \
        - kkat*n1*n2*mR_dot2/power(mR, 2)

    # derivative code
    r1_dot1 = kkat_dot1*n1*n2/mR \
        + kkat*n1_dot1*n2/mR \
        + kkat*n1*n2_dot1/mR \
        - kkat*n1*n2*mR_dot1/power(mR, 2)
    # nominal code
    r1 = kkat * n1 * n2 / mR

    # derivative code
    # TODO how to second order derivative?
    f_ddot[0] = -r1_ddot + a_feed_ddot
    f_ddot[1] = -r1_ddot + b_feed_ddot
    f_ddot[2] = r1_ddot
    f_ddot[3] = 0.0
    f_ddot[4] = -lam_ddot*Ckat - lam*Ckat_ddot + Ckat_feed_ddot

    # derivative code
    f_dot2[0] = -r1_dot2 + a_feed_dot2
    f_dot2[1] = -r1_dot2 + b_feed_dot2
    f_dot2[2] = r1_dot2
    f_dot2[3] = 0.0
    f_dot2[4] = -lam_dot2*Ckat - lam*Ckat_dot2 + Ckat_feed_dot2

    # derivative code
    f_dot1[0] = -r1_dot1 + a_feed_dot1
    f_dot1[1] = -r1_dot1 + b_feed_dot1
    f_dot1[2] = r1_dot1
    f_dot1[3] = 0.0
    f_dot1[4] = -lam_dot1*Ckat - lam*Ckat_dot1 + Ckat_feed_dot1

    # nominal code
    f[0] = -r1 + a_feed
    f[1] = -r1 + b_feed
    f[2] = r1
    f[3] = 0.0
    f[4] = -lam*Ckat + Ckat_feed

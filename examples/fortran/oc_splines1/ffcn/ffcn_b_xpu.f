
      SUBROUTINE FFCN_B_XPU(t, x, x_b, f, f_b, p, p_b, u, u_b)
      IMPLICIT NONE
C
C
      DOUBLE PRECISION x(3), f(3), p(1), u(1), t
      DOUBLE PRECISION x_b(3), f_b(3), p_b(1), u_b(1)
C
      f(1) = x(2)
      f(2) = u(1)
      f(3) = u(1)**2
      u_b(1) = u_b(1) + 2*u(1)*f_b(3)
      f_b(3) = 0.D0
      u_b(1) = u_b(1) + f_b(2)
      f_b(2) = 0.D0
      x_b(2) = x_b(2) + f_b(1)
      END

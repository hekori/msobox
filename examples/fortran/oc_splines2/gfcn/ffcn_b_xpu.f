
      SUBROUTINE FFCN_B_XPU(t, x, x_b, f, f_b, p, p_b, u, u_b)
      IMPLICIT NONE
C
C
      DOUBLE PRECISION x(3), f(1), p(1), u(1), t
      DOUBLE PRECISION x_b(3), f_b(1), p_b(1), u_b(1)
C
      f(1) = x(1) - 0.1
      x_b(1) = x_b(1) + f_b(1)
      END

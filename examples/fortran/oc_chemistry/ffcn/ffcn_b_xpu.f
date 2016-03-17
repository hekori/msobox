
      SUBROUTINE FFCN_B_XPU(t, x, x_b, f, f_b, p, p_b, u, u_b)
      IMPLICIT NONE
C
C
      DOUBLE PRECISION x(2), f(2), p(1), u(1), t
      DOUBLE PRECISION x_b(2), f_b(2), p_b(1), u_b(1)
      DOUBLE PRECISION temp_b
C
      f(1) = -(u(1)*x(1)) + u(1)**2*x(2)
      f(2) = u(1)*x(1) - p(1)*u(1)**2*x(2)
      temp_b = -(u(1)**2*f_b(2))
      u_b(1) = u_b(1) + (x(1)-p(1)*x(2)*2*u(1))*f_b(2)
      x_b(1) = x_b(1) + u(1)*f_b(2)
      p_b(1) = p_b(1) + x(2)*temp_b
      f_b(2) = 0.D0
      x_b(2) = x_b(2) + u(1)**2*f_b(1) + p(1)*temp_b
      u_b(1) = u_b(1) + (x(2)*2*u(1)-x(1))*f_b(1)
      x_b(1) = x_b(1) - u(1)*f_b(1)
      END

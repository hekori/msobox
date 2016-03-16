      subroutine ffcn(f, t, x, p, u)
        implicit none
        real*8 x(2), f(2), p(1), u(0), t
        real*8 v_m2, v_m1, v_0
        real*8 v_1, v_2, v_3, v_4, v_5, v_6, v_7
C       ------------------------------------------------------------------------
        ! Independent values
        v_m2 = x(1)
        v_m1 = x(2)
        v_0 = p(1)
C       ------------------------------------------------------------------------
        ! Intermediate values
        v_1 = dexp(v_m2)
        v_2 = v_m1 + v_0
        v_3 = dsqrt(v_2)
        v_4 = v_1 * v_2
        v_5 = dsin(v_m1)
        v_6 = v_4 + v_5
        v_7 = v_5 - v_3
C       ------------------------------------------------------------------------
        ! Dependent values
        f(1) = v_6
        f(2) = v_7
C       ------------------------------------------------------------------------
      end

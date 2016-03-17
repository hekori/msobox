
      SUBROUTINE FFCN_D_XPU_V(t, x, x_d, f, f_d, p, p_d, u, u_d, nbdirs)
      IMPLICIT NONE
C
C
      DOUBLE PRECISION x(3), f(3), p(1), u(1), t
      DOUBLE PRECISION x_d(nbdirs, 3), f_d(nbdirs, 3), p_d(             
     *    nbdirs, 1), u_d(nbdirs, 1)
      INTEGER nd
      INTEGER nbdirs
      INTEGER ii1
      DO nd=1,nbdirs
        DO ii1=1,3
          f_d(nd, ii1) = 0.D0
        ENDDO
        f_d(nd, 1) = x_d(nd, 2)
        f_d(nd, 2) = u_d(nd, 1)
        f_d(nd, 3) = 2*u(1)*u_d(nd, 1)
      ENDDO
C
      f(1) = x(2)
      f(2) = u(1)
      f(3) = u(1)**2
      END

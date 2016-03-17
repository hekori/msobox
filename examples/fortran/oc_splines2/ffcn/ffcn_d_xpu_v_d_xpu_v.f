
      SUBROUTINE FFCN_D_XPU_V_D_XPU_V(t, x, x_d0, x_d, x_d_d, f, f_d0,  
     *                               f_d, f_d_d, p, p_d0, p_d, p_d_d, u 
     *                               , u_d0, u_d, u_d_d, nbdirs,        
     *                         nbdirs0)
      IMPLICIT NONE
C
C
      DOUBLE PRECISION x(3), f(3), p(1), u(1), t
      DOUBLE PRECISION x_d0(nbdirs0, 3), f_d0(nbdirs0, 3), p_d0(        
     *         nbdirs0, 1), u_d0(nbdirs0, 1)
      INTEGER nbdirs
      DOUBLE PRECISION x_d(nbdirs, 3), f_d(nbdirs, 3), p_d(nbdirs, 1),  
     *                u_d(nbdirs, 1)
      DOUBLE PRECISION x_d_d(nbdirs0, nbdirs, 3), f_d_d(nbdirs0,        
     *          nbdirs, 3), p_d_d(nbdirs0, nbdirs, 1), u_d_d(           
     *      nbdirs0, nbdirs, 1)
      INTEGER nd
      INTEGER ii1
      INTEGER nd0
      INTEGER nbdirs0
      INTEGER ii10
      INTEGER ii2
      DO nd0=1,nbdirs0
        DO ii10=1,3
          DO ii2=1,nbdirs
            f_d_d(nd0, ii2, ii10) = 0.D0
          ENDDO
        ENDDO
      ENDDO
      DO nd=1,nbdirs
        DO ii1=1,3
          DO nd0=1,nbdirs0
            f_d_d(nd0, nd, ii1) = 0.D0
          ENDDO
          f_d(nd, ii1) = 0.d0
        ENDDO
        DO nd0=1,nbdirs0
          f_d_d(nd0, nd, 1) = x_d_d(nd0, nd, 2)
          f_d_d(nd0, nd, 2) = u_d_d(nd0, nd, 1)
          f_d_d(nd0, nd, 3) = 2*(u_d0(nd0, 1)*u_d(nd, 1)+u(1)*u_d_d(nd0 
     *     , nd, 1))
        ENDDO
        f_d(nd, 1) = x_d(nd, 2)
        f_d(nd, 2) = u_d(nd, 1)
        f_d(nd, 3) = 2*u(1)*u_d(nd, 1)
      ENDDO
      DO nd0=1,nbdirs0
        DO ii10=1,3
          f_d0(nd0, ii10) = 0.D0
        ENDDO
        f_d0(nd0, 1) = x_d0(nd0, 2)
        f_d0(nd0, 2) = u_d0(nd0, 1)
        f_d0(nd0, 3) = 2*u(1)*u_d0(nd0, 1)
      ENDDO
C
      f(1) = x(2)
      f(2) = u(1)
      f(3) = u(1)**2
      END

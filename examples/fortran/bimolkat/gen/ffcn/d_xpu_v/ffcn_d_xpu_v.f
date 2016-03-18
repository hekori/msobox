
      SUBROUTINE FFCN_D_XPU_V(f, f_d, t, x, x_d, p, p_d, u, u_d, nbdirs)
      IMPLICIT NONE
C
C
      REAL*8 f(5), t, x(5), p(5), u(4)
      REAL*8 f_d(nbdirs, 5), x_d(nbdirs, 5), p_d(nbdirs, 5),        u_d(
     *nbdirs, 4)
C
      REAL*8 n1, n2, n3, n4
      REAL*8 n1_d(nbdirs), n2_d(nbdirs), n3_d(nbdirs), n4_d(       nbdir
     *s)
      REAL*8 na1, na2, na4
      REAL*8 fg, temp, e, rg, t1, tc
      REAL*8 temp_d(nbdirs), e_d(nbdirs), tc_d(nbdirs)
      REAL*8 r1, mr
      REAL*8 r1_d(nbdirs), mr_d(nbdirs)
      REAL*8 kr1, kkat, ckat, ckat_feed, a_feed, b_feed, ekat
      REAL*8 kr1_d(nbdirs), kkat_d(nbdirs), ckat_d(nbdirs),        ckat_
     *feed_d(nbdirs), a_feed_d(nbdirs), b_feed_d(       nbdirs), ekat_d(
     *nbdirs)
      REAL*8 k1, lambda
      REAL*8 k1_d(nbdirs), lambda_d(nbdirs)
      REAL*8 m1, m2, m3, m4
      REAL*8 dm
      REAL*8 arg1
      REAL*8 arg1_d(nbdirs)
      REAL*8 arg2
      REAL*8 arg2_d(nbdirs)
      INTEGER nd
      INTEGER nbdirs
      INTRINSIC DEXP
      INTEGER ii1
C
C       states
C
      n1 = x(1)
      n2 = x(2)
      n3 = x(3)
      n4 = x(4)
      ckat = x(5)
C
C       control functions
C
      tc = u(1)
C
C       parameters (nature-given)
C
      kr1 = p(1)*1.0d-2
      e = p(2)*60000.0d+0
      k1 = p(3)*0.10d+0
      ekat = p(4)*40000.0d0
      lambda = p(5)*0.5d+0
C
C       molar masses (unit: kg/mol)
C
      m1 = 0.1362d+0
      m2 = 0.09806d+0
      m3 = m1 + m2
      m4 = 0.236d+0
C
      temp = tc + 273.0d+0
      rg = 8.314d+0
      t1 = 293.0d+0
C
C       computation of reaction rates
C
      mr = n1*m1 + n2*m2 + n3*m3 + n4*m4
C
      arg1 = -(e/rg*(1.0d+0/temp-1.0d+0/t1))
      arg2 = -(ekat/rg*(1.0d+0/temp-1.0d+0/t1))
      kkat = kr1*DEXP(arg1) + k1*DEXP(arg2)*ckat
      DO nd=1,nbdirs
        n1_d(nd) = x_d(nd, 1)
        n2_d(nd) = x_d(nd, 2)
        n3_d(nd) = x_d(nd, 3)
        n4_d(nd) = x_d(nd, 4)
        ckat_d(nd) = x_d(nd, 5)
        tc_d(nd) = u_d(nd, 1)
        ckat_feed_d(nd) = u_d(nd, 2)
        a_feed_d(nd) = u_d(nd, 3)
        b_feed_d(nd) = u_d(nd, 4)
        kr1_d(nd) = 1.0d-2*p_d(nd, 1)
        e_d(nd) = 60000.0d+0*p_d(nd, 2)
        k1_d(nd) = 0.10d+0*p_d(nd, 3)
        ekat_d(nd) = 40000.0d0*p_d(nd, 4)
        lambda_d(nd) = 0.5d+0*p_d(nd, 5)
        temp_d(nd) = tc_d(nd)
        mr_d(nd) = m1*n1_d(nd) + m2*n2_d(nd) + m3*n3_d(nd) + m4*n4_d(nd)
        arg1_d(nd) = -(e_d(nd)*(1.0d+0/temp-1.0d+0/t1)/rg-e*temp_d(nd)/(
     *    rg*temp**2))
        arg2_d(nd) = -(ekat_d(nd)*(1.0d+0/temp-1.0d+0/t1)/rg-ekat*temp_d
     *    (nd)/(rg*temp**2))
        kkat_d(nd) = kr1_d(nd)*DEXP(arg1) + kr1*arg1_d(nd)*DEXP(arg1) + 
     *    (k1_d(nd)*ckat+k1*ckat_d(nd))*DEXP(arg2) + k1*ckat*arg2_d(nd)*
     *    DEXP(arg2)
        r1_d(nd) = (((kkat_d(nd)*n1+kkat*n1_d(nd))*n2+kkat*n1*n2_d(nd))*
     *    mr-kkat*n1*n2*mr_d(nd))/mr**2
        DO ii1=1,5
          f_d(nd, ii1) = 0.0
        ENDDO
        f_d(nd, 1) = a_feed_d(nd) - r1_d(nd)
        f_d(nd, 2) = b_feed_d(nd) - r1_d(nd)
        f_d(nd, 3) = r1_d(nd)
        f_d(nd, 4) = 0.0
        f_d(nd, 5) = ckat_feed_d(nd) - lambda*ckat_d(nd) - lambda_d(nd)*
     *    ckat
      ENDDO
C
C       constants
C
      na1 = 1.0
      na2 = 1.0
      na4 = 2.0
      ckat_feed = u(2)
      a_feed = u(3)
      b_feed = u(4)
C
      r1 = kkat*n1*n2/mr
C
      f(1) = -r1 + a_feed
      f(2) = -r1 + b_feed
      f(3) = r1
      f(4) = 0.0d0
      f(5) = -(lambda*ckat) + ckat_feed
      END

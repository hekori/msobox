
      SUBROUTINE FFCN_D_XPU_V_D_XPU_V(f, f_d0, f_d, f_d_d, t, x, x_d0,  
     *                               x_d, x_d_d, p, p_d0, p_d, p_d_d, u 
     *                               , u_d0, u_d, u_d_d, nbdirs,        
     *                         nbdirs0)
      IMPLICIT NONE
C
C
      REAL*8 f(5), t, x(5), p(5), u(4)
      REAL*8 f_d0(nbdirs0, 5), x_d0(nbdirs0, 5), p_d0(nbdirs0, 5)       
     *, u_d0(nbdirs0, 4)
      INTEGER nbdirs
      REAL*8 f_d(nbdirs, 5), x_d(nbdirs, 5), p_d(nbdirs, 5), u_d(nbdirs 
     *      , 4)
      REAL*8 f_d_d(nbdirs0, nbdirs, 5), x_d_d(nbdirs0, nbdirs, 5),      
     *  p_d_d(nbdirs0, nbdirs, 5), u_d_d(nbdirs0, nbdirs, 4)
C
      REAL*8 n1, n2, n3, n4
      REAL*8 n1_d0(nbdirs0), n2_d0(nbdirs0), n3_d0(nbdirs0), n4_d0      
     * (nbdirs0)
      REAL*8 n1_d(nbdirs), n2_d(nbdirs), n3_d(nbdirs), n4_d(nbdirs)
      REAL*8 na1, na2, na4
      REAL*8 fg, temp, e, rg, t1, tc
      REAL*8 temp_d0(nbdirs0), e_d0(nbdirs0), tc_d0(nbdirs0)
      REAL*8 temp_d(nbdirs), e_d(nbdirs), tc_d(nbdirs)
      REAL*8 r1, mr
      REAL*8 r1_d0(nbdirs0), mr_d0(nbdirs0)
      REAL*8 r1_d(nbdirs), mr_d(nbdirs)
      REAL*8 kr1, kkat, ckat, ckat_feed, a_feed, b_feed, ekat
      REAL*8 kr1_d0(nbdirs0), kkat_d0(nbdirs0), ckat_d0(nbdirs0),       
     * ckat_feed_d0(nbdirs0), a_feed_d0(nbdirs0), b_feed_d0(       nbdir
     *s0), ekat_d0(nbdirs0)
      REAL*8 kr1_d(nbdirs), kkat_d(nbdirs), ckat_d(nbdirs), ckat_feed_d(
     *       nbdirs), a_feed_d(nbdirs), b_feed_d(nbdirs), ekat_d(nbdirs)
      REAL*8 k1, lambda
      REAL*8 k1_d0(nbdirs0), lambda_d0(nbdirs0)
      REAL*8 k1_d(nbdirs), lambda_d(nbdirs)
      REAL*8 m1, m2, m3, m4
      REAL*8 dm
      REAL*8 arg1
      REAL*8 arg1_d0(nbdirs0)
      REAL*8 arg1_d(nbdirs)
      REAL*8 arg2
      REAL*8 arg2_d0(nbdirs0)
      REAL*8 arg2_d(nbdirs)
      INTEGER nd
      INTRINSIC DEXP
      INTEGER ii1
      INTEGER nd0
      INTEGER nbdirs0
      REAL*8 a_feed_d_d(nbdirs0, nbdirs)
      REAL*8 n4_d_d(nbdirs0, nbdirs)
      REAL*8 kr1_d_d(nbdirs0, nbdirs)
      REAL*8 tc_d_d(nbdirs0, nbdirs)
      REAL*8 b_feed_d_d(nbdirs0, nbdirs)
      REAL*8 k1_d_d(nbdirs0, nbdirs)
      REAL*8 arg2_d_d(nbdirs0, nbdirs)
      REAL*8 n3_d_d(nbdirs0, nbdirs)
      REAL*8 ckat_d_d(nbdirs0, nbdirs)
      REAL*8 arg1_d_d(nbdirs0, nbdirs)
      REAL*8 n2_d_d(nbdirs0, nbdirs)
      REAL*8 kkat_d_d(nbdirs0, nbdirs)
      REAL*8 mr_d_d(nbdirs0, nbdirs)
      REAL*8 ckat_feed_d_d(nbdirs0,        nbdirs)
      REAL*8 r1_d_d(nbdirs0, nbdirs)
      REAL*8 n1_d_d(nbdirs0, nbdirs)
      REAL*8 ekat_d_d(nbdirs0, nbdirs)
      INTEGER ii10
      REAL*8 temp_d_d(nbdirs0, nbdirs)
      REAL*8 e_d_d(nbdirs0, nbdirs)
      INTEGER ii2
      REAL*8 lambda_d_d(nbdirs0, nbdirs)
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
      arg1 = -(e/rg*(1.0d+0/temp-1.0d+0/t1))
      arg2 = -(ekat/rg*(1.0d+0/temp-1.0d+0/t1))
      DO nd0=1,nbdirs0
        n1_d0(nd0) = x_d0(nd0, 1)
        n2_d0(nd0) = x_d0(nd0, 2)
        n3_d0(nd0) = x_d0(nd0, 3)
        n4_d0(nd0) = x_d0(nd0, 4)
        ckat_d0(nd0) = x_d0(nd0, 5)
        tc_d0(nd0) = u_d0(nd0, 1)
        kr1_d0(nd0) = 1.0d-2*p_d0(nd0, 1)
        e_d0(nd0) = 60000.0d+0*p_d0(nd0, 2)
        k1_d0(nd0) = 0.10d+0*p_d0(nd0, 3)
        ekat_d0(nd0) = 40000.0d0*p_d0(nd0, 4)
        lambda_d0(nd0) = 0.5d+0*p_d0(nd0, 5)
        temp_d0(nd0) = tc_d0(nd0)
        mr_d0(nd0) = m1*n1_d0(nd0) + m2*n2_d0(nd0) + m3*n3_d0(nd0) + m4*
     *    n4_d0(nd0)
        arg1_d0(nd0) = -(e_d0(nd0)*(1.0d+0/temp-1.0d+0/t1)/rg-e*temp_d0(
     *    nd0)/(rg*temp**2))
        arg2_d0(nd0) = -(ekat_d0(nd0)*(1.0d+0/temp-1.0d+0/t1)/rg-ekat*  
     *  temp_d0(nd0)/(rg*temp**2))
        kkat_d0(nd0) = kr1_d0(nd0)*DEXP(arg1) + kr1*arg1_d0(nd0)*DEXP(  
     *  arg1) + (k1_d0(nd0)*ckat+k1*ckat_d0(nd0))*DEXP(arg2) + k1*ckat  
     *  *arg2_d0(nd0)*DEXP(arg2)
      ENDDO
C
C       states
C
      n1 = x(1)
      n2 = x(2)
      n3 = x(3)
      n4 = x(4)
      lambda = p(5)*0.5d+0
C
C       computation of reaction rates
C
      mr = n1*m1 + n2*m2 + n3*m3 + n4*m4
      kkat = kr1*DEXP(arg1) + k1*DEXP(arg2)*ckat
      DO nd0=1,nbdirs0
        DO ii10=1,5
          DO ii2=1,nbdirs
            f_d_d(nd0, ii2, ii10) = 0.0
          ENDDO
        ENDDO
      ENDDO
      DO nd0=1,nbdirs0
        DO ii10=1,nbdirs
          n2_d_d(nd0, ii10) = 0.0
        ENDDO
      ENDDO
      DO nd0=1,nbdirs0
        DO ii10=1,nbdirs
          e_d_d(nd0, ii10) = 0.0
        ENDDO
      ENDDO
      DO nd0=1,nbdirs0
        DO ii10=1,nbdirs
          r1_d_d(nd0, ii10) = 0.0
        ENDDO
      ENDDO
      DO nd0=1,nbdirs0
        DO ii10=1,nbdirs
          arg2_d_d(nd0, ii10) = 0.0
        ENDDO
      ENDDO
      DO nd0=1,nbdirs0
        DO ii10=1,nbdirs
          mr_d_d(nd0, ii10) = 0.0
        ENDDO
      ENDDO
      DO nd0=1,nbdirs0
        DO ii10=1,nbdirs
          b_feed_d_d(nd0, ii10) = 0.0
        ENDDO
      ENDDO
      DO nd0=1,nbdirs0
        DO ii10=1,nbdirs
          n4_d_d(nd0, ii10) = 0.0
        ENDDO
      ENDDO
      DO nd0=1,nbdirs0
        DO ii10=1,nbdirs
          temp_d_d(nd0, ii10) = 0.0
        ENDDO
      ENDDO
      DO nd0=1,nbdirs0
        DO ii10=1,nbdirs
          tc_d_d(nd0, ii10) = 0.0
        ENDDO
      ENDDO
      DO nd0=1,nbdirs0
        DO ii10=1,nbdirs
          n1_d_d(nd0, ii10) = 0.0
        ENDDO
      ENDDO
      DO nd0=1,nbdirs0
        DO ii10=1,nbdirs
          ckat_feed_d_d(nd0, ii10) = 0.0
        ENDDO
      ENDDO
      DO nd0=1,nbdirs0
        DO ii10=1,nbdirs
          ckat_d_d(nd0, ii10) = 0.0
        ENDDO
      ENDDO
      DO nd0=1,nbdirs0
        DO ii10=1,nbdirs
          ekat_d_d(nd0, ii10) = 0.0
        ENDDO
      ENDDO
      DO nd0=1,nbdirs0
        DO ii10=1,nbdirs
          arg1_d_d(nd0, ii10) = 0.0
        ENDDO
      ENDDO
      DO nd0=1,nbdirs0
        DO ii10=1,nbdirs
          kr1_d_d(nd0, ii10) = 0.0
        ENDDO
      ENDDO
      DO nd0=1,nbdirs0
        DO ii10=1,nbdirs
          kkat_d_d(nd0, ii10) = 0.0
        ENDDO
      ENDDO
      DO nd0=1,nbdirs0
        DO ii10=1,nbdirs
          n3_d_d(nd0, ii10) = 0.0
        ENDDO
      ENDDO
      DO nd0=1,nbdirs0
        DO ii10=1,nbdirs
          k1_d_d(nd0, ii10) = 0.0
        ENDDO
      ENDDO
      DO nd0=1,nbdirs0
        DO ii10=1,nbdirs
          lambda_d_d(nd0, ii10) = 0.0
        ENDDO
      ENDDO
      DO nd0=1,nbdirs0
        DO ii10=1,nbdirs
          a_feed_d_d(nd0, ii10) = 0.0
        ENDDO
      ENDDO
      DO nd=1,nbdirs
        n1_d(nd) = x_d(nd, 1)
        n2_d(nd) = x_d(nd, 2)
        n3_d(nd) = x_d(nd, 3)
        n4_d(nd) = x_d(nd, 4)
        ckat_d(nd) = x_d(nd, 5)
        tc_d(nd) = u_d(nd, 1)
        kr1_d(nd) = 1.0d-2*p_d(nd, 1)
        e_d(nd) = 60000.0d+0*p_d(nd, 2)
        k1_d(nd) = 0.10d+0*p_d(nd, 3)
        ekat_d(nd) = 40000.0d0*p_d(nd, 4)
        temp_d(nd) = tc_d(nd)
        mr_d(nd) = m1*n1_d(nd) + m2*n2_d(nd) + m3*n3_d(nd) + m4*n4_d(nd)
        arg1_d(nd) = -(e_d(nd)*(1.0d+0/temp-1.0d+0/t1)/rg-e*temp_d(nd)/(
     *    rg*temp**2))
        arg2_d(nd) = -(ekat_d(nd)*(1.0d+0/temp-1.0d+0/t1)/rg-ekat*temp_d
     *    (nd)/(rg*temp**2))
        kkat_d(nd) = kr1_d(nd)*DEXP(arg1) + kr1*arg1_d(nd)*DEXP(arg1) + 
     *    (k1_d(nd)*ckat+k1*ckat_d(nd))*DEXP(arg2) + k1*ckat*arg2_d(nd)*
     *    DEXP(arg2)
        DO nd0=1,nbdirs0
          n1_d_d(nd0, nd) = x_d_d(nd0, nd, 1)
          n2_d_d(nd0, nd) = x_d_d(nd0, nd, 2)
          n3_d_d(nd0, nd) = x_d_d(nd0, nd, 3)
          n4_d_d(nd0, nd) = x_d_d(nd0, nd, 4)
          ckat_d_d(nd0, nd) = x_d_d(nd0, nd, 5)
          tc_d_d(nd0, nd) = u_d_d(nd0, nd, 1)
          ckat_feed_d_d(nd0, nd) = u_d_d(nd0, nd, 2)
          a_feed_d_d(nd0, nd) = u_d_d(nd0, nd, 3)
          b_feed_d_d(nd0, nd) = u_d_d(nd0, nd, 4)
          kr1_d_d(nd0, nd) = 1.0d-2*p_d_d(nd0, nd, 1)
          e_d_d(nd0, nd) = 60000.0d+0*p_d_d(nd0, nd, 2)
          k1_d_d(nd0, nd) = 0.10d+0*p_d_d(nd0, nd, 3)
          ekat_d_d(nd0, nd) = 40000.0d0*p_d_d(nd0, nd, 4)
          lambda_d_d(nd0, nd) = 0.5d+0*p_d_d(nd0, nd, 5)
          temp_d_d(nd0, nd) = tc_d_d(nd0, nd)
          mr_d_d(nd0, nd) = m1*n1_d_d(nd0, nd) + m2*n2_d_d(nd0, nd) + m3
     *      *n3_d_d(nd0, nd) + m4*n4_d_d(nd0, nd)
          arg1_d_d(nd0, nd) = -((e_d_d(nd0, nd)*(1.0d+0/temp-1.0d+0/t1)-
     *      e_d(nd)*temp_d0(nd0)/temp**2)/rg-((e_d0(nd0)*temp_d(nd)+e*  
     *    temp_d_d(nd0, nd))*rg*temp**2-e*temp_d(nd)*rg*2*temp*temp_d0  
     *    (nd0))/(rg*temp**2)**2)
          arg2_d_d(nd0, nd) = -((ekat_d_d(nd0, nd)*(1.0d+0/temp-1.0d+0/ 
     *     t1)-ekat_d(nd)*temp_d0(nd0)/temp**2)/rg-((ekat_d0(nd0)*      
     *temp_d(nd)+ekat*temp_d_d(nd0, nd))*rg*temp**2-ekat*temp_d(nd      
     *)*rg*2*temp*temp_d0(nd0))/(rg*temp**2)**2)
          kkat_d_d(nd0, nd) = kr1_d_d(nd0, nd)*DEXP(arg1) + kr1_d(nd)*  
     *    arg1_d0(nd0)*DEXP(arg1) + (kr1_d0(nd0)*arg1_d(nd)+kr1*      ar
     *g1_d_d(nd0, nd))*DEXP(arg1) + kr1*arg1_d(nd)*arg1_d0(nd0)*      DE
     *XP(arg1) + (k1_d_d(nd0, nd)*ckat+k1_d(nd)*ckat_d0(nd0)+      k1_d0
     *(nd0)*ckat_d(nd)+k1*ckat_d_d(nd0, nd))*DEXP(arg2) + (      k1_d(nd
     *)*ckat+k1*ckat_d(nd))*arg2_d0(nd0)*DEXP(arg2) + ((      k1_d0(nd0)
     **ckat+k1*ckat_d0(nd0))*arg2_d(nd)+k1*ckat*arg2_d_d      (nd0, nd))
     **DEXP(arg2) + k1*ckat*arg2_d(nd)*arg2_d0(nd0)*DEXP      (arg2)
          r1_d_d(nd0, nd) = ((((kkat_d_d(nd0, nd)*n1+kkat_d(nd)*n1_d0(  
     *    nd0)+kkat_d0(nd0)*n1_d(nd)+kkat*n1_d_d(nd0, nd))*n2+(kkat_d(  
     *    nd)*n1+kkat*n1_d(nd))*n2_d0(nd0)+(kkat_d0(nd0)*n1+kkat*n1_d0  
     *    (nd0))*n2_d(nd)+kkat*n1*n2_d_d(nd0, nd))*mr+((kkat_d(nd)*n1+  
     *    kkat*n1_d(nd))*n2+kkat*n1*n2_d(nd))*mr_d0(nd0)-(kkat_d0(nd0)  
     *    *n1+kkat*n1_d0(nd0))*n2*mr_d(nd)-kkat*n1*(n2_d0(nd0)*mr_d(nd  
     *    )+n2*mr_d_d(nd0, nd)))*mr**2-(((kkat_d(nd)*n1+kkat*n1_d(nd))  
     *    *n2+kkat*n1*n2_d(nd))*mr-kkat*n1*n2*mr_d(nd))*2*mr*mr_d0(nd0  
     *    ))/(mr**2)**2
        ENDDO
        ckat_feed_d(nd) = u_d(nd, 2)
        a_feed_d(nd) = u_d(nd, 3)
        b_feed_d(nd) = u_d(nd, 4)
        lambda_d(nd) = 0.5d+0*p_d(nd, 5)
        r1_d(nd) = (((kkat_d(nd)*n1+kkat*n1_d(nd))*n2+kkat*n1*n2_d(nd))*
     *    mr-kkat*n1*n2*mr_d(nd))/mr**2
        DO ii1=1,5
          DO nd0=1,nbdirs0
            f_d_d(nd0, nd, ii1) = 0.0
          ENDDO
          f_d(nd, ii1) = 0.0
        ENDDO
        DO nd0=1,nbdirs0
          f_d_d(nd0, nd, 1) = a_feed_d_d(nd0, nd) - r1_d_d(nd0, nd)
          f_d_d(nd0, nd, 2) = b_feed_d_d(nd0, nd) - r1_d_d(nd0, nd)
          f_d_d(nd0, nd, 3) = r1_d_d(nd0, nd)
          f_d_d(nd0, nd, 4) = 0.0
          f_d_d(nd0, nd, 5) = ckat_feed_d_d(nd0, nd) - lambda_d0(nd0)*  
     *    ckat_d(nd) - lambda*ckat_d_d(nd0, nd) - lambda_d_d(nd0, nd)*  
     *    ckat - lambda_d(nd)*ckat_d0(nd0)
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
      DO nd0=1,nbdirs0
        ckat_feed_d0(nd0) = u_d0(nd0, 2)
        a_feed_d0(nd0) = u_d0(nd0, 3)
        b_feed_d0(nd0) = u_d0(nd0, 4)
        r1_d0(nd0) = (((kkat_d0(nd0)*n1+kkat*n1_d0(nd0))*n2+kkat*n1*    
     *n2_d0(nd0))*mr-kkat*n1*n2*mr_d0(nd0))/mr**2
        DO ii10=1,5
          f_d0(nd0, ii10) = 0.0
        ENDDO
        f_d0(nd0, 1) = a_feed_d0(nd0) - r1_d0(nd0)
        f_d0(nd0, 2) = b_feed_d0(nd0) - r1_d0(nd0)
        f_d0(nd0, 3) = r1_d0(nd0)
        f_d0(nd0, 4) = 0.0
        f_d0(nd0, 5) = ckat_feed_d0(nd0) - lambda*ckat_d0(nd0) -     lam
     *bda_d0(nd0)*ckat
      ENDDO
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

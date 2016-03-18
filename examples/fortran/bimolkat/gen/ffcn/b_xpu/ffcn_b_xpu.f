
      SUBROUTINE FFCN_B_XPU(f, f_b, t, x, x_b, p, p_b, u, u_b)
      IMPLICIT NONE
C
C
      REAL*8 f(5), t, x(5), p(5), u(4)
      REAL*8 f_b(5), x_b(5), p_b(5), u_b(4)
C
      REAL*8 n1, n2, n3, n4
      REAL*8 n1_b, n2_b, n3_b, n4_b
      REAL*8 na1, na2, na4
      REAL*8 fg, temp, e, rg, t1, tc
      REAL*8 temp_b, e_b, tc_b
      REAL*8 r1, mr
      REAL*8 r1_b, mr_b
      REAL*8 kr1, kkat, ckat, ckat_feed, a_feed, b_feed, ekat
      REAL*8 kr1_b, kkat_b, ckat_b, ckat_feed_b, a_feed_b, b_feed_b,    
     *    ekat_b
      REAL*8 k1, lambda
      REAL*8 k1_b, lambda_b
      REAL*8 m1, m2, m3, m4
      REAL*8 dm
      REAL*8 temp3
      REAL*8 temp0_b
      REAL*8 temp2
      REAL*8 temp1
      REAL*8 temp0
      INTRINSIC DEXP
      REAL*8 temp0_b0
      REAL*8 temp1_b
      REAL*8 temp4_b
      REAL*8 temp4
C
C       states
C
      n1 = x(1)
      n2 = x(2)
      n3 = x(3)
      n4 = x(4)
      ckat = x(5)
C
C       constants
C
      na1 = 1.0
      na2 = 1.0
      na4 = 2.0
C
C       control functions
C
      tc = u(1)
      ckat_feed = u(2)
      a_feed = u(3)
      b_feed = u(4)
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
      kkat = kr1*DEXP(-(e/rg*(1.0d+0/temp-1.0d+0/t1))) + k1*DEXP(-(ekat/
     *  rg*(1.0d+0/temp-1.0d+0/t1)))*ckat
C
      r1 = kkat*n1*n2/mr
C
      f(1) = -r1 + a_feed
      f(2) = -r1 + b_feed
      f(3) = r1
      f(4) = 0.0d0
      f(5) = -(lambda*ckat) + ckat_feed
      ckat_feed_b = f_b(5)
      lambda_b = -(ckat*f_b(5))
      ckat_b = -(lambda*f_b(5))
      f_b(5) = 0.0
      f_b(4) = 0.0
      r1_b = f_b(3)
      f_b(3) = 0.0
      b_feed_b = f_b(2)
      r1_b = r1_b - f_b(2)
      f_b(2) = 0.0
      a_feed_b = f_b(1)
      r1_b = r1_b - f_b(1)
      temp4 = n2/mr
      temp4_b = kkat*n1*r1_b/mr
      kkat_b = temp4*n1*r1_b
      mr_b = -(temp4*temp4_b)
      n1_b = m1*mr_b + temp4*kkat*r1_b
      n2_b = m2*mr_b + temp4_b
      temp3 = e/rg
      temp2 = 1.0/temp - 1.0/t1
      temp0_b = kr1*DEXP(-(temp2*temp3))*kkat_b
      temp1 = 1.0/temp - 1.0/t1
      temp0 = -(temp1*ekat/rg)
      temp0_b0 = k1*ckat*DEXP(temp0)*kkat_b
      temp1_b = DEXP(temp0)*kkat_b
      kr1_b = DEXP(-(temp2*temp3))*kkat_b
      temp_b = ekat*temp0_b0/(temp**2*rg) + temp3*temp0_b/temp**2
      e_b = -(temp2*temp0_b/rg)
      ekat_b = -(temp1*temp0_b0/rg)
      k1_b = ckat*temp1_b
      ckat_b = ckat_b + k1*temp1_b
      n3_b = m3*mr_b
      n4_b = m4*mr_b
      tc_b = temp_b
      p_b(5) = p_b(5) + 0.5d+0*lambda_b
      p_b(4) = p_b(4) + 40000.0d0*ekat_b
      p_b(3) = p_b(3) + 0.10d+0*k1_b
      p_b(2) = p_b(2) + 60000.0d+0*e_b
      p_b(1) = p_b(1) + 1.0d-2*kr1_b
      u_b(4) = u_b(4) + b_feed_b
      u_b(3) = u_b(3) + a_feed_b
      u_b(2) = u_b(2) + ckat_feed_b
      u_b(1) = u_b(1) + tc_b
      x_b(5) = x_b(5) + ckat_b
      x_b(4) = x_b(4) + n4_b
      x_b(3) = x_b(3) + n3_b
      x_b(2) = x_b(2) + n2_b
      x_b(1) = x_b(1) + n1_b
      END

C ===============================================================================
C
C 	fortran gfcn for the splines example
C
C 	x1_dot = x2
C 	x2_dot = u
C 	x3_dot = u ** 2
C
C 	x1 < 0.1
C
C ===============================================================================

	subroutine gfcn(f, t, x, p, u)

		implicit none

		double precision x(3), f(1), p(1), u(1), t

		f(1) = x(1) - 0.1

	end

C ===============================================================================
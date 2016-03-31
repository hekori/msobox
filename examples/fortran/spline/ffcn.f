C ===============================================================================
C
C 	fortran ffcn for the splines example
C
C 	x1_dot = x2
C 	x2_dot = u
C 	x3_dot = u ** 2
C
C 	x1 < 0.1
C
C ===============================================================================

	subroutine ffcn(f, t, x, p, u)

		implicit none

		double precision f(3), t, x(3), p(1), u(1)

		f(1) = x(2)
		f(2) = u(1)
		f(3) = u(1) ** 2

	end

C ===============================================================================

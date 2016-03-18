C ===============================================================================
C
C fortran ffcn for the rocket car example
C
C 	x1_dot = - u * x1 + u ** 2 * x2
C 	x2_dot = u * x1 - p * u ** 2 * x2
C
C 	0 < u < 1
C
C ===============================================================================

	subroutine ffcn(f, t, x, p, u)

		implicit none

		double precision x(3), f(3), p(1), u(1), t

		f(1) = x(2) * x(3)
		f(2) = u(1) * x(3)
		f(3) = 0

	end

C ===============================================================================
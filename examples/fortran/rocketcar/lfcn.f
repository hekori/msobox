C-------------------------------------------------------------------------------
C       Model function definitions of the rocket car example.
C-------------------------------------------------------------------------------

      subroutine lfcn(l, t, x, u)
C       ------------------------------------------------------------------------
        ! Lagrange objective of the rocket car example, which is one possible
        ! formulation of the minimal time optimal control problem, in the form
        ! of:

        !     min int_0^T L(t, x, u) dt = int_0^T 1 dt = T
        !      T

        ! NOTE: add an additional state to efficiently solve the objective.
        ! NOTE: this has to be rescaled as well
C       ------------------------------------------------------------------------
        implicit none
        double precision l(1), t(1), x(2), u(1)
C       ------------------------------------------------------------------------
        l(1) = 1.0d0
C       ------------------------------------------------------------------------
      end

C-------------------------------------------------------------------------------

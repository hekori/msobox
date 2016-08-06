C-------------------------------------------------------------------------------
C       Model function definitions of the rocket car example.
C-------------------------------------------------------------------------------

      subroutine mfcn(m, t, x, p)
C       ------------------------------------------------------------------------
        ! Mayer objective of the rocket car example, which is one possible
        ! formulation of the minimal time optimal control problem, in the form
        ! of:

        !     min E(T, x(T), p) = T
        !      T
C       ------------------------------------------------------------------------
        implicit none
        double precision m(1), t(1), x(2), p(1)
C       ------------------------------------------------------------------------
        m(1) = p(1)
C       ------------------------------------------------------------------------
      end

C-------------------------------------------------------------------------------
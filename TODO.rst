==========================================================================
MSOBox - A Tool Box for Mathematical Modeling, Simulation and Optimization
==========================================================================

Features
========

Python conform SNOPT interface
------------------------------

The interplay between Fortran and Python is followed by some problems concerning
arrays starting from 1 instead of zero in Fortran. ::

[x] python conform interface to SNOPT7
[ ] support scipy.sparse coo formats


Finish higher-order derivatives back end
----------------------------------------

Derivative-based optimization methods rely on high-quality derivatives.
Therefore MSOBox provides back ends to state-of-the-art AD tools. ::

    [x] comfortable and fully automatic AD back ends
    [x] easy derivative evaluation and assignment
    [ ] load example models
        [ ] academic
        [ ] bimolkat
        [ ] rocket car
        [ ] bouncing ball
        [ ] simplest walker
    [ ] check first-order derivatives forward and reverse mode
    [ ] add second-order derivatives forward and reverse
    [ ] add the rest of the AD back ends already written
    [ ] port unit tests from old back end code

Goal: Calculate Hessian in mixed first-order forward, second-order reverse mode


IND: RK4Classic and RKFSWT
--------------------------

[ ] implement first-order forward derivatives
[ ] implement first-order reverse derivatives

[ ] implement second-order forward derivatives
[ ] implement second-order reverse derivatives

[ ] finish RKFSWT
  [ ] finish and test interpolation or replace with scipy.rootfind
  [ ] add switch execution
  [ ] add sensitivity forward update
  [ ] add sensitivity reverse update
  [ ] what about second order?
  [ ] run bouncing ball


Implement LP-Newton for NLPs
----------------------------

* use SNOPT interface to finish LP-Newton implementation
* How to get Lagrange multiplier?
* Use analogous RC interface of SNOPT but with internal FSM as in IND

Goal: Implement rocket-car example for LP-Newton solver


Implement SQP for NLPs
----------------------

* re-use Andreas Potschka's code as a starter
* use real Hessian first and no globalization
* Use analogous RC interface of SNOPT but with internal FSM as in IND

Goal: Implement rocket-car example for SQP solver


Add polynomials as convenience functions for control and state discretization
-----------------------------------------------------------------------------

* add functions u(t, q)
* as callable or functor which evaluates different functions vector-wise
* add derivatives u_dot(u, u_dot, t, q, q_dot), u_bar(u, u_bar, t, q, q_bar)



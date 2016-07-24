==========================================================================
MSOBox - A Tool Box for Mathematical Modeling, Simulation and Optimization
==========================================================================

Action Points
=============

Finish higher-order derivatives back end
----------------------------------------

* check first-order derivatives forward and reverse mode
* add second-order derivatives forward and reverse

Goal: Calculate Hessian with mixed first-order forward, second-order reverse


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


Finish implementation of derivatives in RC-IND solvers

Add polynomials as convenience functions for control and state discretization
-----------------------------------------------------------------------------

* add functions u(t, q)
* as callable or functor which evaluates different functions vector-wise
* add derivatives u_dot(u, u_dot, t, q, q_dot), u_bar(u, u_bar, t, q, q_bar)





=============================================================================================
integration of ODEs + derivatives (directional and adjoint)
=============================================================================================


Description
-----------

    INDegrator is a library of IND integration schemes.

    They allow you to evaluate the solution :math:`x(t; x_0, p, q)` of initial value
    problems (IVP) of the form


    .. math::

        \dot x =& f(x, p, q) \\
        x(0) =& x_0

    where :math:`\dot x` denotes the derivative of :math:`x` w.r.t. :math:`t`,
    and additionally

    * first-order derivatives

      .. math::

        \frac{\partial x}{\partial (x_0, p, q)}(t; x_0, p, q) \;, \\

    * and second-order derivatives of the solution

      .. math::

        \frac{\partial^2 x}{\partial (x_0, p, q)^2}(t; x_0, p, q)
      
    in an accurate and efficient way.

    The derivatives w.r.t. :math:`x_0`, :math:`p` and :math:`q` are computed based on the IND and automatic differentiation (AD)
    principles. Both forward and reverse/adjoint mode computations are supported.

Rationale
---------

    * For optimal control (direct approach) one requires accurate derivatives of the solution w.r.t. controls ``q``.

    * For least-squares parameter estimation algorithms one requires derivatives of the solution w.r.t. parameters ``p``.

    * For experimental design optimization one requires accurate second-order derivatives of the solution w.r.t. ``p`` and ``q``


Features
--------

    * Explicit Euler, fixed stepsize

         - first-order forward
         - second-order forward
         - first-order reverse

    * Runge Kutta 4 (RK4), fixed stepsize

         - first-order forward
         - second-order forward
         - first-order reverse

Example
-------

:math:`x(t; x_0, p, q)`

.. literalinclude:: ../examples/ind.py
   :language: python
   :lines: 1-1000



Example 1: zero-order forward
`````````````````````````````

    Compute trajectory :math:`x(t; x_0, p, q)`, i.e., a simple forward evaluation.


    .. literalinclude:: ../examples/ind_zo_forward_bimolkat.py

.. image:: ../examples/ind_zo_forward_bimolkat.png
    :align: center
    :scale: 100



Example 2: first-order forward
``````````````````````````````

    Compute trajectory :math:`\frac{\partial x}{\partial p}(t; x_0, p, q)`
    and :math:`\frac{\partial x}{\partial q}(t; x_0, p, q)`

    .. literalinclude:: ../examples/ind_fo_forward_bimolkat.py

  .. image:: ../examples/ind_fo_forward_bimolkat_p.png
    :align: center
    :scale: 100

  .. image:: ../examples/ind_fo_forward_bimolkat_q.png
    :align: center
    :scale: 100


Example 3: first-order reverse
``````````````````````````````

    Compute the gradients of the state :math:`x(t=2; x_0, p, q)` w.r.t. :math:`x_0, p, q`, i.e.,

    .. math::

        \nabla_{x_0} x(t=2; x_0, p, q) \;, \\
        \nabla_{p} x(t=2; x_0, p, q) \;, \\
        \nabla_{q} x(t=2; x_0, p, q) \;.


    .. literalinclude:: bimolkat_fo_reverse.py

    where one obtains the output::

        gradient of x(t=2; x0, p, q) w.r.t. p  =
        [-0.05553826 -0.26378935 -0.18022685 -0.57068079  0.0912674 ]
        gradient of x(t=2; x0, p, q) w.r.t. q  =
        [[[ -5.88595438e-08   0.00000000e+00]
          [ -5.93303903e-08   0.00000000e+00]
          [ -5.98124990e-08   0.00000000e+00]
          ...,
          [ -1.55170436e-04   0.00000000e+00]
          [ -1.57724384e-04   0.00000000e+00]
          [  0.00000000e+00   0.00000000e+00]]

         [[ -1.91875726e-04   0.00000000e+00]
          [ -1.92256884e-04   0.00000000e+00]
          [ -1.92638781e-04   0.00000000e+00]
          ...,
          [ -1.09278673e-05   0.00000000e+00]
          [ -3.67184216e-06   0.00000000e+00]
          [  0.00000000e+00   0.00000000e+00]]

         [[ -1.88832838e-03   0.00000000e+00]
          [ -1.88832379e-03   0.00000000e+00]
          [ -1.88831907e-03   0.00000000e+00]
          ...,
          [ -4.59399550e-05   0.00000000e+00]
          [ -1.54234950e-05   0.00000000e+00]
          [  0.00000000e+00   0.00000000e+00]]

         [[  2.08733332e-03   0.00000000e+00]
          [  2.08733821e-03   0.00000000e+00]
          [  2.08734323e-03   0.00000000e+00]
          ...,
          [  3.96132167e-03   0.00000000e+00]
          [  3.99233933e-03   0.00000000e+00]
          [  0.00000000e+00   0.00000000e+00]]]
        gradient of x(t=2; x0, p, q) w.r.t. x0 =
        [-0.47113849  0.52078906  0.04958182  0.0499501  -0.04782551]



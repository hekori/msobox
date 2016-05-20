"""
===============================================================================
"""

NOTATION

"""
===============================================================================
"""

OCP:

x		states
NX		number of states
y		differential states
NY 		number of differential states
z		algebraic states
NZ		number of algebraic states
bc		upper and lower bounds on variables
NBC		number of upper and lower bounds for variables
g 		constraint functions
NG		number of constraint functions
bcg 	upper and lower bounds on constraint functions
p		parameters
NP	 	number of parameters
u		control functions
NU		number of control functions

"""
===============================================================================
"""

DISCRETIZATION:

ts 		time steps
NTS		number of time steps
s 		shooting variables
NS 		number of shooting variables (= NTS * NX)
bcs     upper and lower bounds on shooting variables
tsi 	time steps in intervals
NTSI 	number of time steps in intervals
q		control variables
NQI     number of control variables in interval
NQ	 	number of control variables (= NTS * NU * NQI)
c		constraints
NC 		number of constraints (= NTS * NG)

"""
===============================================================================
"""

SENSITIVITY ANALYSIS:

ca 		indices of constraints active
NCA		number of constraints active
mul		non-zero multipliers
NMUL	number of non-zero multipliers

"""
===============================================================================
"""

OCMS DATASTRUCTURE:

x = [discretized control functions, shooting nodes]
  = [u_1(t_0), ..., u_1(t_end), u_2(t_0), ..., u_2(t_end), ..., u_NU(t_0)..., u_NU(t_end), s_1_1, ..., s_1_NX, s_2_1, ..., s_2_NX, ..., s_NTS_NX]
  = [q_0, ..., q_NTS, q_NTS+1, ..., q_2*NTS, ..., q_(NU-1)*NTS+1, ..., q_NU*NTS, s_1_1, ..., s_1_NX, s_2_1, ..., s_2_NX, ..., s_NTS_NX]

dim x = 1 x (NQ + NS)

F = [objective , discretized constraints constraintwise, matching conditions nodewise]
  = [(dim = 1), (dim = NC), (dim = NS - 1)]

dim F = 1 x (1 + NC + NS - 1)

G = [objective_dq , objective_ds,
	 constraints1_dq, constraints1_ds,
	 ...,
	 constraints(NG)_dq, constraints(NG)_ds,
	 matching conditions1_dq, matching conditions1_ds,
	 ...,
	 matching conditions(NS - 1)_dq, matching conditions(NS - 1)_ds]

dim G = 1 x (NQ + NS + NC * (NQ + NS) + (NS - 1) * (NQ + NS))

"""
===============================================================================
"""
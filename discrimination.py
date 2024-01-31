"""
State discrimination example

Discriminating |0> chosen with prob 1/2,
|+> chosen with prob 1/4 and |-> chosen with prob 1/4.

max \sum_x p(x) Tr[rho_x M_x]
s.t. \sum_x M_x = I
    M_x >= 0
"""

import picos as pc

# Initialize the SDP object
sdp = pc.Problem(verbosity = 0)

# Defining the constants in the problem
rho0 = pc.Constant([[1,0],[0,0]])
rho1 = pc.Constant([[1/2,1/2],[1/2,1/2]])
rho2 = pc.Constant([[1/2,-1/2],[-1/2,1/2]])

# Defining the variables (3 matrices)
M0 = pc.HermitianVariable("M0", (2, 2))
M1 = pc.HermitianVariable("M1", (2, 2))
M2 = pc.HermitianVariable("M2", (2, 2))

# Defining the POVM constraints
sdp.add_constraint( M0 + M1 + M2 == [[1,0],[0,1]] )
sdp.add_constraint( M0 >> 0 )
sdp.add_constraint( M1 >> 0 )
sdp.add_constraint( M2 >> 0 )

# Defining the objectve
obj = pc.trace(0.5 * M0 * rho0 + 0.25 * M1 * rho1 + 0.25 * M2 * rho2)
sdp.set_objective("max", obj)

sdp.solve(solver='mosek')

print("Opimal distinguishing probability: ", sdp.value)
print("Optimal measurements:")
print(M0.value)
print(M1.value)
print(M2.value)

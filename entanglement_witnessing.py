"""
Entanglement witnessing example

Here we'll verify that states are entangled. We'll
focus on the family of states
    cos(t) |00> + sin(t) |11>
for t \in (0, pi/4].

We use the primal SDP

max lambda
s.t. rho^{T_B} - lambda I >= 0

(where rho^{T_B} is the partial transpose on system B).

We'll also use the dual SDP

min Tr[W rho]
s.t. Tr[W]=1
     W^{T_B} >= 0

which will help us derive our entanglement witness
(which gives an experimental procedure to detect the entanglement).
"""

import picos as pc
from math import pi, cos, sin
import numpy as np

# Defining the state we care about
t = pi/8
rho = pc.Constant([[cos(t)**2,0,0,cos(t)*sin(t)],
                [0,0,0,0],
                [0,0,0,0],
                [cos(t)*sin(t),0,0,sin(t)**2]])

"""
Let's start with the primal SDP
"""
sdp_p = pc.Problem(verbosity = 0)

# Primal just has a single real variable
l = pc.RealVariable("l", 1)

# Defining the constraint
sdp_p.add_constraint( pc.partial_transpose(rho, 1) - l * np.eye(4) >> 0 )

# Defining the objectve
sdp_p.set_objective("max", l)

# Solve the sdp
sdp_p.solve()
# Also compare with just directly computing eigenvalues
lmin =  min(np.linalg.eigvals( pc.partial_transpose(rho, 1)))

print("Minimal eigenvalue (SDP)", sdp_p.value)
print("Minimal eigenvalue (eigvals)", lmin)



"""
Now we formulate the dual problem and solve that to extract a witness
"""

sdp_d = pc.Problem(verbosity=0)

# Define dual variable
W = pc.HermitianVariable("W", [4,4])

# Add the constraits on W
sdp_d.add_constraint( pc.trace(W) == 1 )
sdp_d.add_constraint( pc.partial_transpose(W, 1) >> 0 )

# Define the objective
sdp_d.set_objective("min", pc.trace( rho * W ))

# Now we solve
sdp_d.solve()

# Check solution and print out witness
print("Dual value", sdp_d.value)
print("Entanglement Witness:")
print(W.value)

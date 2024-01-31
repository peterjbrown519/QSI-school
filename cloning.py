"""
Here we implement the SDP to find the optimal cloner for a
given set of states. If the states are |psi_x> and they are sent with
probability p(x) then we need to solve

max \sum_x p(x) <psi_x'|<psi_x|<psi_x| C_{A_1A_2A_3} |psi_x'>|psi_x>|psi_x>
s.t. Tr_{A_2A_3}[ C_{A_1A_2A_3}] = I_{A_1}
     C_{A_1A_2A_3} >= 0

where |psi_x'> denotes the complex conjugate of |psi_x>.

In this example we are interested in the states |0>|1>|+>|->
each sent with probability 1/4.
"""

import picos as pc
import numpy as np

# Defining the states in the problem
rho0 = pc.Constant( [[1,0],[0,0]] )
rho1 = pc.Constant( [[0,0],[0,1]] )
rho2 = pc.Constant( [[1/2,1/2],[1/2,1/2]] )
rho3 = pc.Constant( [[1/2,-1/2],[-1/2,1/2]] )

# Defining the SDP object
sdp = pc.Problem(verbosity = 0)

# Defining the Choi matrix
C = pc.HermitianVariable("C", [8,8])

# Defining the constraints
sdp.add_constraint( pc.partial_trace(C, 1, [2,4]) == np.eye(2) )
sdp.add_constraint( C >> 0 )

# Defining the objectve (I converted into slightly simpler form). Note that there
# are no imaginary components in state so don't care about complex conjugate.
obj = 0.25 * (pc.trace( (rho0 @ rho0 @ rho0 ) * C) + \
             pc.trace( (rho1 @ rho1 @ rho1 ) * C) + \
             pc.trace( (rho2 @ rho2 @ rho2 ) * C) + \
             pc.trace( (rho3 @ rho3 @ rho3 ) * C))
sdp.set_objective("max", obj)

# Solve the sdp
sdp.solve()

print("Optimal average cloning fidelity:", sdp.value)
print("Optimal channel (choi form)")
print(C)

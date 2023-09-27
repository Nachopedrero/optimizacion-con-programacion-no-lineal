"""
-Nuestras variables de decisión son los presupuestos (positivos) de cada canal

-Nuestra restricción es que la suma de todos los presupuestos no debe exceder 
el presupuesto total

-Nuestro objetivo es maximizar la rentabilidad total, que es la suma de las 
rentabilidades de cada canal
"""
import cvxpy as cp

# Variables
google   = cp.Variable(pos=True)
facebook = cp.Variable(pos=True)
twitter  = cp.Variable(pos=True)

# Constraint
constraint = [google + facebook + twitter <= TOTAL_BUDGET]

# Objective
obj = cp.Maximize(alphas[0] + betas[0] * cp.log(google)
                + alphas[1] + betas[1] * cp.log(facebook)
                + alphas[2] + betas[2] * cp.log(twitter))

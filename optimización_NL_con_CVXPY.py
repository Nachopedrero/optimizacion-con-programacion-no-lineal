"""
-Nuestras variables de decisión son los presupuestos (positivos) de cada canal

-Nuestra restricción es que la suma de todos los presupuestos no debe exceder 
el presupuesto total

-Nuestro objetivo es maximizar la rentabilidad total, que es la suma de las 
rentabilidades de cada canal
"""
import cvxpy as cp

def main():
    try:
        TOTAL_BUDGET = float(input("Enter the total budget: "))
        TOTAL_BUDGET = round(TOTAL_BUDGET,2)
    except ValueError:
        print("Please enter a number")
        exit(1)
    alphas = [-9453.72,-8312.84,-7371.33]
    betas  = [8256.21,7764.20,7953.36]


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
    
    # Solve
    prob = cp.Problem(obj, constraint)
    prob.solve(solver='ECOS', verbose=False)

    # Print solution
    print('='*59 + '\n' + ' '*24 + 'Solution' + ' '*24 + '\n' + '='*59)
    print(f'Status = {prob.status}')
    print(f'Returns = ${round(prob.value):,}\n')
    print('Marketing allocation:')
    print(f' - Google Ads   = ${round(google.value):,}')
    print(f' - Facebook Ads = ${round(facebook.value):,}')
    print(f' - Twitter Ads  = ${round(twitter.value):,}')

"""
-Nuestras variables de decisión son los presupuestos (positivos) de cada canal

-Nuestra restricción es que la suma de todos los presupuestos no debe exceder 
el presupuesto total

-Nuestro objetivo es maximizar la rentabilidad total, que es la suma de las 
rentabilidades de cada canal
"""
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

from optimización_NL_sin_CVXPY import greedy_optimization

def main():
    TOTAL_BUDGET = 1000000
    alphas = [-9453.72,-8312.84,-7371.33]
    betas  = [8256.21,7764.20,7953.36]

    # Linearly spaced numbers
    x = np.linspace(1, TOTAL_BUDGET, TOTAL_BUDGET)

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


    # Plot the functions and the results
    fig = plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(x, alphas[0] + betas[0] * np.log(x), color='red', label='Google Ads')
    plt.plot(x, alphas[1] + betas[1] * np.log(x), color='blue', label='Facebook Ads')
    plt.plot(x, alphas[2] + betas[2] * np.log(x), color='green', label='Twitter Ads')

    # Plot optimal points
    plt.scatter([google.value, facebook.value, twitter.value],
                [alphas[0] + betas[0] * np.log(google.value),
                alphas[1] + betas[1] * np.log(facebook.value),
                alphas[2] + betas[2] * np.log(twitter.value)],
                marker="+", color='black', zorder=10)

    plt.xlabel('Budget ($)')
    plt.ylabel('Returns ($)')
    plt.legend()
    plt.show()
    # List to store the best objective value for each number of iterations
    best_obj_list = []

    # Range of number of iterations to test
    num_iterations_range = np.logspace(0, 6, 20).astype(int)

    # Run the greedy algorithm for each number of iterations and store the best objective value
    for num_iterations in num_iterations_range:
        _, best_obj = greedy_optimization(TOTAL_BUDGET, alphas, betas, num_iterations)
        best_obj_list.append(best_obj)

    # Print solution
    print('='*59 + '\n' + ' '*24 + 'Solution' + ' '*24 + '\n' + '='*59)
    print(f'Status = {prob.status}')
    print(f'Returns = ${round(prob.value):,}\n')
    print('Marketing allocation:')
    print(f' - Google Ads   = ${round(google.value):,}')
    print(f' - Facebook Ads = ${round(facebook.value):,}')
    print(f' - Twitter Ads  = ${round(twitter.value):,}')

if __name__ == '__main__':
    main()
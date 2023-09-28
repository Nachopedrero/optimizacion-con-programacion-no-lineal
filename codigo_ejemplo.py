import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

def greedy_optimization(TOTAL_BUDGET, alphas, betas, num_iterations=1_000):

    # Initialize the budget allocation and the best objective value
    google_budget = facebook_budget = twitter_budget = TOTAL_BUDGET / 3
    obj = alphas[0] + betas[0] * np.log(google_budget) + alphas[1] + betas[1] * np.log(facebook_budget) + alphas[2] + betas[2] * np.log(twitter_budget)

    for _ in range(num_iterations):

        # Generate a new random allocation
        random_allocation = np.random.dirichlet(np.ones(3)) * TOTAL_BUDGET
        google_budget_new, facebook_budget_new, twitter_budget_new = random_allocation

        # Calculate the new objective value
        new_obj = alphas[0] + betas[0] * np.log(google_budget_new) + alphas[1] + betas[1] * np.log(facebook_budget_new) + alphas[2] + betas[2] * np.log(twitter_budget_new)

        # If the new allocation improves the objective value, keep it
        if new_obj > obj:
            google_budget, facebook_budget, twitter_budget = google_budget_new, facebook_budget_new, twitter_budget_new
            obj = new_obj

    # Return the best allocation and the corresponding objective value
    return (google_budget, facebook_budget, twitter_budget), obj

"""
Somos una empresa que ha diseñado tres productos muy innovadores en el sector tecnológico.
Por temas de marketing, sabemos que para lograr la mayor rentabilidad para la empresa, 
lo mejor es que el producto que vaya a tener más impacto sea el último en ser lanzado.
Por lo tanto necesitamos saber cual de los siguientes productos debe ser lanzado primero, 
cual segundo y cual tercero (sabiendo tambien que hay que hacer una inversión inicial
para lanzaro):
- Reloj inteligente con asistente personal con IA (WATCH-IA)
- Teléfono móvil con funciones de ordenador (PHONE-PC)
- Dron con cámara de alta resolución (DRONE-CAM)

Sabemos que el interes del público está directamente relacionado con una fórmula

    rentabilidad = alpha + beta * log(presupuesto)

Las alphas son las siguientes :
(-475.25) - WATCH
(-1234.60) - PHONE
(-799.99) - DRONE

Las betas son las siguientes :
(876.23) - WATCH
(1621.78) - PHONE
(180.12) - DRONE

El presupuesto total para el lanzamiento de los tres productos es de 100.000 de dolares.

La primera gráfic representa las tres funciones de rentabilidad de cada producto.

La segunda gráfica representa que el algoritmo greedy alcanza la solución más óptima 
que puede alcanzar la librería cvxpy en menos de 1000 iteraciones.
"""

def main():
    TOTAL_BUDGET = 100000
    alphas = [-475.25,-1234.60,-799.99]
    betas  = [876.23,1621.78,180.12]

    # Linearly spaced numbers
    x = np.linspace(1, TOTAL_BUDGET, TOTAL_BUDGET)

    # Variables
    watch   = cp.Variable(pos=True)
    phone = cp.Variable(pos=True)
    drone  = cp.Variable(pos=True)

    # Constraint
    constraint = [watch + phone + drone <= TOTAL_BUDGET]

    # Objective
    obj = cp.Maximize(alphas[0] + betas[0] * cp.log(watch)
                    + alphas[1] + betas[1] * cp.log(phone)
                    + alphas[2] + betas[2] * cp.log(drone))
    
    # Solve
    prob = cp.Problem(obj, constraint)
    prob.solve(solver='ECOS', verbose=False)

    # Print solution
    print('='*59 + '\n' + ' '*24 + 'Solution' + ' '*24 + '\n' + '='*59)
    print(f'Status = {prob.status}')
    print(f'Returns = ${round(prob.value):,}\n')
    print('Product allocation:')
    print(f' - Watch    = ${round(watch.value):,}')
    print(f' - Phone  = ${round(phone.value):,}')
    print(f' - Drone   = ${round(drone.value):,}')
    

    # Plot the functions and the results
    fig = plt.figure(figsize=(10, 5), dpi=300)
    plt.plot(x, alphas[0] + betas[0] * np.log(x), color='red', label='Watch ')
    plt.plot(x, alphas[1] + betas[1] * np.log(x), color='blue', label='Phone ')
    plt.plot(x, alphas[2] + betas[2] * np.log(x), color='green', label='Drone ')

    # Plot optimal points
    plt.scatter([watch.value, phone.value, drone.value],
                [alphas[0] + betas[0] * np.log(watch.value),
                alphas[1] + betas[1] * np.log(phone.value),
                alphas[2] + betas[2] * np.log(drone.value)],
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

    # Plot the results
    plt.figure(figsize=(10, 5), dpi=300)
    plt.ticklabel_format(useOffset=False)
    plt.plot(num_iterations_range, best_obj_list, label='Greedy algorithm')
    plt.axhline(y=prob.value, color='r', linestyle='--', label='Optimal solution (CVXPY)')
    plt.xlabel('Number of iterations')
    plt.xticks(num_iterations_range)
    plt.xscale("log")
    plt.ylabel('Best returns ($)')
    plt.title('Best returns found by the greedy algorithm for different numbers of iterations')
    plt.legend()
    plt.show()

if __name__=='__main__':
    main()
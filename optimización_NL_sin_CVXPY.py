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

if __name__=="__main__":

    TOTAL_BUDGET = 1_000_000
    alphas = [1_000_000, 1_000_000, 1_000_000]
    betas = [2, 2, 2]

    # Run the greedy optimization
    (best_google, best_facebook, best_twitter), obj = greedy_optimization(TOTAL_BUDGET, alphas, betas)

    # Print the resultprint('='*59 + '\n' + ' '*24 + 'Solution' + ' '*24 + '\n' + '='*59)
    print(f'Returns = ${round(obj):,}\n')
    print('Marketing allocation:')
    print(f' - Google Ads   = ${round(best_google):,}')
    print(f' - Facebook Ads = ${round(best_facebook):,}')
    print(f' - Twitter Ads  = ${round(best_twitter):,}')
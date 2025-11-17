import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# =====================================================
# Load Dataset (You can replace with your own)
# =====================================================
data = load_breast_cancer()
X = data.data
y = data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

# =====================================================
# Fitness Function: SVM Accuracy (Higher = Better)
# =====================================================
def fitness_function(params):
    C, gamma = params

    # ensure ranges
    if C <= 0 or gamma <= 0:
        return 0

    model = svm.SVC(C=C, gamma=gamma, kernel='rbf')
    scores = cross_val_score(model, X, y, cv=5)
    return scores.mean()  # accuracy

# =====================================================
# Grey Wolf Optimization (GWO)
# =====================================================
def GWO(population_size=20, max_iter=10):
    # Parameter bounds
    C_min, C_max = 0.1, 100
    g_min, g_max = 0.0001, 10

    # Initialize wolves
    wolves = np.zeros((population_size, 2))
    for i in range(population_size):
        wolves[i][0] = np.random.uniform(C_min, C_max)
        wolves[i][1] = np.random.uniform(g_min, g_max)

    # Fitness values
    fitness = np.array([fitness_function(w) for w in wolves])

    # Leaders
    alpha = wolves[np.argmax(fitness)]
    beta  = wolves[np.argsort(fitness)[-2]]
    delta = wolves[np.argsort(fitness)[-3]]

    # Main loop
    for t in range(max_iter):
        a = 2 - t * (2 / max_iter)

        for i in range(population_size):
            for j in range(2):
                
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha[j] - wolves[i][j])
                X1 = alpha[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta[j] - wolves[i][j])
                X2 = beta[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta[j] - wolves[i][j])
                X3 = delta[j] - A3 * D_delta

                wolves[i][j] = (X1 + X2 + X3) / 3

            # Boundary check
            wolves[i][0] = np.clip(wolves[i][0], C_min, C_max)
            wolves[i][1] = np.clip(wolves[i][1], g_min, g_max)

        # Update fitness
        fitness = np.array([fitness_function(w) for w in wolves])

        # Update alpha, beta, delta
        best_idx = np.argsort(fitness)
        alpha = wolves[best_idx[-1]]
        beta  = wolves[best_idx[-2]]
        delta = wolves[best_idx[-3]]

        print(f"Iteration {t+1}/{max_iter} | Best Accuracy: {fitness.max():.4f}")

    return alpha, fitness.max()

# =====================================================
# RUN GWO
# =====================================================
best_params, best_acc = GWO()

print("\n===== BEST PARAMETERS FOUND BY GWO =====")
print(f"C      = {best_params[0]}")
print(f"gamma  = {best_params[1]}")
print(f"Best Accuracy = {best_acc}")

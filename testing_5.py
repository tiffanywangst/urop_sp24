import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from GroupProximalOperator import L1ProximalOperator, SCADProximalOperator, MCPProximalOperator
from Penalty import L1Penalty, SCADPenalty, MCPPenalty
from admm import admm, admm_proxy
from Utilities import create2DSignal, penalty_matrix, create2DPath



# fix gamma and tau to be the optimal values (0.1 and 0.8)
# fix p and vary noise sigma_sq on the log scale 0.0001 to 10 maybe?
# fix noise sigma_sq and vary p from 10 to 50 increments of 5 and graph


# fix noise sigma_sq to be 8e-4 and vary p from 10 to 50 increments of 5

# def vary_p_test(p_values, sigma_sq=8e-4, gamma=0.1, tau=0.8, n1=20, d=50, Y_HIGH=10, Y_LOW=5, P_HIGH=0.7, P_LOW=0.3, num_tests=10):
#     results = []
#     for p in p_values:
#         nnmse_sum = 0
#         for _ in range(num_tests):
#             nnmse_sum += single_test(p, n1, d, gamma, tau, sigma_sq, Y_HIGH, Y_LOW, P_HIGH, P_LOW)
#         avg_nnmse = nnmse_sum / num_tests
#         results.append((p, avg_nnmse))
#     return results


# def single_test(p, n1, d, gamma, tau, sigma_sq, Y_HIGH, Y_LOW, P_HIGH, P_LOW):
#     Gnx, signal_2d, b_true, xs, ys = create2DSignal(0, n1, Y_HIGH=Y_HIGH, Y_LOW=Y_LOW)
    
#     n = nx.number_of_nodes(Gnx)
#     B = np.zeros((d, n))
#     class1 = Y_HIGH * np.random.choice([0, 1], size=(d, 1), p=[P_HIGH, P_LOW])
#     class2 = Y_LOW * np.random.choice([0, 1], size=(d, 1), p=[P_HIGH, P_LOW])
#     class3 = np.random.choice([0, 1], size=(d, 1), p=[P_HIGH, P_LOW])
    
#     for i in range(n):
#         if b_true[i] == Y_HIGH:
#             B[:, i] = class1.ravel()
#         if b_true[i] == Y_LOW:
#             B[:, i] = class2.ravel()
#         if b_true[i] == 1:
#             B[:, i] = class3.ravel()
    
#     Dk = penalty_matrix(Gnx, 0)
    
#     X = np.random.normal(scale=1/np.sqrt(d), size=(p, d))
#     y_true = X.dot(B)
#     Y = y_true + np.random.normal(scale=np.sqrt(sigma_sq), size=(p, n))
#     print(np.linalg.norm(y_true))
    
#     output = admm(Y=Y, X=X, gamma=gamma, tau=tau, Dk=Dk, penalty_f='L1', penalty_param=3, max_iter=10000)
#     B_hat = output['B']
#     nnmse = np.linalg.norm(B_hat - B)**2 / np.linalg.norm(B)**2
#     err_path = output['err_path']
#     print('B_hat:', B_hat)
#     print('B:', B)
#     print('err_path[1]:', err_path[1])
#     print('err_path[3]:', err_path[3])

#     return nnmse


# def plot_results(results):
#     p_values, nnmse_values = zip(*results)
#     plt.plot(p_values, nnmse_values)
#     plt.xlabel('p')
#     plt.ylabel('NNMSE')
#     plt.title('p vs. NNMSE (Fixed Sigma_sq)')
#     plt.show()


# # Fixing sigma_sq
# sigma_sq = 1
# print('sigma_sq:', sigma_sq)

# # Define the range of p values to test
# p_values = np.arange(10, 55, 5)

# # Perform tests
# results = vary_p_test(p_values, sigma_sq)

# # Plot results
# plot_results(results)


# graph the err_path[1] and err_path[3] in the admm.py main function
# try to test when p = 50 and compare B_hat and B
# set penalty_f = 'L1' and see
# put everything on colab


# increase num_iterations, also can change the hyperparameters to see if the results converge better
# invert the matrix (quick calculation just once for each p), plot the OLS solution for every p



def vary_sigma_sq_test(sigma_sq_values, p=30, gamma=0.1, tau=0.8, n1=20, d=50, Y_HIGH=10, Y_LOW=5, P_HIGH=0.7, P_LOW=0.3, num_tests=10):
    results = []
    for sigma_sq in sigma_sq_values:
        print('sigma_sq:', sigma_sq)
        nnmse_sum = 0
        for _ in range(num_tests):
            nnmse_sum += single_test(p, n1, d, gamma, tau, sigma_sq, Y_HIGH, Y_LOW, P_HIGH, P_LOW)
        avg_nnmse = nnmse_sum / num_tests
        print('avg_nnmse:', avg_nnmse)
        results.append((sigma_sq, avg_nnmse))
    return results


def single_test(p, n1, d, gamma, tau, sigma_sq, Y_HIGH, Y_LOW, P_HIGH, P_LOW):
    Gnx, signal_2d, b_true, xs, ys = create2DSignal(0, n1, Y_HIGH=Y_HIGH, Y_LOW=Y_LOW)
    
    n = nx.number_of_nodes(Gnx)
    B = np.zeros((d, n))
    class1 = Y_HIGH * np.random.choice([0, 1], size=(d, 1), p=[P_HIGH, P_LOW])
    class2 = Y_LOW * np.random.choice([0, 1], size=(d, 1), p=[P_HIGH, P_LOW])
    class3 = np.random.choice([0, 1], size=(d, 1), p=[P_HIGH, P_LOW])
    
    for i in range(n):
        if b_true[i] == Y_HIGH:
            B[:, i] = class1.ravel()
        if b_true[i] == Y_LOW:
            B[:, i] = class2.ravel()
        if b_true[i] == 1:
            B[:, i] = class3.ravel()
    
    Dk = penalty_matrix(Gnx, 0)
    
    X = np.random.normal(scale=1/np.sqrt(d), size=(p, d))
    y_true = X.dot(B)
    Y = y_true + np.random.normal(scale=np.sqrt(sigma_sq), size=(p, n))
    
    output = admm(Y=Y, X=X, gamma=gamma, tau=tau, Dk=Dk, penalty_f='L1', penalty_param=3, max_iter=300)
    B_hat = output['B']
    nnmse = np.linalg.norm(B_hat - B)**2 / np.linalg.norm(B)**2
    
    print('B_hat:', B_hat)
    print('B:', B)

    return nnmse


def plot_results(results):
    sigma_sq_values, nnmse_values = zip(*results)
    plt.plot(sigma_sq_values, nnmse_values)
    plt.xlabel('sigma_sq')
    plt.ylabel('NNMSE')
    plt.title('sigma_sq vs. NNMSE (Fixed p)')
    plt.xscale('log')
    plt.show()


# Fixing p
p = 40
print('p:', p)

# Define the range of sigma_sq values to test
sigma_sq_values = np.logspace(-3, 1, num=10)

# Perform tests
results = vary_sigma_sq_test(sigma_sq_values, p, gamma=1, tau=0.8)

plot_results(results)



from scipy.optimize import minimize

# loss function
fun = lambda x: x[0]**2 + x[1]**2 - 6*x[0]*x[1] - 4*x[0] - 5*x[1]

# constraints
c1 = lambda x:  -(x[0] - 2)**2 + 4 - x[1]
c1_updated = lambda x:  -(x[0] - 2)**2 + 4.1 - x[1]
c2 = lambda x: x[1] + x[0] - 1

cons = ({'type': 'ineq', 'fun': c1},
{'type': 'ineq', 'fun': c2})

res = minimize(fun, (2, 0), method='SLSQP', constraints=cons)

print(res)

cons = ({'type': 'ineq', 'fun': c1_updated},
{'type': 'ineq', 'fun': c2})

res = minimize(fun, (2, 0), method='SLSQP', constraints=cons)

print(res)
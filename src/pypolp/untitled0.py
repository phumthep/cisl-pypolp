import gurobipy as gp
import pandas as pd

constr_names = ['a', 'b', 'c']

model = gp.Model()
for constr_name in constr_names:
    model.addConstr(0 == 0, name=constr_name)
    
model.update()


foo = pd.DataFrame([0]*5, index=[f'c_{i+1}' for i in range(5)], columns=['value'])

list(foo.index)

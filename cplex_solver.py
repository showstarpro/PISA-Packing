from solve import Program
program = Program()
program.get_dependency()
blocks = program.blocks
import random
N = len(blocks)
# M = 1204
M = 600 # debug
set_I = range(0, N) # 基本快
set_J = range(0, M) # 流水线    
# 已知变量
t = {i: blocks[i].resources['tcam'] for i in set_I}
h = {i: blocks[i].resources['hash'] for i in set_I}
a = {i: blocks[i].resources['alu'] for i in set_I}
q = {i: blocks[i].resources['qualify'] for i in set_I}
# 次序关系
lt_table = {}
for i in set_I:
    for j in range(0,i):
        if j in blocks[i].le:
            lt_table[(i,j)] = 2 
        if j in blocks[i].lt:
            lt_table[(i,j)] = 1  
        else:
            lt_table[(i,j)] = 0
# 流水线限制
T = {j: 1 for j in set_J}
H = {j: 2 for j in set_J}
A = {j: 56 for j in set_J}
Q = {j: 64 for j in set_J}
# 引入求解器
import docplex.cp.model as cpx
opt_model = cpx.CpoModel(name="PISA")
# if x is Binary
x = {(i,j): opt_model.binary_var(name=f"x_{i}_{j}") for i in set_I for j in set_J}
block_con = { 
    i : 
    opt_model.add_constraint(opt_model.sum(x[i,j] for j in set_J) ==1 ) 
    for i in set_I
}
tcam_con = { 
    j : 
    opt_model.add_constraint(opt_model.sum(t[i] * x[i,j] for i in set_I) <= T[j]) 
    for j in set_J
}
alu_con = { 
    j : 
    opt_model.add_constraint(opt_model.sum(a[i] * x[i,j] for i in set_I) <= A[j]) 
    for j in set_J
}
hash_con = { 
    j : 
    opt_model.add_constraint(opt_model.sum(h[i] * x[i,j] for i in set_I) <= H[j]) 
    for j in set_J
}
qualify_con = {
    j : 
    opt_model.add_constraint(opt_model.sum(q[i] * x[i,j] for i in set_I) <= Q[j]) 
    for j in set_J   
}
even_con = opt_model.add_constraint(opt_model.sum(t[i] * x[i,j] for i in set_I for j in set_J if j%2==0 ) <= 5 ) 
tmp = []
for a in range(0,15):
    b = a+ 16
    if b > M:
        break
    else:
        tmp.append(opt_model.add_constraint(opt_model.sum(t[i] * x[i,b] for i in set_I) <= 1 ))
        tmp.append(opt_model.add_constraint(opt_model.sum(h[i] * x[i,b] for i in set_I) <= 3 ))
objective = (opt_model.sum(opt_model.max(x[i,j] for i in set_I) for j in set_J))
opt_model.minimize(objective)
# control dependency:
# 次序关系
r  = { i: opt_model.integer_var(0,  M-1,name=f"r_{i}") for i in set_I} 
for a in set_I:
    for b in blocks[a].lt: # ra < rb
        opt_model.add_constraint(r[a] <= r[b]-1 )
    for b in blocks[a].le:
        opt_model.add_constraint(r[a] <= r[b])
opt_model.add_constraint(cpx.element([x[i,j] for j in set_J],r[i]) == 1 for i in set_I)

# solving with local cplex
msol = opt_model.solve(log_output=True,FailLimit=100000, TimeLimit=1000)
msol.print_solution() 
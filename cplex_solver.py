from solve import Program
# 引入求解器
import docplex.mp.model as cpx
opt_model = cpx.Model(name="PISA")

def main():
    program = Program()
    program.get_dependency()
    blocks = program.blocks

    N = 607
    # M = 1204
    M =2 # debug
    set_I = range(0, N) # 基本快
    set_J = range(1, M+1) # 流水线
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

    # x_ij: block i in pipeline j
    x = {(i,j): opt_model.binary_var(name=f"x_{i}_{j}") for i in set_I for j in set_J}
    constraint = { 
        j : 
        opt_model.add_constraint(ct= opt_model.sum(t[i] * x[i,j] for i in set_I) <= T[j], ctname=f"constraint_{j}") 
        for j in set_J
    }


if __name__ == '__main__':
    try:
        main()
    except:
        import sys,pdb,bdb
        type, value, tb = sys.exc_info()
        if type == bdb.BdbQuit:
            exit()
        print(type,value)
        pdb.post_mortem(tb)
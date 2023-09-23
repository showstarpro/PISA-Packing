from solve import Program
from sko.GA import GA

# 将problem1的求解结果作为初始值，开始启发式搜索
filename = 'data/result__52.txt'
ROOT = 365
program = Program()
program.get_dependency()
blocks = program.blocks
sons = program.son
N = len(blocks)
set_I = range(0, N) 
in_node = [0]*N
for v,l in program.graph.items():
    for i in l:
        in_node[i] += 1

t = {i: blocks[i].resources['tcam'] for i in set_I}
h = {i: blocks[i].resources['hash'] for i in set_I}
alu = {i: blocks[i].resources['alu'] for i in set_I}
q = {i: blocks[i].resources['qualify'] for i in set_I}


rbest = []
for i,line in enumerate(open(filename)):
    if i==0:
        continue
    num = int(line[:-1])
    rbest.append(num)    


def longest_path(graph,ifin,hash,alu,in_):
    from queue import Queue

    hash_ = [h if i else 0 for i,h in zip(ifin,hash) ]
    alu_ = [h if i else 0 for i,h in zip(ifin,alu) ]

    maxhash = [0 for i in range(len(graph))]
    maxalu = [0 for i in range(len(graph))]

    topo = [ROOT]
    maxhash[ROOT] = hash_[ROOT]
    maxalu[ROOT] = alu_[ROOT]
    q = Queue()
    q.put(ROOT)
    while q.qsize() != 0:
        cur = q.get()
        for next in graph[cur]:
            in_[next] -= 1
            maxhash[next]=max(maxhash[next],maxhash[cur]+hash_[next])
            maxalu[next] = max(maxalu[next],maxalu[cur]+alu_[next])
            if in_[next] == 0:
                q.put(next)
                topo.append(next)
    return max(maxhash),max(maxalu)

# tuple
def opt_func_(rs):
    rs = list(int(r) for r in rs)
    # 流水线最大编号
    rmax = max(rs)
    eq_func = []
    # 流水线 -> blk
    pipe = {}
    for i,r in enumerate(rs):
        if r not in pipe:
            pipe[r] = [i]
        else:
            pipe[r].append(i)
    # 资源限制
    even = 0
    for k,v in pipe.items():
        if len(v) == 0:
            continue
        tcam_ = 0
        hash_ = 0
        alu_ = 0
        qualify_ =0 
        for blk in v:
            tcam_ += t[blk]
            hash_ += h[blk]
            alu_ += alu[blk]
            qualify_ += q[blk]
        
        eq_func.append(tcam_-1)
        eq_func.append(hash_-2)
        eq_func.append(alu_-56)
        eq_func.append(qualify_-64)
        if k % 2 == 0:
            even += tcam_
    eq_func.append(even-5)

    keys = list(pipe.keys())
    keys.sort()
    for k in keys:
        if k > 15 or k+16 > rmax:
            break
        if k + 16 not in pipe:
            continue
        print(k,k+16)
        tcam_ = 0
        hash_ = 0
        alu_ = 0
        qualify_ =0 
        for blk in pipe[k]:
            tcam_ += t[blk]
            hash_ += h[blk]
            alu_ += alu[blk]
            qualify_ += q[blk]
        for blk in pipe[k+16]:
            tcam_ += t[blk]
            hash_ += h[blk]
            alu_ += alu[blk]
            qualify_ += q[blk]
        eq_func.append(tcam_-1)
        eq_func.append(hash_-3)

    for a in set_I:
        for b in blocks[a].lt: 
            eq_func.append(rs[a] - (rs[b]-1))
        for b in blocks[a].le:
            eq_func.append(rs[a] - rs[b])

    loss = 0
    for e in eq_func:
        loss += max(e, 0)
    
    return rmax , loss

# For problem2
def opt_func2_(rs):

    rs = list(int(r) for r in rs)
    # 流水线最大编号
    rmax = max(rs)
    # 流水线 -> blk
    pipe = {}
    for i,r in enumerate(rs):
        if r not in pipe:
            pipe[r] = [i]
        else:
            pipe[r].append(i)
    # 资源限制 1,4,6
    eq_func = []
    even = 0
    for k,v in pipe.items():
        if len(v) == 0:
            continue
        tcam_ = 0
        qualify_ =0 
        for blk in v:
            tcam_ += t[blk]
            qualify_ += q[blk]
        eq_func.append(tcam_-1)
        eq_func.append(qualify_-64)
        if k % 2 == 0:
            even += tcam_
    eq_func.append(even-5)
    # 2，3，
    # 同一控制流程上的限制用最长路实现
    import copy
    maxhs = []
    for j in range(rmax+1):
        ifin = [ 1 if r==j else 0 for r in rs ]
        maxh,maxalu = longest_path(program.graph,ifin,list(h.values()),list(alu.values()),copy.deepcopy(in_node))
        maxhs.append(maxh)
        eq_func.append(maxh - 2)
        eq_func.append(maxalu - 56)

    # 资源限制 5
    keys = list(pipe.keys())
    keys.sort()
    for k in keys:
        if k > 15 or k+16 > rmax:
            break
        if k + 16 not in pipe:
            continue

        tcam_ = 0
        for blk in pipe[k]:
            tcam_ += t[blk]
        for blk in pipe[k+16]:
            tcam_ += t[blk]        
        hash_ = maxhs[k] + maxhs[k+16]

        eq_func.append(tcam_-1)
        eq_func.append(hash_-3)

    # 控制限制
    for a in set_I:
        for b in blocks[a].lt: 
            eq_func.append(rs[a] - (rs[b]-1))
        for b in blocks[a].le:
            eq_func.append(rs[a] - rs[b])

    loss = 0
    for e in eq_func:
        loss += max(e, 0)
    
    return rmax , loss


def opt_func(rs):
    losses = opt_func2_(rs)
    return  1000 * losses[0] + 1000 * losses[1]  

def ueq_funcs():
    funcs = []
    for a in set_I:
        for b in blocks[a].lt: # ra < rb
            funcs.append(lambda rs: rs[a] - (rs[b]-1))
        for b in blocks[a].le:
            funcs.append(lambda rs: rs[a] - rs[b])
    return tuple(funcs)


# @ti.kernel
def ga_solve(M:int):

    # 遗传算法直接看参数，输入正确即可
    # 直接优化bi
    ga = GA(
        func=opt_func, 
        n_dim=N, 
        max_iter=500, 
        lb=0, 
        ub=M-1, 
        # precision为整数，则表示整数规划
        precision=1,
        # a list of equal functions with ceq[i] = 0
        constraint_eq = tuple(),
        # a list of unequal constraint functions with c[i] <= 0
        constraint_ueq= tuple(),
    )
    best_x, best_y = ga.run()
    print(opt_func_(best_x))
    with open('data/ga.txt','w') as f:
        print('best:',max(best_x),file=f)
        for x in best_x:
            print(int(x),file=f)

def pso_solve():
    from sko.PSO import PSO
    M = len(rbest)
    print('M:',M)    
    pso = PSO(
        func=opt_func, 
        n_dim=N, 
        pop=400, 
        max_iter=150, 
        lb=0, 
        ub=M-1, 
        w=0.8, c1=0.5, c2=0.5)
    pso.X = [rbest for _ in range(M)]
    pso.cal_y(); pso.update_gbest(); pso.update_pbest()
    pso.run()
    
    best_x = pso.gbest_x
    print(opt_func_(best_x))
    with open('data/pso.txt','w') as f:
        print('best:',max(best_x),file=f)
        for x in best_x:
            print(int(x),file=f)

def sa_solve():
    from sko.SA import SA
    sa = SA(func=opt_func, x0=rbest, T_max=1, T_min=1e-9, L=10, max_stay_counter=15)
    best_x, best_y = sa.run()
    result = opt_func2_(best_x)
    print(result)
    if result[1] != 0:
        return False
    with open('data/sa.txt','w') as f:
        print('best:',max(best_x)+1,file=f)
        for x in best_x:
            print(int(x),file=f)
        return True

def de_solve():
    from sko.DE import DE
    import numpy as np
    M = len(rbest)
    print('M:',M)
    de = DE(func=opt_func, n_dim=N, max_iter=800, lb=0, ub=M-1,
            constraint_eq=tuple(), constraint_ueq=tuple())
    # de.X = np.array(rbest)
    best_x, best_y = de.run()
    print(opt_func_(best_x))
    with open('data/de.txt','w') as f:
        print('best:',max(best_x),file=f)
        for x in best_x:
            print(int(x),file=f)



def rechecker():
    r = []
    for i,line in enumerate(open(filename)):
        if i==0:
            continue
        num = int(line[:-1])
        r.append(num)
    print(opt_func2_(r))


sa_solve()
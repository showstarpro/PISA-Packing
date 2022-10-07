
from tqdm import trange

class pipeline:
    def __init__(self,):
        self.scheduled = [] # blk list
        self.resource = {
            'tcam': 1,
            'hash': 2,
            'alu': 56,
            'qualify': 64,   
        }
        self.num = 0

    def self_check(self):
        # 检查第i级流水线
        for r,k in self.resource.items():
            if sum(blk.resource[r] for blk in self.scheduled) > k:
                return False
        return True

    def get_resource(self,name):
        # 检查流水线之间的限制
        return sum(blk.resource[name] for blk in self.scheduled)


class pipelines:
    def __init__(self,):
        self.lists = []

    def add(self,pipe):
        self.lists.append(pipe)
    
    def check_tcam(self,):
        # 有TCAM资源的偶数级数量不超过5；
        tcam_cnt = 0
        for i,l in enumerate(self.list):
            if i%2: continue
            tcam_cnt += l.get_resource('tcam')
            if tcam_cnt > 5 : return False
        return True

    def check_fold(self,):
        # 约定流水线第0级与第16级，第1级与第17级，...，第15级与第31级为折叠级数，折叠的两级TCAM资源加起来最大为1，HASH资源加起来最大为3。注：如果需要的流水线级数超过32级，则从第32开始的级数不考虑折叠资源限制；
        if len(self.lists) < 17:
            return True
        max_fold_idx = min(31,len(self.lists)-1) - 16
        for i in range(max_fold_idx):
            if self.lists[i].get_resource('tcam') + self.lists[i+16].get_resource('tcam') > 1:
                return False
            if self.lists[i].get_resource('hash') + self.lists[i+16].get_resource('hash') > 3:
                return False
        return True


class block:
    def __init__(self,*args):
        self.resources = {
            'tcam': args[0],
            'hash': args[1],
            'alu': args[2],
            'qualify': args[3],
        }
        self.r = set()
        self.w = set()
        self.le = set() # <=
        self.lt = set() # <
    
    def __repr__(self,):
        return f'(\nr:{self.r},\nw:{self.w},\nle:{self.le},\nlt:{self.lt}\n)'

ROOT = 365 # 根节点
blocks = []
readmap,writemap = dict(),dict() # 556,779
graph = dict()

def read_data():
    # 资源
    for i,line in enumerate(open('attachment1.csv')):
        if i == 0:continue
        line = line[:-1]
        blocks.append(block(*line.split(',')[1:]))
    # 数据依赖
    for i,line in enumerate(open('attachment2.csv')):
        if i == 0:continue
        line = line[:-1].split(',')
        if len(line) == 2:continue
        if line[1] == 'R': 
            blocks[int(line[0])].r = set(line[2:])
            for v in line[2:]:
                if v not in readmap:
                    readmap[v] = []
                readmap[v].append(int(line[0]))
        elif line[1] == 'W': 
            blocks[int(line[0])].w = set(line[2:])
            for v in line[2:]:
                if v not in writemap:
                    writemap[v] = []
                writemap[v].append(int(line[0])) 
    # CFG
    for i,line in enumerate(open('attachment3.csv')):
        line = line[:-1].split(',')
        graph[int(line[0])] = [int(v) for v in line[1:]]    
    # 可视化 
    # draw_graph(graph)

    # root 365
    # leaves [2, 88, 360, 381, 454]
    # dfs(ROOT,[0]*607) 
    # bfs(ROOT,[0]*607)
    # toposort(graph,ROOT)
    return graph
    # 连通的

def draw_graph(graph):

    import networkx as nx
    import matplotlib.pyplot as plt  
    import pygraphviz
    edges = []
    for node in graph:
        for i in graph[node]:
            edges.append((node, i))
    
    G = nx.DiGraph()
    G.add_edges_from(edges)
    colors = ['green' if node_name in [2, 88, 360, 381, 454] else 'blue' for node_name in list(G.nodes)]
    colors[list(G.nodes).index(ROOT)] = 'red'
    plt.figure(num=None, figsize=(50, 50), dpi=80)
    nx.draw(G, node_size=600, node_color = colors,linewidths=0.25, with_labels=True,pos=nx.nx_agraph.graphviz_layout(G))
    plt.savefig('graph.png') 

# 贪心方法: 首先处理出依赖图，显然只能从没有任何依赖的节点开始(记为ready), 然后计算出ready的优先级，尝试分配到active, 如果满足运行条件的话，更新ready
def schedule(pdg):
    # pdg 为控制依赖 数据依赖处理好的图
    # 贪心选择
    def greedy_choose():

        total = dict()
        for name in ['tcam','alu','hash','qualify']:
            total[name] = 0
            for u in used:
                total[name] += blocks[u].resources[name]
        # 挑选可以选择的candidate
        for cur in candidate:
            for name in ['tcam','alu','hash','qualify']:
                if total[name] + blocks[cur].resources[name]:
                    pass

    cycle = 1 # 流水线层级
    
    # 从没有依赖的节点开始
    candidate = set([x for x in range(607) if x.get_deps() == set([]) ])
    used = set([])
    while candidate.union(used) != set([]):
        if len(candidate) != 0:
            cur = greedy_choose(candidate)
            used.add(cur)
            scheduled[cycle].append(cur)
            # 该节点所支配的节点：
            for next in pdg[cur]:
                pass
    pass



# 数据依赖 
def data_dependency(son):
    # 首先获取每个节点的所有child
    for parent in range(len(graph)):
        for child in son[parent]:
            # r -> w  <=
            if len(blocks[parent].r.union(blocks[child].w))!=0:
                blocks[parent].le.add(child)
            # w -> r  <
            if len(blocks[parent].w.union(blocks[child].r))!=0:
                blocks[parent].lt.add(child)
            # w -> w  <
            if len(blocks[parent].w.union(blocks[child].w))!=0:
                blocks[parent].lt.add(child)
        
# 控制依赖
def control_dependency():

    dom = dict()
    son = dict()
    def get_diff_node(cur):
        if len(graph[cur]) == 0:
            son[cur] = set([])
            return
        path = []
        son[cur] = set([next for next in graph[cur]])
        for next in graph[cur]:
            if next not in son:
                get_diff_node(next)
            son[cur] |= son[next]
            path.append(son[next]|set([next]))
        
        if len(path) == 1:
            dom[cur] = set([])
            return
        dom[cur] = set([])
        for i in range(len(path)-1):
            for j in range(i,len(path)):
                dom[cur] |= path[i].symmetric_difference(path[j])
    
    get_diff_node(ROOT)

    for k,v in dom.items():
        blocks[k].le |= v
    return son
    
# 依赖图 =  数据依赖 + 控制依赖; 直接维护<节点和<=节点即可
def dependency():
    son = control_dependency()
    data_dependency(son)

# => 4D bin-packing 分配即可


if __name__ ==  '__main__':
    try:
        read_data()
        dependency()
    except:
        import sys,pdb,bdb
        type, value, tb = sys.exc_info()
        if type == bdb.BdbQuit:
            exit()
        print(type,value)
        pdb.post_mortem(tb)
    
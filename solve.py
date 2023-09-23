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

ROOT = 365 # 根节点
class block:
    def __init__(self,*args):
        self.resources = {
            'tcam': int(args[0]),
            'hash': int(args[1]),
            'alu': int(args[2]),
            'qualify': int(args[3]),
        }
        self.r = set()
        self.w = set()
        self.le = set() # <=
        self.lt = set() # <
    
    def __repr__(self,):
        return f'(\nr:{self.r},\nw:{self.w},\nle:{self.le},\nlt:{self.lt}\n)'

#  贪心方法: 首先处理出依赖图，显然只能从没有任何依赖的节点开始(记为ready), 然后计算出ready的优先级，尝试分配到active, 如果满足运行条件的话，更新ready
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
    while candidate.intersection(used) != set([]):
        if len(candidate) != 0:
            cur = greedy_choose(candidate)
            used.add(cur)
            scheduled[cycle].append(cur)
            # 该节点所支配的节点：
            for next in pdg[cur]:
                pass
    pass

class Program:
    def __init__(self,):
        self.graph = dict()
        self.blocks = []
        self.read_data()

    def read_data(self,):
        # 检查csv最后一行要是空的
        # 资源
        for i,line in enumerate(open('attachment1.csv')):
            if i == 0:continue
            line = line[:-1]
            self.blocks.append(block(*line.split(',')[1:]))
        # 数据依赖
        for i,line in enumerate(open('attachment2.csv')):
            line = line[:-1].split(',')
            if len(line) == 2:continue
            if line[1] == 'R': 
                self.blocks[int(line[0])].r = set(line[2:])
            elif line[1] == 'W': 
                self.blocks[int(line[0])].w = set(line[2:]) 
        # CFG
        for i,line in enumerate(open('attachment3.csv')):
            line = line[:-1].split(',')
            self.graph[int(line[0])] = [int(v) for v in line[1:]]

    # 依赖图 =  数据依赖 + 控制依赖; 直接维护<节点和<=节点即可
    def get_dependency(self,):
        # 由于control有个求解son的过程，所以要先于data
        self.control_dependency()
        self.data_dependency()

    def control_dependency(self,):
        dom = dict()
        son = dict()
        def get_diff_node(cur):
            if len(self.graph[cur]) == 0:
                son[cur] = set([])
                return
            path = []
            son[cur] = set([next for next in self.graph[cur]])
            for next in self.graph[cur]:
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
            self.blocks[k].le |= v
        self.son = son

    def data_dependency(self,):
        # 首先获取每个节点的所有child
        # <优先级高于<=
        for parent in range(len(self.graph)):
            for child in self.son[parent]:
                if parent in (self.blocks[child].lt | self.blocks[child].le):
                    raise Exception('parent dependent on child!')
                # w -> r  <
                if len(self.blocks[parent].w.intersection(self.blocks[child].r))!=0:
                    self.blocks[parent].lt.add(child)
                    if child in self.blocks[parent].le:
                        self.blocks[parent].le.remove(child)
                # w -> w  <
                if len(self.blocks[parent].w.intersection(self.blocks[child].w))!=0:
                    self.blocks[parent].lt.add(child)
                    if child in self.blocks[parent].le:
                        self.blocks[parent].le.remove(child)
                # r -> w  <=
                if len(self.blocks[parent].r.intersection(self.blocks[child].w))!=0:
                    if child in self.blocks[parent].lt:
                        continue
                    self.blocks[parent].le.add(child)


if __name__ ==  '__main__':
    program = Program()
    program.get_dependency()
    print(program.blocks)
    

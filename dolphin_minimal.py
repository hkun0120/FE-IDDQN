import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from collections import deque, namedtuple
import random, os, glob, tqdm

device = torch.device('cpu')
# ---------- 1. 读入数据 ----------
BASE = '~/Downloads/ddqndata'
print("当前工作目录:", os.getcwd())
print("匹配到的文件:", glob.glob(os.path.join(BASE, '*process_definition*')))
dtype = {str(i): 'string' for i in range(100)}   # 先全部当字符串读

def load_data(path):
    return pd.read_csv(path
                # ,on_bad_lines='skip',   # 忽略异常行
                # nrows=1000
                       )
process_def = pd.read_csv(glob.glob(f'{BASE}/*process_definition*')[0])
# process_def = load_data(glob.glob(f'{BASE}/*process_definition*')[0])
process_inst = load_data(glob.glob(f'{BASE}/*process_instance*')[0])
task_def = pd.read_csv(glob.glob(f'{BASE}/*task_definition*')[0])
task_inst = load_data(glob.glob(f'{BASE}/*task_instance*')[0])
task_relation = pd.read_csv(glob.glob(f'{BASE}/*process_task_relation*')[0])


# ---------- 2. 构造 DAG ----------
def build_dag(process_code):
    """
    通过 task_relation 与 task_def 构造 nx.DiGraph
    """
    # 1. 找出该工作流的所有任务代码
    rel = task_relation[task_relation['process_definition_code'] == process_code]
    task_codes_in_flow = set(rel['pre_task_code']).union(set(rel['post_task_code']))

    # 2. 按 code 拿到任务定义
    tasks = task_def[task_def['code'].isin(task_codes_in_flow)][
        ['code', 'name']
    ].copy()

    # 3. 计算平均运行时长（秒）
    ins = task_inst[
        (task_inst['task_code'].isin(task_codes_in_flow)) &
        (task_inst['state'].isin(['7', '6'])) &
        (task_inst['start_time'].notna()) &
        (task_inst['end_time'].notna())
        ].copy()

    # 把字符串时间转成 datetime
    ins['start_time'] = pd.to_datetime(ins['start_time'])
    ins['end_time'] = pd.to_datetime(ins['end_time'])
    ins['duration'] = (ins['end_time'] - ins['start_time']).dt.total_seconds()

    # 取每个 task_code 的平均
    dur_map = ins.groupby('task_code')['duration'].mean().to_dict()
    tasks['duration'] = tasks['code'].map(dur_map).fillna(60)  # 缺省 60 秒

    # 4. 资源占位：先写死，后续再替换
    tasks['cpu'] = 1  # 固定 1 vCPU
    tasks['mem'] = 1024  # 固定 1024 MB
    # 4. 建图
    G = nx.DiGraph()
    for _, row in tasks.iterrows():
        G.add_node(row['code'],
                   duration=row['duration'],
                   cpu=row['cpu'],
                   mem=row['mem'])

    # 5. 加边

    print('task_codes_in_flow:', len(task_codes_in_flow))
    print('edges found:', len(rel))
    for _, r in rel.iterrows():
        if r['pre_task_code'] in G and r['post_task_code'] in G:
            G.add_edge(r['pre_task_code'], r['post_task_code'])
    return G


# 选一个出现次数最多的工作流做演示
proc_code = process_inst['process_definition_code'].value_counts().idxmax()
DAG = build_dag(proc_code)
TASKS = list(DAG.nodes)
n_tasks = len(TASKS)


# ---------- 3. 简单调度环境 ----------
class SimpleEnv:
    """
    状态：每个任务是否完成 (0/1)
    动作：选一个未就绪的就绪任务调度到某台机器
    奖励：-1 每步，直到完成；结束额外 -makespan
    机器：2 台，同质
    """

    def __init__(self, dag):
        self.dag = dag
        self.n_machines = 2
        self.reset()

    def reset(self):
        self.done = [False] * len(TASKS)
        self.machine_busy = [0] * self.n_machines  # 当前机器可用时间
        self.t = 0
        return self._state()

    def _state(self):
        return np.array(self.done, dtype=np.float32)

    def _ready(self):
        ready = [i for i, d in enumerate(self.done) if not d and
                 all(self.done[TASKS.index(p)] for p in self.dag.predecessors(TASKS[i]))]
        return ready

    def step(self, action):
        # action: 0..n_tasks-1 选一个就绪任务
        if action not in self._ready():
            return self._state(), -10, False, {}  # 非法动作
        idx = action
        task_code = TASKS[idx]
        dur = self.dag.nodes[task_code]['duration']
        # 贪心放到最早空闲机器
        m = np.argmin(self.machine_busy)
        start = max(self.machine_busy[m], self.t)
        end = start + dur
        self.machine_busy[m] = end
        self.done[idx] = True
        self.t = max(self.t, end)
        if all(self.done):
            return self._state(), -self.t, True, {'makespan': self.t}
        return self._state(), -1, False, {}


# ---------- 4. 简化 Q 网络 ----------
class QNet(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_out)
        )

    def forward(self, x):
        return self.net(x)


# ---------- 5. DDQN Agent ----------
class DDQNAgent:
    def __init__(self, n_states, n_actions, lr=1e-3, gamma=0.99, eps=1.0, eps_min=0.01, eps_decay=0.995):
        # 动作空间大小
        self.n_actions = n_actions
        # 折扣因子γ，决定未来奖励的影响
        self.gamma = gamma
        # 当前探索率ε，控制随机动作概率
        self.eps = eps
        # 最小探索率，防止ε降为0
        self.eps_min = eps_min
        # 探索率衰减系数，每次更新后ε乘以该值
        self.eps_decay = eps_decay
        # 主Q网络，用于估算Q值
        self.q_net = QNet(n_states, n_actions)
        # 目标Q网络，用于计算目标Q值，提升训练稳定性
        self.q_target = QNet(n_states, n_actions)
        # 初始化目标网络参数，使其与主Q网络一致
        self.q_target.load_state_dict(self.q_net.state_dict())
        # Adam优化器，负责更新主Q网络参数
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        # 经验回放池，保存最近10000条经验
        self.memory = deque(maxlen=10000)
        # 定义经验元组格式（状态，动作，奖励，下一个状态，是否结束）
        self.Transition = namedtuple('T', ('s', 'a', 'r', 's_', 'd'))

    def act(self, state):
        # ε-贪婪策略：以ε概率随机选动作（探索），否则选Q值最大的动作（利用）
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state).unsqueeze(0)  # 转为张量并增加batch维
            return self.q_net(s).argmax().item()  # 选Q值最大的动作

    def store(self, *args):
        # 存储一条经验到回放池
        self.memory.append(self.Transition(*args))

    def replay(self, batch=64):
        # 经验回放，采样一批数据进行训练
        if len(self.memory) < batch: return  # 数据不足时跳过
        samples = random.sample(self.memory, batch)  # 随机采样batch条经验
        s, a, r, s_, d = zip(*samples)  # 拆分为各自的分量
        s = torch.tensor(np.array(s))
        s_ = torch.tensor(np.array(s_))
        a = torch.tensor(a)
        r = torch.tensor(r, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32)

        # 计算当前Q值
        q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            # 计算下一个状态的最大动作
            a_next = self.q_net(s_).argmax(1)
            # 用目标网络计算下一个状态的Q值
            q_next = self.q_target(s_).gather(1, a_next.unsqueeze(1)).squeeze()
            # 计算目标Q值
            target = r + self.gamma * q_next * (1 - d)
        # 均方误差损失
        loss = nn.MSELoss()(q, target)
        self.opt.zero_grad();  # 梯度清零
        loss.backward();       # 反向传播
        self.opt.step()        # 更新参数

    def update_target(self):
        # 更新目标网络参数为主Q网络参数
        self.q_target.load_state_dict(self.q_net.state_dict())
        # 衰减探索率ε
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


# ---------- 6. 训练 ----------
env = SimpleEnv(DAG)
agent = DDQNAgent(n_states=n_tasks, n_actions=n_tasks)

for ep in tqdm.tqdm(range(100)):
    s = env.reset()
    while True:
        a = agent.act(s)
        s_, r, done, info = env.step(a)
        agent.store(s, a, r, s_, done)
        s = s_
        if done:
            print(f"ep={ep}  makespan={info['makespan']:.1f}  eps={agent.eps:.3f}")
            break
    agent.replay()
    if ep % 10 == 0:
        agent.update_target()

# 保存模型
torch.save(agent.q_net.state_dict(), 'ddqn_skeleton.pt')
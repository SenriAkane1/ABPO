from collections import deque
from typing import List, Dict, Iterable, Iterator, Optional, Sequence, Literal
import random
class CurriculumBatchSampler:
    """
    极简课程学习 BatchSampler（队列语义：都从队头取、放队尾）
    - 训练模式：从“正在学”优先取满一个 batch，不足则用“已学会”补齐
    - 困难题模式：按顺序成批读出“困难题”用于 rollout（不修改队列）
    """

    def __init__(
        self,
        dataset_len: int,
        batch_size: int,
        *,
        init_learning_indices: Optional[Sequence[int]] = None,  # 初始“正在学”的索引；不传则为空
        hard_indices: Optional[Sequence[int]] = None,           # 初始“困难题”的索引；不传则为 0..N-1
        acc_threshold: float = 0.85,                             # 准确率阈值（>= 则晋级已学会）
        
        trunc_len_list: int =  [0,2000,2500,3000,3500,4000,4500],      # 初始阶段的截断长度
        review_rate: float = 0.1   #复习的比例
    ):
        assert dataset_len > 0 and batch_size > 0
        self.dataset_len = int(dataset_len)
        self.batch_size = int(batch_size)
        self.acc_threshold = float(acc_threshold)
        self.review_rate = review_rate

        # 三个队列（都从队头取、放队尾）
        all_idx = list(range(self.dataset_len))
        if hard_indices is None:
            hard_indices = all_idx
        hard_set = set(int(i) for i in hard_indices)

        if init_learning_indices is None:
            init_learning_indices = []


        # 去掉重复、越界；并保证三者互斥
        learn_seen = set()
        self.learning: deque[int] = deque()
        # self.learning = deque(list(range(self.dataset_len)))
        self.mastered: deque[int] = deque()  # 初始为空
        self.hard: deque[int] = deque([i for i in hard_set if i not in learn_seen])

        # 记录初始“正在学”规模，用于阶段提升阈值
        self.initial_learning_size = 0

        # 阶段与截断长度
        self.stage = 0
        self.trunc_len_list = trunc_len_list
        self.current_trunc_len = trunc_len_list[0]


        # 每个样本的最大输出长度（在“困难题”加入“正在学”时锁定）
        self.stage_map: Dict[int, int] = {}

        # 迭代模式：'train' 从 学习+已学会 取；'hard' 从 困难题 取
        self.mode: Literal['train', 'hard'] = 'train'

        # 困难题模式的只读游标（不改变队列）
        self._hard_read_ptr = 0  # 指向 hard 中的相对位置

    # ---------- 基本接口 ----------
    def __len__(self) -> int:
        # 一个“epoch”的步数：训练模式按“当前可训练量”估计；困难题模式按“剩余困难题”估计
        if self.mode == 'train':
            total = len(self.learning) + len(self.mastered)
        else:  # 'hard'
            total = len(self.hard) - self._hard_read_ptr
        return max(1, (total + self.batch_size - 1) // self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        if self.mode == 'train':
            yield from self._iter_train_mode()
        else:
            yield from self._iter_hard_mode()

        # ---------- 训练模式：优先从“正在学”取，不足用“已学会”补 ----------
    def _iter_train_mode(self) -> Iterator[List[int]]:


        # 目标配额
        batch: List[int] = []
        want_mastered = int(self.review_rate * self.batch_size)
        want_learning = self.batch_size - want_mastered

        print("mastered_len:",len(self.mastered))
        print("learning_len:",len(self.learning))
        print("hard_len:",len(self.hard))
        print("stage:",self.stage)

        # print("mastered:",self.mastered)
        # print("learning:",self.learning)
        # print("hard:",self.hard)

        # 1) 先从“已学会”队头取到目标配额
        take_mastered = min(want_mastered, len(self.mastered))
        for _ in range(take_mastered):
            batch.append(self.mastered.popleft())

        # 2) 再从“正在学”队头取到目标配额
        take_learning = min(want_learning, len(self.learning))
        for _ in range(take_learning):
            batch.append(self.learning.popleft())

        # 3) 若配额不足，互相借：优先把 batch 补满

        while len(batch) < self.batch_size and self.mastered:
            batch.append(self.mastered.popleft())

        while len(batch) < self.batch_size and self.learning:
            batch.append(self.learning.popleft())

        assert len(batch) == self.batch_size
        yield batch

    # def _iter_train_mode(self) -> Iterator[List[int]]:
    #     batch: List[int] = []

    #     # 可用数量检查
    #     if len(self.mastered) + len(self.learning) < self.batch_size:
    #         raise RuntimeError(
    #             f"Not enough available samples: mastered={len(self.mastered)}, "
    #             f"learning={len(self.learning)}, need batch_size={self.batch_size}"
    #         )

    #     # 1) 随机配额：从 0..batch_size 均匀选一个给 mastered，其余给 learning
    #     k_mastered = random.randint(0, self.batch_size)
    #     k_learning = self.batch_size - k_mastered

    #     print("mastered_len:",len(self.mastered))
    #     print("learning_len:",len(self.learning))
    #     print("hard_len:",len(self.hard))
    #     print("stage:",self.stage)

    #     # 2) 先按随机配额从队头取
    #     take_mastered = min(k_mastered, len(self.mastered))
    #     for _ in range(take_mastered):
    #         batch.append(self.mastered.popleft())

    #     take_learning = min(k_learning, len(self.learning))
    #     for _ in range(take_learning):
    #         batch.append(self.learning.popleft())

    #     # 3) 若不足，随机决定“借用顺序”并把 batch 补满
    #     sources = [self.mastered, self.learning]
    #     random.shuffle(sources)  # 借用顺序也随机
    #     for dq in sources:
    #         while len(batch) < self.batch_size and dq:
    #             batch.append(dq.popleft())

    #     if len(batch) != self.batch_size:
    #         raise RuntimeError(
    #             f"Failed to fill a full batch: got {len(batch)}, need {self.batch_size}"
    #         )

    #     yield batch

    # def _iter_train_mode(self) -> Iterator[List[int]]:
    #     batch: List[int] = []

    #     self.hard_indices_tmp = [] #用来记录哪些是从hard中取出来
    #     # 可用数量检查
    #     if len(self.learning) + len(self.hard) < self.batch_size:
    #         raise RuntimeError(
    #             f"Not enough available samples: learning={len(self.learning)}, "
    #             f"hard={len(self.hard)}, need batch_size={self.batch_size}"
    #         )

    #     # 1) 随机配额：在 0..batch_size 均匀选一个给 learning，其余给 hard
    #     k_learning = random.randint(0, self.batch_size)
    #     k_hard = self.batch_size - k_learning

    #     print("learning_len:", len(self.learning))
    #     print("hard_len:", len(self.hard))
    #     print("stage:", self.stage)

    #     # 2) 先按随机配额从队头取
    #     take_learning = min(k_learning, len(self.learning))
    #     for _ in range(take_learning):
    #         batch.append(self.learning.popleft())

    #     take_hard = min(k_hard, len(self.hard))
    #     for _ in range(take_hard):
    #         indices = self.hard.popleft()
    #         self.hard_indices_tmp.append(indices)
    #         batch.append(indices)

    #     # 3) 若不足，从hrad中并补满
    #     while len(batch) < self.batch_size and self.hard:
    #         indices = self.hard.popleft()
    #         self.hard_indices_tmp.append(indices)
    #         batch.append(indices)

    #     if len(batch) != self.batch_size:
    #         raise RuntimeError(
    #             f"Failed to fill a full batch: got {len(batch)}, need {self.batch_size}"
    #         )

    #     yield batch

    # ---------- 困难题模式：按顺序读困难题（不修改队列） ----------
    def _iter_hard_mode(self) -> Iterator[List[int]]:
        n = len(self.hard)
        start = self._hard_read_ptr
        end = n
        batch = [self.hard[i] for i in range(start, end)]
        yield batch

    # ---------- 训练后回写：按准确率把样本放回相应队列尾 ----------
    def update_with_rewards(
        self,
        indices: Sequence[int],
        rewards: Sequence[float],
    ) -> None:
        """
        输入：上一批训练的样本 indices 和它们的 reward（可视作正确率/得分）
        逻辑：reward >= 阈值 → 放入“已学会”队尾；否则 → 放入“正在学”队尾
        """
        assert len(indices) == len(rewards)
        if len(indices) != self.batch_size:
            breakpoint()
        assert len(indices) == self.batch_size
        for idx, r in zip(indices, rewards):
            idx = int(idx)
            # if idx in self.hard_indices_tmp:
            #     self.hard.append(idx)
            #     continue
            # 队列语义：训练时取走了“正在学/已学会”的样本，因此这里只管“归还到队尾”
            if float(r) >= self.acc_threshold:
                self.mastered.append(idx)
            else:
                self.learning.append(idx)

    # ---------- 阶段推进判断 + 提升 ----------
    def should_advance_stage(self) -> bool:
        if len(self.hard)==0:
            return False
        if self.stage == len(self.trunc_len_list) - 1:
            return False
        """触发条件：正在学 < 0.5*初始正在学，或 < 一个 batch_size"""
        if (len(self.learning) < self.initial_learning_size // 2):
            print(f"转阶段:初始learning大小:{self.initial_learning_size}  现在大小:{len(self.learning)}")
        return (self.stage==0) or (len(self.learning) < self.initial_learning_size // 2)

    def advance_stage(self) -> None:
        """阶段 +1，截断长度增加"""
        self.stage += 1
        self.current_trunc_len = self.trunc_len_list[self.stage]

    # ---------- 把困难题的一部分放入正在学（在加入时锁定该样本的最大输出长度） ----------
    def move_hard_to_learning(self, indices: Sequence[int]) -> None:
        """
        由外部（根据 rollout 成功）指定哪些困难题加入正在学。
        语义：保持队列操作——从“困难题”中移除，并 append 到“正在学”尾部；
             同时锁定这些样本的 max 输出长度为“当前阶段截断长度”。
        """
        to_add = set(int(i) for i in indices)
        if not to_add:
            return

        # 从 hard 队列头到尾遍历，遇到命中的移除；其余保持原顺序
        new_hard = deque()
        while self.hard:
            x = self.hard.popleft()
            if x in to_add:
                self.learning.append(x)
                # 加入时刻固定该样本的最大输出长度
                assert x not in self.stage_map
                self.stage_map[x] = self.stage
            else:
                new_hard.append(x)
        self.hard = new_hard
        # random.shuffle(self.learning) #随机一下
        self.initial_learning_size = len(self.learning)

        #learning中的长度全部设置为新的阶段的长度
        # for x in self.learning:
            # self.stage_map[x] = self.stage


    # ---------- 模式切换与长度查询 ----------
    def set_mode(self, mode: Literal['train', 'hard']) -> None:
        """
        切换 DataLoader 读取模式：
          - 'train'：从“正在学+已学会”采样
          - 'hard'：读取“困难题”供 rollout（只读，不修改队列）
        """
        if mode == 'hard':
            self._hard_read_ptr = 0  # 重置只读游标
        self.mode = mode


    def get_max_lengths(self, indices: Sequence[int]) -> List[int]:
        """
        返回每个样本当前应使用的“最大输出长度”：
        - 若样本在加入“正在学”的时刻已锁定 → 返回锁定值
        - 否则 → 返回当前阶段的默认截断长度
        """
        out = []
        for i in indices:
            if  self.mode == 'train':
                if i in self.stage_map:
                    add_stage = self.stage_map[i]
                else:
                    add_stage = self.stage
                max_len = self.trunc_len_list[add_stage]
            else :
                max_len = self.trunc_len_list[self.stage + 1]
            out.append(max_len)
        return out


    def state_dict(self):
        """将全部训练/困难题队列与进度序列化，便于 DataLoader 或你手动保存。"""
        return {
            # 固定超参
            "dataset_len": self.dataset_len,
            "batch_size": self.batch_size,
            "acc_threshold": self.acc_threshold,
            "review_rate": self.review_rate,
            "trunc_len_list": list(self.trunc_len_list),

            # 训练进度相关
            "learning": list(self.learning),
            "mastered": list(self.mastered),
            "hard": list(self.hard),
            "initial_learning_size": self.initial_learning_size,

            # 阶段 & 截断
            "stage": self.stage,
            "current_trunc_len": self.current_trunc_len,
            "stage_map": dict(self.stage_map),  # {sample_idx: stage_when_locked}

            # 模式与只读游标
            "mode": self.mode,
            "_hard_read_ptr": self._hard_read_ptr,
        }

    def load_state_dict(self, state):
        """从 state 恢复全部内部状态。"""
        # 固定超参（通常不改，但为防配置不一致，这里仍以 state 为准）
        self.dataset_len = int(state["dataset_len"])
        self.batch_size = int(state["batch_size"])
        self.acc_threshold = float(state["acc_threshold"])
        self.review_rate = float(state["review_rate"])
        self.trunc_len_list = list(state["trunc_len_list"])
        # current_trunc_len 可由 stage 推导，但为兼容性，也一并恢复
        self.current_trunc_len = state["current_trunc_len"]

        # 队列与规模
        from collections import deque
        self.learning = deque(int(x) for x in state["learning"])
        self.mastered = deque(int(x) for x in state["mastered"])
        self.hard = deque(int(x) for x in state["hard"])
        self.initial_learning_size = int(state["initial_learning_size"])

        # 阶段/映射
        self.stage = int(state["stage"])
        self.stage_map = {int(k): int(v) for k, v in state["stage_map"].items()}

        # 模式 & 游标
        self.mode = state["mode"]
        self._hard_read_ptr = int(state["_hard_read_ptr"])
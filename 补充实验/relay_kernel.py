"""文档第 1—8 节的全部模型：任务超图、关键性、打击、恢复和韧性。

本文件不读写实验文件，不依赖绘图；实验只需导入 Config 和 UAVModel。
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from relay_definition import ALLOW_MATRIX, BIT_MATRIX, STATE_COUNT, ROLE_CODES, TRANSITIONS, RELATION_BITS as RELAY_BITS, MINIMUM_HOPS, NONTERMINAL_ROLES, UNRESTRICTED_MASK
from dataclasses import dataclass, field
from itertools import combinations

import networkx as nx
import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda func: func


@njit(cache=True)
def _task_supported(starts, ends, requirement, candidates, bits, sizes, max_hops):
    """角色/剩余关系状态的最早到达 BFS；只排除数学上不可能成功的搜索。"""
    N = sizes.shape[0]
    seen = np.zeros((N, STATE_COUNT), dtype=np.bool_)
    queue = np.zeros((N * STATE_COUNT, 3), dtype=np.int32)
    tail = 0
    for start in starts:
        queue[tail, 0], queue[tail, 1] = start, requirement
        seen[start, requirement] = True
        tail += 1
    head = 0
    while head < tail:
        current, remaining, distance = queue[head]
        head += 1
        if distance >= max_hops:
            continue
        for j in range(sizes[current, remaining]):
            target = candidates[current, remaining, j]
            rest = remaining & ~bits[current, remaining, j]
            if rest == 0 and ends[target]:
                return True
            if not seen[target, rest]:
                queue[tail, 0], queue[tail, 1], queue[tail, 2] = target, rest, distance + 1
                tail += 1
                seen[target, rest] = True
    return False


@njit(cache=True)
def _walk(neighbors, degrees, roles, starts_mask, ends_mask, requirements, minimum,
          draws, lam, collect, reuse_mask=None, reuse_efficiency=None):
    """与参考搜索使用相同的候选顺序、随机数和累计权重；只加速搜索循环。"""
    M, K, length = draws.shape
    N = len(roles)
    success = np.zeros((M, K), dtype=np.float64)
    paths = np.full((M, K if collect else 0, length), -1, dtype=np.int32)
    lengths = np.zeros((M, K if collect else 0), dtype=np.int32)
    p = np.zeros(M)
    supported = np.zeros(M, dtype=np.bool_)
    for k in range(M):
        starts = np.where(starts_mask[k])[0]
        if len(starts) == 0 or not np.any(ends_mask[k]):
            continue
        if reuse_mask is not None and not np.any(reuse_mask[k]):
            success[k] = reuse_efficiency[k]
            p[k] = np.count_nonzero(success[k]) / K
            supported[k] = True  # 批量功能查询不使用该保守标记提前截断。
            continue
        # 候选分布仅依赖任务、当前位置和未满足关系。
        width = neighbors.shape[1]
        candidates = np.full((N, STATE_COUNT, width), -1, dtype=np.int32)
        bits = np.zeros((N, STATE_COUNT, width), dtype=np.int32)
        cdf = np.zeros((N, STATE_COUNT, width))
        sizes = np.zeros((N, STATE_COUNT), dtype=np.int32)
        for current in range(N):
            for remaining in range(STATE_COUNT):
                if remaining & ~requirements[k]:
                    continue
                total, count = 0.0, 0
                for q in range(degrees[current]):
                    target = neighbors[current, q]
                    a, b = roles[current], roles[target]
                    if not ALLOW_MATRIX[a, b]:
                        continue
                    if not (starts_mask[k, target] or UNRESTRICTED_MASK[b] or (remaining == 0 and ends_mask[k, target])):
                        continue
                    bit = BIT_MATRIX[a, b]
                    total += lam if (remaining & bit) or (remaining == 0 and ends_mask[k, target]) else 1.0
                    candidates[current, remaining, count] = target
                    bits[current, remaining, count] = bit
                    cdf[current, remaining, count] = total
                    count += 1
                sizes[current, remaining] = count
        if not collect:
            supported[k] = _task_supported(starts, ends_mask[k], requirements[k], candidates, bits, sizes, length - 1)
            if not supported[k]:
                continue
        for r in range(K):
            if reuse_mask is not None and not reuse_mask[k, r]:
                success[k, r] = reuse_efficiency[k, r]
                if success[k, r] > 0:
                    p[k] += 1 / K
                continue
            current = starts[min(int(draws[k, r, 0] * len(starts)), len(starts) - 1)]
            remaining = requirements[k]
            if collect:
                paths[k, r, 0], lengths[k, r] = current, 1
            for hop in range(1, length):
                count = sizes[current, remaining]
                if count == 0:
                    break
                threshold = draws[k, r, hop] * cdf[current, remaining, count - 1]
                selected = 0
                while selected < count - 1 and cdf[current, remaining, selected] <= threshold:
                    selected += 1
                target = candidates[current, remaining, selected]
                remaining &= ~bits[current, remaining, selected]
                current = target
                if collect:
                    paths[k, r, hop], lengths[k, r] = current, hop + 1
                if remaining == 0 and ends_mask[k, current]:
                    p[k] += 1 / K
                    success[k, r] = minimum[k] / hop
                    break
        if collect:
            supported[k] = p[k] > 0 or _task_supported(starts, ends_mask[k], requirements[k], candidates, bits, sizes, length - 1)
    return p, success, paths, lengths, supported


@dataclass
class Group:
    count: int = 0
    efficiency_sum: float = 0.0
    links: dict = field(default_factory=dict)  # link -> [count, efficiency_sum]


@njit(cache=True)
def _scfc_kernel(count, efficiency, removed, removed_eff, overlap):
    """每列一个失效对象；功能贡献最后统一除以 MK。"""
    G, A = removed.shape
    output = np.zeros(A)
    for a in range(A):
        alive = count - removed[:, a]
        total_alive = alive.sum()
        loss = removed_eff[:, a].sum()
        if total_alive > 0:
            eta = np.zeros(G)
            for j in range(G):
                if alive[j] > 0:
                    eta[j] = (efficiency[j] - removed_eff[j, a]) / alive[j]
            for i in range(G):
                if removed[i, a] == 0:
                    continue
                denominator, numerator = 0.0, 0.0
                for j in range(G):
                    weight = alive[j] * overlap[i, j]
                    denominator += weight
                    numerator += weight * eta[j]
                if denominator > 0:
                    loss -= removed[i, a] * numerator / denominator
                else:
                    loss -= removed[i, a] * np.dot(alive, eta) / total_alive
        output[a] = loss
    return output


@njit(cache=True)
def _aggregate_task(paths, lengths, efficiencies, edge_ids):
    """把逐次轨迹归并为成员组和去重链路骨架；128节点以上使用通用归并。"""
    K, H = paths.shape
    group_ids = {(np.uint64(0), np.uint64(0)): -1}
    group_ids.clear()
    skeleton_ids = {(0, 0): -1}
    skeleton_ids.clear()
    masks = np.zeros((K, 2), dtype=np.uint64)
    counts, eta = np.zeros(K, dtype=np.int64), np.zeros(K)
    skeleton = np.zeros((K * H, 3), dtype=np.int64)
    skeleton_eta = np.zeros(K * H)
    ng, ns = 0, 0
    for r in range(K):
        length = lengths[r]
        if length == 0:
            continue
        lo, hi = np.uint64(0), np.uint64(0)
        for h in range(length):
            node = paths[r, h]
            if node < 64:
                lo |= np.uint64(1) << np.uint64(node)
            else:
                hi |= np.uint64(1) << np.uint64(node - 64)
        key = (lo, hi)
        if key not in group_ids:
            group_ids[key] = ng
            masks[ng, 0], masks[ng, 1] = lo, hi
            ng += 1
        g = group_ids[key]
        counts[g] += 1
        eta[g] += efficiencies[r]
        used = np.full(H, -1, dtype=np.int32)
        nu = 0
        for h in range(1, length):
            edge = edge_ids[paths[r, h - 1], paths[r, h]]
            seen = False
            for j in range(nu):
                if used[j] == edge:
                    seen = True
                    break
            if seen:
                continue
            used[nu] = edge
            nu += 1
            skey = (g, edge)
            if skey not in skeleton_ids:
                skeleton_ids[skey] = ns
                skeleton[ns, 0], skeleton[ns, 1] = g, edge
                ns += 1
            j = skeleton_ids[skey]
            skeleton[j, 2] += 1
            skeleton_eta[j] += efficiencies[r]
    return masks[:ng], counts[:ng], eta[:ng], skeleton[:ns], skeleton_eta[:ns]


@njit(cache=True)
def _batch_geometry(edges, failed, damaged, coverage_s, coverage_i, width):
    N = len(failed)
    neighbors = np.full((N, width), -1, dtype=np.int32)
    degrees = np.zeros(N, dtype=np.int32)
    for j in range(len(edges)):
        a, c = edges[j]
        if damaged[j] or failed[a] or failed[c]:
            continue
        neighbors[a, degrees[a]], neighbors[c, degrees[c]] = c, a
        degrees[a] += 1
        degrees[c] += 1
    starts, ends = coverage_s.copy(), coverage_i.copy()
    for i in range(N):
        if failed[i]:
            starts[:, i], ends[:, i] = False, False
    return neighbors, degrees, starts, ends


@njit(cache=True)
def _function_batch(edges, roles, coverage_s, coverage_i, requirements, minimum,
                    failed_nodes, damaged_edges, draws, lam, width):
    """候选状态批处理；每个候选仍逐次执行完全相同的配对搜索。"""
    B, N = failed_nodes.shape
    M, K, _ = draws.shape
    efficiency = np.zeros((B, M, K))
    if B == 1:
        neighbors, degrees, starts, ends = _batch_geometry(edges, failed_nodes[0], damaged_edges[0], coverage_s, coverage_i, width)
        _, eta, _, _, _ = _walk(neighbors, degrees, roles, starts, ends, requirements, minimum, draws, lam, False)
        efficiency[0] = eta
        return efficiency
    # 各候选的共同受损子图：所有失效集合的并集。搜索只会因恢复而新增候选。
    ref_failed, ref_damaged = np.zeros(N, dtype=np.bool_), np.zeros(len(edges), dtype=np.bool_)
    for b in range(B):
        ref_failed |= failed_nodes[b]
        ref_damaged |= damaged_edges[b]
    neighbors, degrees, starts, ref_ends = _batch_geometry(edges, ref_failed, ref_damaged, coverage_s, coverage_i, width)
    _, ref_eta, ref_paths, ref_lengths, _ = _walk(neighbors, degrees, roles, starts, ref_ends,
        requirements, minimum, draws, lam, True)
    for b in range(B):
        neighbors, degrees, starts, ends = _batch_geometry(edges, failed_nodes[b], damaged_edges[b], coverage_s, coverage_i, width)
        restored = ref_failed & ~failed_nodes[b]
        affected = restored.copy()
        for j in range(len(edges)):
            a, c = edges[j]
            if restored[a] or restored[c] or (ref_damaged[j] and not damaged_edges[b, j]):
                affected[a], affected[c] = True, True
        mask = np.zeros((M, K), dtype=np.bool_)
        for k in range(M):
            changed_starts = False
            for i in range(N):
                if restored[i] and coverage_s[k, i]:
                    changed_starts = True
            changed_empty_ends = not np.any(ref_ends[k]) and np.any(ends[k])
            for r in range(K):
                if changed_starts or changed_empty_ends:
                    mask[k, r] = True
                else:
                    for h in range(ref_lengths[k, r]):
                        if affected[ref_paths[k, r, h]]:
                            mask[k, r] = True
                            break
        p, eta, paths, lengths, possible = _walk(neighbors, degrees, roles, starts, ends,
                                               requirements, minimum, draws, lam, False, mask, ref_eta)
        efficiency[b] = eta
    return efficiency

try:  # 同时支持从 UAV 根目录导入及 python -m UAV...。
    from UAV.config import Config
except ImportError:
    from UAV.config import Config

from relay_definition import TASK_NAMES
ALLOWED = TRANSITIONS
RELATION_BITS = RELAY_BITS
Edge = tuple[int, int]


def canonical_edge(i: int, j: int) -> Edge:
    """无向链路以小编号在前表示。"""
    return (min(i, j), max(i, j))


@dataclass(frozen=True)
class Task:
    task_id: int
    position: tuple[float, float]
    task_type: int
    requirements: int  # 三个比特分别表示 S→S、D→D、D→S。

    @property
    def minimum_hops(self):
        return MINIMUM_HOPS[self.requirements]


@dataclass
class NetworkState:
    failed_nodes: set[int] = field(default_factory=set)
    damaged_edges: set[Edge] = field(default_factory=set)

    def copy(self):
        return NetworkState(set(self.failed_nodes), set(self.damaged_edges))

    def key(self):
        return (tuple(sorted(self.failed_nodes)), tuple(sorted(self.damaged_edges)))


@dataclass
class Evaluation:
    """统计和超图分开保存，超边只使用无序节点集合及次数。"""
    search_count: int
    p_task: np.ndarray
    f_task: np.ndarray
    p_type: np.ndarray
    f_type: np.ndarray
    P: float
    E: float
    F: float
    hyperedge_counts: Counter
    task_combinations: list[Counter]
    link_counts: Counter
    collected: bool
    groups: list[dict] = field(default_factory=list)
    feasible_tasks: np.ndarray | None = None

    @property
    def hyperedge_weights(self):
        denominator = len(self.p_task) * self.search_count
        return {e: count / denominator for e, count in self.hyperedge_counts.items()}


@dataclass
class Scores:
    hyperdegree: np.ndarray
    s: np.ndarray
    c: np.ndarray
    b: np.ndarray
    u: dict[Edge, float]
    h: dict[Edge, float]
    concentration: np.ndarray  # 没有成功组合的任务为 NaN，表示未定义。
    degree: np.ndarray
    betweenness: np.ndarray
    edge_betweenness: dict[Edge, float]


@dataclass(frozen=True)
class RecoveryAction:
    kind: str
    target: int | Edge

    def key(self):
        # 固定并列规则：节点优先，然后按节点编号或链路端点编号。
        return (0, int(self.target), -1) if self.kind == "node" else (1, *self.target)


def relative(value: float, baseline: float):
    """基准为零时返回 NaN；不裁剪超过 1 的保持率。"""
    return float(value / baseline) if baseline > 0 else float("nan")


class UAVModel:
    def __init__(self, config: Config, scene_seed: int | None = None):
        self.config = config.validate()
        self.scene_seed = config.seed if scene_seed is None else scene_seed
        rng = np.random.default_rng(self.scene_seed)
        self.positions = rng.uniform(0, config.region_size, (config.node_count, 2))
        # 随机打乱角色编号，避免节点编号与部署区域或某一种角色关联。
        self.roles = np.array(["S"] * config.n_s + ["D"] * config.n_d + ["I"] * config.n_i)
        rng.shuffle(self.roles)
        distance = np.linalg.norm(self.positions[:, None] - self.positions[None, :], axis=2)
        self.initial_edges = tuple((int(i), int(j)) for i, j in
                                   zip(*np.where(np.triu(distance <= config.communication_radius, 1))))
        self.edge_set = set(self.initial_edges)
        self.edge_index = {e: n for n, e in enumerate(self.initial_edges)}
        task_positions = rng.uniform(0, config.region_size, (config.task_count, 2))
        types = np.repeat(np.arange(5), config.task_count // 5)
        rng.shuffle(types)
        self.tasks = []
        for k, (pos, task_type) in enumerate(zip(task_positions, types)):
            requirement = (0, 1, 2, 4, 0)[task_type]
            if task_type == 4:
                requirement = int(sum(rng.choice([1, 2, 4], config.composite_requirement_count, replace=False)))
            self.tasks.append(Task(k, tuple(map(float, pos)), int(task_type), requirement))
        self.coverage_s, self.coverage_i = [], []
        for task in self.tasks:
            d = np.linalg.norm(self.positions - task.position, axis=1)
            self.coverage_s.append(set(map(int, np.flatnonzero((self.roles == "S") & (d <= config.sensing_radius)))))
            self.coverage_i.append(set(map(int, np.flatnonzero((self.roles == "I") & (d <= config.execution_radius)))))

    def validate_state(self, state: NetworkState):
        if not state.failed_nodes <= set(range(self.config.node_count)):
            raise ValueError("失效节点包含未知编号")
        if not state.damaged_edges <= self.edge_set:
            raise ValueError("直接受损链路必须来自 E0，且端点按升序排列")

    def set_positions(self, positions):
        """实验改变部署方式时重建几何网络，保留角色、任务库与搜索规则。"""
        positions = np.asarray(positions, dtype=float)
        if positions.shape != self.positions.shape or not np.isfinite(positions).all():
            raise ValueError("部署坐标必须为有限的 (N,2) 数组")
        if np.any((positions < 0) | (positions > self.config.region_size)):
            raise ValueError("部署坐标超出区域")
        self.positions = positions.copy()
        distance = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
        self.initial_edges = tuple((int(i), int(j)) for i, j in
                                   zip(*np.where(np.triu(distance <= self.config.communication_radius, 1))))
        self.edge_set = set(self.initial_edges)
        self.edge_index = {e: n for n, e in enumerate(self.initial_edges)}
        self.coverage_s, self.coverage_i = [], []
        for task in self.tasks:
            d = np.linalg.norm(positions - task.position, axis=1)
            self.coverage_s.append(set(map(int, np.flatnonzero((self.roles == "S") & (d <= self.config.sensing_radius)))))
            self.coverage_i.append(set(map(int, np.flatnonzero((self.roles == "I") & (d <= self.config.execution_radius)))))

    def graph(self, state: NetworkState | None = None):
        state = NetworkState() if state is None else state
        self.validate_state(state)
        graph = nx.Graph()
        graph.add_nodes_from(i for i in range(self.config.node_count) if i not in state.failed_nodes)
        graph.add_edges_from(e for e in self.initial_edges if e not in state.damaged_edges
                             and not state.failed_nodes.intersection(e))
        return graph

    def random_inputs(self, seed: int, search_count: int | None = None):
        """按任务、重复搜索、跳数预生成随机数，提前终止不改变其他搜索输入。"""
        K = self.config.search_count if search_count is None else search_count
        if not isinstance(K, int) or K <= 0:
            raise ValueError("搜索次数必须为正整数")
        return np.random.default_rng(seed).random((len(self.tasks), K, self.config.max_hops + 1))

    def evaluate(self, state: NetworkState | None = None, *, seed: int = 0,
                 search_count: int | None = None, random_inputs=None, collect: bool = True,
                 on_success=None):
        graph = self.graph(state)
        draws = self.random_inputs(seed, search_count) if random_inputs is None else np.asarray(random_inputs)
        if (draws.ndim != 3 or draws.shape[0] != len(self.tasks)
                or draws.shape[2] != self.config.max_hops + 1 or draws.shape[1] < 1
                or not np.isfinite(draws).all() or np.any((draws < 0) | (draws >= 1))):
            raise ValueError("随机输入必须为 [0,1) 内的 (M,K,L_max+1) 数组")
        if search_count is not None and draws.shape[1] != search_count:
            raise ValueError("随机输入 K 与 search_count 不一致")
        n, M, K = self.config.node_count, len(self.tasks), draws.shape[1]
        width = max((graph.degree(i) for i in graph), default=0)
        neighbors = np.full((n, width), -1, dtype=np.int32)
        degrees = np.zeros(n, dtype=np.int32)
        for i in graph:
            row = sorted(graph[i])
            neighbors[i, :len(row)], degrees[i] = row, len(row)
        starts, ends = np.zeros((M, n), bool), np.zeros((M, n), bool)
        for k in range(M):
            starts[k, list(self.coverage_s[k] & set(graph))] = True
            ends[k, list(self.coverage_i[k] & set(graph))] = True
        p_task, efficiency, paths, lengths, feasible = _walk(
            neighbors, degrees, np.array([ROLE_CODES[r] for r in self.roles]),
            starts, ends, np.array([t.requirements for t in self.tasks]),
            np.array([t.minimum_hops for t in self.tasks]), draws,
            self.config.priority_weight, collect or on_success is not None)
        f_task = efficiency.mean(axis=1)
        groups, records = [dict() for _ in self.tasks], [Counter() for _ in self.tasks]
        hyperedges, links = Counter(), Counter()
        if collect and on_success is None and n <= 128:
            edge_ids = np.full((n, n), -1, dtype=np.int32)
            for i, (a, b) in enumerate(self.initial_edges):
                edge_ids[a, b] = edge_ids[b, a] = i
            for k in range(M):
                masks, counts, eta, skeleton, skeleton_eta = _aggregate_task(
                    paths[k], lengths[k], efficiency[k], edge_ids)
                group_list, members_list = [], []
                for j in range(len(counts)):
                    members = []
                    for part, offset in zip(masks[j], (0, 64)):
                        bits = int(part)
                        while bits:
                            bit = bits & -bits
                            members.append(offset + bit.bit_length() - 1)
                            bits -= bit
                    members = frozenset(members)
                    group = Group(int(counts[j]), float(eta[j]), {})
                    group_list.append(group)
                    members_list.append(members)
                    groups[k][members] = group
                    if eta[j] > 0:
                        hyperedges[members] += int(counts[j])
                        records[k][members] += int(counts[j])
                for row, e_sum in zip(skeleton, skeleton_eta):
                    j, edge, count = map(int, row)
                    link = self.initial_edges[edge]
                    group_list[j].links[link] = [count, float(e_sum)]
                    if group_list[j].efficiency_sum > 0:
                        links[link] += count
        elif collect or on_success is not None:
            for k in range(M):
                for r in range(K):
                    path = tuple(map(int, paths[k, r, :lengths[k, r]]))
                    if not path:
                        continue  # 没有感知端或执行端的搜索不产生协同组。
                    eta = float(efficiency[k, r])
                    if eta > 0 and on_success is not None:
                        on_success(k, r, path)
                    if not collect:
                        continue
                    members = frozenset(path)
                    used = {canonical_edge(a, b) for a, b in zip(path, path[1:])}
                    group = groups[k].setdefault(members, Group())
                    group.count += 1
                    group.efficiency_sum += eta
                    for edge in used:
                        skeleton = group.links.setdefault(edge, [0, 0.0])
                        skeleton[0] += 1
                        skeleton[1] += eta
                    if eta > 0:
                        hyperedges[members] += 1
                        records[k][members] += 1
                        links.update(used)
        types = np.array([task.task_type for task in self.tasks])
        p_type = np.array([p_task[types == t].mean() if np.any(types == t) else np.nan for t in range(5)])
        f_type = np.array([f_task[types == t].mean() if np.any(types == t) else np.nan for t in range(5)])
        P, F = float(p_task.mean()), float(f_task.mean())
        return Evaluation(K, p_task, f_task, p_type, f_type, P, F / P if P else 0., F,
                          hyperedges, records, links, collect, groups, feasible)

    def scfc(self, evaluation: Evaluation, *, alpha: float = 3., reroute: bool = True,
             include_failed: bool = True, kind: str | None = None):
        """文档 6.2–6.5；同时支持节点、实际链路骨架、未闭合组及三种消融。"""
        if not evaluation.collected or not evaluation.groups:
            raise ValueError("SCFC 需要包含协同组及链路骨架的 evaluate(collect=True) 结果")
        if alpha < 0 or not np.isfinite(alpha):
            raise ValueError("alpha 必须为非负有限值")
        if kind not in (None, "node", "edge"):
            raise ValueError("kind 必须为 node、edge 或 None")
        n, E = self.config.node_count, len(self.initial_edges)
        output = np.zeros(n + E)
        for record in evaluation.groups:
            entries = [(g, v) for g, v in record.items() if include_failed or v.efficiency_sum > 0]
            if not entries:
                continue
            G = len(entries)
            count = np.array([v.count for _, v in entries], dtype=float)
            efficiency = np.array([v.efficiency_sum for _, v in entries])
            removed, removed_eff = np.zeros((G, n + E)), np.zeros((G, n + E))
            members = np.zeros((G, n))
            for i, (g, v) in enumerate(entries):
                members[i, [j for j in g if self.roles[j] in NONTERMINAL_ROLES]] = 1
                if kind != "edge":
                    removed[i, list(g)] = v.count
                    removed_eff[i, list(g)] = v.efficiency_sum
                if kind != "node":
                    for edge, (c, e) in v.links.items():
                        removed[i, n + self.edge_index[edge]] = c
                        removed_eff[i, n + self.edge_index[edge]] = e
            if not reroute:
                output += removed_eff.sum(axis=0)
                continue
            used = np.flatnonzero(removed.sum(axis=0))
            overlap = (members @ members.T) ** alpha  # NumPy 约定 0**0=1。
            output[used] += _scfc_kernel(count, efficiency, removed[:, used], removed_eff[:, used], overlap)
        output /= len(self.tasks) * evaluation.search_count
        return output[:n], dict(zip(self.initial_edges, output[n:]))

    def evaluate_functions(self, states, *, random_inputs):
        """批量计算恢复候选的 F，与逐个 evaluate(collect=False) 等价。"""
        states = list(states)
        if not states:
            return np.empty(0)
        draws = np.asarray(random_inputs)
        M, n = len(self.tasks), self.config.node_count
        if (draws.ndim != 3 or draws.shape[0] != M or draws.shape[1] < 1
                or draws.shape[2] != self.config.max_hops + 1
                or not np.isfinite(draws).all() or np.any((draws < 0) | (draws >= 1))):
            raise ValueError("批量搜索随机输入的形状或范围无效")
        nodes = np.zeros((len(states), n), dtype=bool)
        edges = np.zeros((len(states), len(self.initial_edges)), dtype=bool)
        for j, state in enumerate(states):
            self.validate_state(state)
            nodes[j, list(state.failed_nodes)] = True
            edges[j, [self.edge_index[e] for e in state.damaged_edges]] = True
        cs, ci = np.zeros((M, n), bool), np.zeros((M, n), bool)
        for k in range(M):
            cs[k, list(self.coverage_s[k])] = True
            ci[k, list(self.coverage_i[k])] = True
        initial = np.array(self.initial_edges, dtype=np.int32).reshape(-1, 2)
        degree = np.bincount(initial.ravel(), minlength=n)
        efficiency = _function_batch(initial, np.array([ROLE_CODES[r] for r in self.roles]),
            cs, ci, np.array([t.requirements for t in self.tasks]),
            np.array([t.minimum_hops for t in self.tasks]), nodes, edges, draws,
            self.config.priority_weight, int(degree.max()))
        return efficiency.mean(axis=2).mean(axis=1)

    def evaluate_reference(self, state: NetworkState | None = None, *, seed: int = 0,
                 search_count: int | None = None, random_inputs=None, collect: bool = True,
                 on_success=None):
        """重新执行文档的有偏随机游走，固定任务库并保留全部任务分母。

        random_inputs 可跨状态复用，起点使用第 0 个数，每跳使用对应的数；
        即使可用候选集合改变，也保持同一任务、同一次搜索、同一跳的输入配对。
        on_success 为可选诊断回调，接收任务编号、搜索编号和实际有序节点链；
        默认不保存访问顺序，超边仍只保存无序节点集合。回调不参与采样决策。
        """
        graph = self.graph(state)
        if random_inputs is None:
            random_inputs = self.random_inputs(seed, search_count)
        draws = np.asarray(random_inputs)
        if draws.ndim != 3 or draws.shape[0] != len(self.tasks) or draws.shape[2] != self.config.max_hops + 1:
            raise ValueError("配对随机输入形状必须为 (M, K, L_max+1)")
        if draws.shape[1] < 1 or not np.isfinite(draws).all() or np.any((draws < 0) | (draws >= 1)):
            raise ValueError("随机输入必须是 [0, 1) 内的有限值")
        if search_count is not None and draws.shape[1] != search_count:
            raise ValueError("随机输入的 K 与 search_count 不一致")
        K = draws.shape[1]
        active = set(graph)
        neighbors = {i: tuple(sorted(graph[i])) for i in graph}
        p_task = np.zeros(len(self.tasks))
        f_task = np.zeros(len(self.tasks))
        hyperedge_counts, link_counts = Counter(), Counter()
        combinations_by_task = [Counter() for _ in self.tasks]
        roles, lam = self.roles, self.config.priority_weight
        for k, task in enumerate(self.tasks):
            starts = sorted(self.coverage_s[k] & active)
            ends = self.coverage_i[k] & active
            if not starts or not ends:
                continue
            feasible = set(starts) | {i for i in active if roles[i] in NONTERMINAL_ROLES - {"S"}}
            # 候选分布取决于当前位置和剩余关系，缓存分布不会改变采样规则。
            distributions = {}
            successes, efficiency_sum = 0, 0.0
            for r in range(K):
                row = draws[k, r]
                current = starts[min(int(row[0] * len(starts)), len(starts) - 1)]
                remaining = task.requirements
                if on_success is not None:
                    ordered_path = [current]
                if collect:
                    visited, used_links = {current}, set()
                for hop in range(1, self.config.max_hops + 1):
                    key = (current, remaining)
                    if key not in distributions:
                        candidates, bits, weights = [], [], []
                        for target in neighbors[current]:
                            transition = (roles[current], roles[target])
                            if transition not in ALLOWED:
                                continue
                            if target not in feasible and not (remaining == 0 and target in ends):
                                continue
                            bit = RELATION_BITS.get(transition, 0)
                            candidates.append(target)
                            bits.append(bit)
                            weights.append(lam if (remaining & bit) or (remaining == 0 and target in ends) else 1.0)
                        cumulative = np.cumsum(weights).tolist()
                        distributions[key] = (candidates, bits, cumulative)
                    candidates, bits, cumulative = distributions[key]
                    if not candidates:
                        break
                    selected = bisect_right(cumulative, float(row[hop]) * cumulative[-1])
                    target = candidates[selected]
                    if on_success is not None:
                        ordered_path.append(target)
                    if collect:
                        visited.add(target)
                        used_links.add(canonical_edge(current, target))
                    remaining &= ~bits[selected]
                    current = target
                    if remaining == 0 and current in ends:
                        successes += 1
                        efficiency_sum += task.minimum_hops / hop
                        if collect:
                            edge = frozenset(visited)
                            hyperedge_counts[edge] += 1
                            combinations_by_task[k][edge] += 1
                            link_counts.update(used_links)
                        if on_success is not None:
                            on_success(k, r, tuple(ordered_path))
                        break
            p_task[k] = successes / K
            f_task[k] = efficiency_sum / K
        types = np.array([task.task_type for task in self.tasks])
        p_type = np.array([p_task[types == t].mean() for t in range(5)])
        f_type = np.array([f_task[types == t].mean() for t in range(5)])
        P, F = float(p_task.mean()), float(f_task.mean())
        E = F / P if P > 0 else 0.0
        result = Evaluation(K, p_task, f_task, p_type, f_type, P, E, F,
                            hyperedge_counts, combinations_by_task, link_counts, collect)
        if collect and not np.isclose(sum(result.hyperedge_weights.values()), P):
            raise AssertionError("超边权重之和应等于任务成功率")
        return result

    def reachable_execution_nodes(self, task_id: int, state: NetworkState | None = None):
        """枚举（当前位置、剩余关系）状态，精确判断跳数限制内可达的执行端点。

        该查询不估计随机成功率，只判断是否存在合法任务链；所有合法候选权重
        为正，因此任一符合约束的路径都属于搜索规则的支持集。状态最早到达时
        的剩余跳数最多，可安全合并同一状态的更晚到达，不影响可达性。
        """
        if not isinstance(task_id, int) or not 0 <= task_id < len(self.tasks):
            raise ValueError("任务编号必须位于任务库范围内")
        graph = self.graph(state)
        active = set(graph)
        starts = self.coverage_s[task_id] & active
        ends = self.coverage_i[task_id] & active
        if not starts or not ends:
            return set()
        feasible = starts | {i for i in active if self.roles[i] in NONTERMINAL_ROLES - {"S"}}
        frontier = {(i, self.tasks[task_id].requirements) for i in starts}
        seen, reached = set(frontier), set()
        for _ in range(self.config.max_hops):
            following = set()
            for current, remaining in frontier:
                for target in graph[current]:
                    relation = (self.roles[current], self.roles[target])
                    if relation not in ALLOWED or (target not in feasible and not (remaining == 0 and target in ends)):
                        continue
                    rest = remaining & ~RELATION_BITS.get(relation, 0)
                    if rest == 0 and target in ends:
                        reached.add(target)
                    elif (target, rest) not in seen:
                        following.add((target, rest))
            if not following:
                break
            seen.update(following)
            frontier = following
        return reached

    def scores(self, evaluation: Evaluation, state: NetworkState | None = None):
        """文档第 4 节：频次、任务内替代性、集中度和普通图对照。"""
        if not evaluation.collected:
            raise ValueError("计算指标需要 collect=True 的超图构建结果")
        n, M = self.config.node_count, len(self.tasks)
        hyperdegree, s, c, b = (np.zeros(n) for _ in range(4))
        h = dict.fromkeys(self.initial_edges, 0.0)
        for edge, weight in evaluation.hyperedge_weights.items():
            nodes = sorted(edge)
            hyperdegree[nodes] += 1
            s[nodes] += weight
            for i, j in combinations(nodes, 2):
                if (i, j) in h:
                    h[(i, j)] += weight
        concentration = np.full(M, np.nan)
        for k, record in enumerate(evaluation.task_combinations):
            if not record:
                continue
            common = set.intersection(*(set(e) for e in record))
            c[list(common)] += 1 / M
            for edge in record:
                b[list(edge)] += 1 / (M * len(record))
            total = sum(record.values())
            concentration[k] = sum((count / total) ** 2 for count in record.values())
        u = {e: evaluation.link_counts[e] / (M * evaluation.search_count) for e in self.initial_edges}
        graph = self.graph(state)
        degree = np.array([graph.degree(i) if i in graph else 0 for i in range(n)], dtype=float)
        node_bc = nx.betweenness_centrality(graph, normalized=True)
        edge_bc = {canonical_edge(*e): v for e, v in nx.edge_betweenness_centrality(graph, normalized=True).items()}
        result = Scores(hyperdegree, s, c, b, u, h, concentration, degree,
                        np.array([node_bc.get(i, 0.0) for i in range(n)]),
                        {e: edge_bc.get(e, 0.0) for e in self.initial_edges})
        if np.any(c > b + 1e-12) or np.any(b > 1 + 1e-12):
            raise AssertionError("任务替代性指标应满足 0 ≤ c ≤ b ≤ 1")
        return result

    def attack_order(self, kind: str, strategy: str, scores: Scores, *, seed: int = 0,
                     state: NetworkState | None = None):
        """基准打击使用事先固定的评分；只返回当前可用对象。"""
        graph = self.graph(state)
        objects = sorted(graph.nodes if kind == "node" else (canonical_edge(*e) for e in graph.edges))
        if kind not in ("node", "edge"):
            raise ValueError("打击对象类型必须为 node 或 edge")
        if strategy == "random":
            rng = np.random.default_rng(seed)
            return [objects[int(i)] for i in rng.permutation(len(objects))]
        if kind == "node":
            if strategy == "task":
                return sorted(objects, key=lambda i: (-scores.c[i], -scores.b[i], -scores.s[i], i))
            mapping = {"degree": scores.degree, "betweenness": scores.betweenness, "strength": scores.s}
        else:
            mapping = {"betweenness": scores.edge_betweenness, "usage": scores.u, "co_participation": scores.h}
        if strategy not in mapping:
            raise ValueError(f"未知的 {kind} 打击策略：{strategy}")
        values = mapping[strategy]
        return sorted(objects, key=lambda obj: (-values[obj], obj))

    def attack(self, state: NetworkState, kind: str, targets):
        """同时打击一批可用对象；节点引起的链路中断不计入 damaged_edges。"""
        updated = state.copy()
        graph = self.graph(state)
        for obj in targets:
            if kind == "node":
                if obj not in graph:
                    raise ValueError("只能打击当前可用节点")
                updated.failed_nodes.add(int(obj))
            elif kind == "edge":
                edge = canonical_edge(*obj)
                if not graph.has_edge(*edge):
                    raise ValueError("只能打击当前可用链路")
                updated.damaged_edges.add(edge)
            else:
                raise ValueError("打击对象类型必须为 node 或 edge")
        return updated

    def loss_fractions(self, state: NetworkState):
        self.validate_state(state)
        return (len(state.failed_nodes) / self.config.node_count,
                len(state.damaged_edges) / len(self.initial_edges) if self.initial_edges else 0.0)

    def available_actions(self, state: NetworkState, node_budget: int, edge_budget: int,
                          time: int, horizon: int):
        """动作在 time 开始、time+1 完成；端点失效的链路暂不能单独修复。"""
        self.validate_state(state)
        if time + 1 > horizon:
            return []
        actions = []
        if node_budget > 0:
            actions.extend(RecoveryAction("node", i) for i in sorted(state.failed_nodes))
        if edge_budget > 0:
            actions.extend(RecoveryAction("edge", e) for e in sorted(state.damaged_edges)
                           if not state.failed_nodes.intersection(e))
        return sorted(actions, key=lambda action: action.key())

    def restore(self, state: NetworkState, action: RecoveryAction):
        updated = state.copy()
        if action.kind == "node" and action.target in state.failed_nodes:
            updated.failed_nodes.remove(action.target)
        elif action.kind == "edge" and action.target in state.damaged_edges and not state.failed_nodes.intersection(action.target):
            updated.damaged_edges.remove(action.target)
        else:
            raise ValueError("该恢复动作不可执行")
        return updated

    def choose_recovery(self, state: NetworkState, actions: list[RecoveryAction], strategy: str,
                        scores: Scores, *, rng=None, gain_inputs=None):
        """增益恢复在每个状态重估，使用独立批次并配对比较候选与当前状态。"""
        if not actions:
            return None, None
        ordered = sorted(actions, key=lambda action: action.key())
        if strategy == "random":
            rng = np.random.default_rng(0) if rng is None else rng
            return ordered[int(rng.integers(len(ordered)))], None
        if strategy == "task":
            if any(action.kind != "node" for action in actions):
                raise ValueError("任务不可替代度恢复仅适用于纯节点场景")
            return min(ordered, key=lambda a: (-scores.c[a.target], -scores.b[a.target], -scores.s[a.target], a.key())), None
        if strategy in ("strength", "co_participation"):
            def score(action):
                if action.kind == "node":
                    return scores.s[action.target]
                return (scores.h if strategy == "co_participation" else scores.u)[action.target]
            return min(ordered, key=lambda action: (-score(action), action.key())), None
        if strategy == "gain":
            if gain_inputs is None or len(gain_inputs) == 0:
                raise ValueError("增益恢复需要非空独立配对随机批次")
            base = np.array([self.evaluate(state, random_inputs=draw, collect=False).F for draw in gain_inputs])
            gains = []
            for action in ordered:
                candidate = self.restore(state, action)
                gains.append(float(np.mean([self.evaluate(candidate, random_inputs=draw, collect=False).F
                                            - old for draw, old in zip(gain_inputs, base)])))
            index = max(range(len(ordered)), key=lambda i: (gains[i], tuple(-v for v in ordered[i].key())))
            return ordered[index], gains[index]  # 收益为零或负也不人为禁止动作。
        raise ValueError(f"未知恢复策略：{strategy}")


def performance_loss(before: Evaluation, after: Evaluation):
    """保留符号，删点有时会减少无效分支并提高搜索性能。"""
    return {"delta_P": before.P - after.P, "delta_F": before.F - after.F}


def resilience(times, functions, *, attack_time: int, recovery_start: int,
               baseline_function: float, target: float = 0.9, function_threshold=None):
    """共同评价时域上计算抵抗能力、首次达标时间、梯形过程韧性和最终功能。"""
    times = np.asarray(times)
    functions = np.asarray(functions, dtype=float)
    if times.ndim != 1 or functions.shape != times.shape or len(times) < 2 or np.any(np.diff(times) != 1):
        raise ValueError("韧性评估需要连续的整数时间序列")
    if not times[0] <= attack_time <= recovery_start < times[-1]:
        raise ValueError("时间序列必须覆盖打击、恢复开始和共同终点")
    phi = functions / baseline_function if baseline_function > 0 else np.full_like(functions, np.nan)
    after_attack = times >= attack_time
    resistance = (float(np.min(phi[(times >= attack_time) & (times <= recovery_start)]))
                  if baseline_function > 0 else float("nan"))
    area = (float(np.trapezoid(phi[after_attack], times[after_attack]) / (times[-1] - attack_time))
            if baseline_function > 0 else float("nan"))
    reached = (functions >= function_threshold) if function_threshold is not None else (phi >= target)
    hits = times[(times >= recovery_start) & reached]
    return {"R_res": resistance, "T_rec": int(hits[0] - recovery_start) if len(hits) else None,
            "R_F": area, "Phi_final": float(phi[-1]), "F_final": float(functions[-1]),
            "recovered": bool(len(hits))}


def robustness(fractions, normalized_functions, *, functions=None, function_threshold=None):
    """只有实际覆盖 [0,1] 才计算全区间抗毁面积；不插补未计算的区间。"""
    fractions = np.asarray(fractions, dtype=float)
    normalized_functions = np.asarray(normalized_functions, dtype=float)
    if fractions.shape != normalized_functions.shape or fractions.ndim != 1 or not len(fractions):
        raise ValueError("损失比例和保持率必须为相同长度的一维数组")
    if np.any(~np.isfinite(fractions)) or np.any((fractions < 0) | (fractions > 1)) or np.any(np.diff(fractions) <= 0):
        raise ValueError("损失比例必须严格递增且位于 [0,1]")
    covered = np.isclose(fractions[0], 0) and np.isclose(fractions[-1], 1)
    area = float(np.trapezoid(normalized_functions, fractions)) if covered and np.isfinite(normalized_functions).all() else None
    fc = None
    if function_threshold is not None:
        values = np.asarray(functions, dtype=float)
        if values.shape != fractions.shape:
            raise ValueError("绝对功能数组形状不匹配")
        hits = fractions[values < function_threshold]
        fc = float(hits[0]) if len(hits) else None
    return {"R_rob": area, "f_c": fc, "full_interval": bool(covered)}

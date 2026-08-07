#!/usr/bin/env python3
"""EasyWar 地图拓扑分析器

三层建模：
  格层   -> 格子图（cell graph）
  战略层 -> base graph：据点=节点，不经过第三据点的直连走廊=边
  玩家层 -> 自同构轨道 / 距离谱 -> 平衡性

输出每张图的拓扑指纹 + 平衡性指标 + 库内同构检查。
"""
import re
import sys
from collections import Counter, deque

import networkx as nx


# ---------- 1. 解析地图 ----------

def extract_maps(html_path):
    s = open(html_path, encoding='utf-8').read()
    out = {}
    for mm in re.finditer(r"id:\s*'([^']+)'.*?rows:\s*\[(.*?)\]\s*,\s*\n", s, re.S):
        rid = mm.group(1)
        rows = []
        for r in mm.group(2).splitlines():
            m2 = re.match(r"\s*'([^']*)',?\s*$", r)
            if m2:
                rows.append(m2.group(1))
        out[rid] = rows
    return out


# ---------- 2. base graph 提取 ----------

def build_base_graph(rows):
    """据点=节点（id=(x,y)，支持同名多据点如 X×3/G×2）；边=不经过第三据点的直连走廊。"""
    H, W = len(rows), len(rows[0])
    bases = {}
    for y in range(H):
        for x in range(W):
            ch = rows[y][x]
            if ch.isupper():
                bases[(x, y)] = ch

    G = nx.Graph()
    for pos, ch in bases.items():
        G.add_node(pos, name=ch)

    def neighbors(p):
        x, y = p
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx_, ny = x + dx, y + dy
            if 0 <= nx_ < W and 0 <= ny < H and rows[ny][nx_] != '.':
                yield (nx_, ny)

    for start in bases:
        seen = {start}
        q = deque([start])
        while q:
            p = q.popleft()
            for n in neighbors(p):
                if n in seen:
                    continue
                seen.add(n)
                if n in bases:
                    if not G.has_edge(start, n) and start != n:
                        G.add_edge(start, n)
                    continue
                q.append(n)
    return G, bases


def names_of(G, prefix):
    """返回 base graph 中学科名为 prefix 的节点。"""
    return [n for n, d in G.nodes(data=True) if d.get('name') == prefix]


# ---------- 3. 拓扑不变量 ----------

def degree_sequence(G):
    return sorted(d for _, d in G.degree())


def cyclomatic(G):
    """环秩 rho = E - V + C（每个连通分量）。0 => 树 => 无环"""
    return G.number_of_edges() - G.number_of_nodes() + nx.number_connected_components(G)


def distance_profile(G, node):
    """节点到全图各节点的最短路距离分布（排序后 = 距离谱）。"""
    lens = nx.single_source_shortest_path_length(G, node)
    return sorted(lens.values())


def player_profiles(G, player_names):
    """每个玩家（同名节点全部取）的距离谱。"""
    out = {}
    for name in player_names:
        for n in names_of(G, name):
            out[f"{name}@{n}"] = distance_profile(G, n)
    return out


def players_symmetric(G, players):
    """玩家是否在同一自同构轨道（几何等价）。小图暴力枚举自同构。"""
    orbits = automorphism_orbits(G)
    for orb in orbits:
        if all(p in orb for p in players) and len(orb) >= len(players):
            return True
    return False


def automorphism_orbits(G):
    """自同构群轨道：枚举 G 上的所有同构映射到自身。小图（<=10 节点）可行。"""
    if G.number_of_nodes() > 12:
        return None
    matcher = nx.algorithms.isomorphism.GraphMatcher(G, G)
    nodes = list(G.nodes())
    orbits = {}
    for m in matcher.isomorphisms_iter():
        for u, v in m.items():
            orbits.setdefault(u, set()).add(v)
    if not orbits:
        return [{n} for n in nodes]
    # 合并成轨道
    merged = []
    unvisited = set(nodes)
    while unvisited:
        seed = unvisited.pop()
        orb = orbits[seed] & unvisited
        orb.add(seed)
        unvisited -= orb
        merged.append(orb)
    return merged


def wl_fingerprint(G, rounds=4):
    """Weisfeiler-Lehman 颜色细化指纹：迭代颜色 -> 颜色直方图。"""
    G2 = nx.convert_node_labels_to_integers(G)
    colors = {n: 1 for n in G2.nodes()}
    hist = [Counter(colors.values())]
    for _ in range(rounds):
        new = {}
        for n in G2.nodes():
            sig = tuple(sorted(colors[v] for v in G2[n]))
            new[n] = hash((colors[n], sig))
        colors = new
        hist.append(Counter(colors.values()))
    return tuple(sorted((k, v) for h in hist for k, v in h.items()))


def holes(rows):
    """格层拓扑洞数 = 不与地图边界连通的 void 区域数。
    这是平面图真正的 genus：ring(1 内环) vs quad(街旁缝隙) vs H(2 洞) 在此区分。"""
    H, W = len(rows), len(rows[0])
    voids = {(x, y) for y in range(H) for x in range(W) if rows[y][x] == '.'}
    if not voids:
        return 0
    seen = set()
    n_holes = 0
    for start in voids:
        if start in seen:
            continue
        comp = {start}
        q = [start]
        seen.add(start)
        touches_edge = False
        while q:
            x, y = q.pop()
            if x == 0 or y == 0 or x == W - 1 or y == H - 1:
                touches_edge = True
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (x + dx, y + dy)
                if n in voids and n not in seen:
                    seen.add(n)
                    comp.add(n)
                    q.append(n)
        if not touches_edge:
            n_holes += 1
    return n_holes


def fingerprint(G):
    """完整指纹：节点数/边数/环秩/度序列/直径/围长/连通分量/WL。"""
    n = G.number_of_nodes()
    e = G.number_of_edges()
    rho = cyclomatic(G)
    is_path = (rho == 0) and (max((d for _, d in G.degree()), default=0) <= 2)
    is_2conn = nx.is_biconnected(G) if n > 2 else False
    diams = []
    for cc in nx.connected_components(G):
        diams.append(nx.diameter(G.subgraph(cc)))
    girth = None
    for cc in nx.connected_components(G):
        g = nx.girth(G.subgraph(cc))
        if girth is None or (g is not None and g < girth):
            girth = g
    return {
        'n': n, 'e': e, 'rho': rho, 'deg': degree_sequence(G),
        'is_path': is_path, 'is_2conn': is_2conn,
        'diam': max(diams), 'girth': girth,
        'cc': nx.number_connected_components(G),
        'wl': wl_fingerprint(G),
    }


# ---------- 4. 主流程 ----------

# ---------- 4. 产能审计（BALANCE.md 数值） ----------

# 设计上限（v5 规范，见 docs/MAP_GUIDE.md）：
#   地图尺寸 <= 15x13（5 人 60 格环等分特例 16x16）
#   单据点地块 <= 10（产能 <= 4.5/s，驻军上限 <= 180）
#   据点间唯一路径（树状走廊，无环）
#   减少并排格（1 格宽走廊为主）
PRODUCTION_BASE = 2.5      # 据点基础产能（兵/秒）
PRODUCTION_PER_TILE = 0.2  # 每块关联地块 +0.2
CAP_BASE = 80              # 驻军基础上限
CAP_PER_TILE = 10          # 每块地 +10
TILE_LIMIT = 10            # 单据点地块上限
SIZE_MAX = (17, 13)        # (宽, 高) 上限（多臂星形需要）
SIZE_MAX_FFA5 = (16, 16)   # 5 人特例

# ---------- 5. 树状走廊检查（v5） ----------

def cell_graph(rows):
    """格层图：每个非 void 格 = 节点，四邻接 = 边。"""
    H, W = len(rows), len(rows[0])
    G = nx.Graph()
    cells = {}
    for y in range(H):
        for x in range(W):
            if rows[y][x] != '.':
                cells[(x, y)] = G.number_of_nodes()
                G.add_node(cells[(x, y)])
    for (x, y), i in cells.items():
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (x + dx, y + dy) in cells:
                j = cells[(x + dx, y + dy)]
                if i < j:
                    G.add_edge(i, j)
    return G, cells


def unique_path_check(rows):
    """据点间唯一最短路径：对每对据点，all_shortest_paths 数量必须 == 1。
    等价于：格层图是树（连通 + E == V - 1）时自动成立，但显式检查更严格。"""
    H, W = len(rows), len(rows[0])
    G, cells = cell_graph(rows)
    bases = [cells[(x, y)] for y in range(H) for x in range(W)
             if rows[y][x].isupper()]
    if len(bases) < 2:
        return 0, []
    viol = []
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            try:
                paths = list(nx.all_shortest_paths(G, bases[i], bases[j]))
            except nx.NetworkXNoPath:
                viol.append(f"据点 {i}<->{j} 不可达")
                continue
            if len(paths) > 1:
                viol.append(f"据点 {i}<->{j} 有 {len(paths)} 条最短路")
    return len(viol), viol


def parallel_check(rows):
    """并排格检测：统计 2x2 全地块（含据点）块数；1 格宽走廊 = 0。"""
    H, W = len(rows), len(rows[0])
    n2x2 = 0
    for y in range(H - 1):
        for x in range(W - 1):
            if all(rows[yy][xx] != '.' for yy in (y, y + 1) for xx in (x, x + 1)):
                n2x2 += 1
    # 最宽的连续横向/纵向走廊段
    max_run = 0
    for row in rows:
        run = cur = 0
        for ch in row:
            cur = cur + 1 if ch != '.' else 0
            run = max(run, cur)
        max_run = max(max_run, run)
    for x in range(W):
        run = cur = 0
        for y in range(H):
            cur = cur + 1 if rows[y][x] != '.' else 0
            run = max(run, cur)
        max_run = max(max_run, run)
    return n2x2, max_run


def tree_check(rows):
    """树状非直线：格层图是树（无环）且有分支（最大度 >= 3）。"""
    G, cells = cell_graph(rows)
    n = G.number_of_nodes()
    e = G.number_of_edges()
    is_tree = nx.is_connected(G) and e == n - 1
    max_deg = max(d for _, d in G.degree()) if n else 0
    return is_tree, max_deg, (is_tree and max_deg >= 3)


def production_audit(rows, players=None):
    """每据点地块数（同名多据点取平均）+ 产能 + 驻军上限。"""
    joined = ''.join(rows)
    from collections import Counter
    tiles = Counter(c.lower() for c in joined if c.islower())
    bases = Counter(c for c in joined if c.isupper())
    per = {k: round(v / bases.get(k.upper(), 1), 1) for k, v in tiles.items()}
    viol = []
    for b, n in per.items():
        prod = PRODUCTION_BASE + PRODUCTION_PER_TILE * n
        cap = CAP_BASE + CAP_PER_TILE * n
        if n > TILE_LIMIT:
            viol.append(f"{b}: {n} 块地/座(>{TILE_LIMIT}) -> {prod:.1f}/s, 上限{cap}")
    w, h = len(rows[0]), len(rows)
    size_ok = (w <= SIZE_MAX[0] and h <= SIZE_MAX[1]) or (w, h) == SIZE_MAX_FFA5
    if not size_ok:
        viol.append(f"尺寸 {w}x{h} 超限(>{SIZE_MAX[0]}x{SIZE_MAX[1]}, 5人特例16x16)")
    summary = {
        'size': f"{w}x{h}", 'total_tiles': sum(tiles.values()),
        'total_bases': sum(bases.values()),
        'max_per_base': max(per.values()) if per else 0,
        'size_ok': size_ok,
    }
    return summary, viol

def audit_report(rows, players=None):
    summary, viol = production_audit(rows, players)
    s = f"  {summary['size']:8s} 地块{summary['total_tiles']:>3} 据点{summary['total_bases']:>2} 单据点最多{summary['max_per_base']:>2}"
    if viol:
        s += "  !! " + "; ".join(viol)
    return s

def analyze(rows, players):
    G, bases = build_base_graph(rows)
    fp = fingerprint(G)
    hole = holes(rows)
    profiles = player_profiles(G, players)
    sym = None
    if len(profiles) >= 2:
        # 自同构轨道判断（玩家节点在轨道内等价）
        player_nodes = list(profiles)
        orbits = automorphism_orbits(G)
        if orbits is not None:
            for orb in orbits:
                if all(p in orb for p in player_nodes):
                    sym = True
                    break
            else:
                sym = False
    balanced = len(set(tuple(d) for d in profiles.values())) == 1 if profiles else None
    return G, fp, hole, sym, balanced, profiles


def main():
    html = sys.argv[1] if len(sys.argv) > 1 else 'index.html'
    maps = extract_maps(html)
    # 每张图的玩家（按 HTML 里 players 字段：脚本简单起见手传）
    players_of = {
        'h-chain': 'AM', 'y-fork': 'AM', 'ladder': 'AM',
        'tristar': 'AMC', 'crosstar': 'AMCU', 'twinstar': 'AMCUGP',
    }
    fps = {}
    print(f"{'map':10s} | {'V':>2} {'E':>2} rho hole | path tree | 2conn | 唯一路径 | 并排 | deg      | balance |")
    print("-" * 115)
    for rid, rows in maps.items():
        players = players_of.get(rid, '')
        G, fp, hole, sym, balanced, profiles = analyze(rows, players)
        fps[rid] = fp
        nv, vv = unique_path_check(rows)
        n2x2, max_run = parallel_check(rows)
        is_tree, max_deg, is_branch = tree_check(rows)
        degs = ','.join(map(str, fp['deg']))
        bal_s = 'Y' if balanced else ('N' if profiles else '-')
        note = ''
        if fp['is_path']:
            note += ' [直线!]'
        if nv:
            note += f' [多路径x{nv}!]'
        if n2x2:
            note += f' [并排{n2x2}x2x2]'
        if not balanced and profiles:
            note += ' [距离谱不等]'
        print(f"{rid:10s} | {fp['n']:>2} {fp['e']:>2} {fp['rho']:>3} {hole:>4} | {str(fp['is_path']):>5} {str(is_tree):>5} | {str(fp['is_2conn']):>5} | {nv:>4}       | {n2x2:>2}/{max_run:>2} | {degs:8s} | {bal_s:>7} |{note}")
        audit = audit_report(rows, players)
        print(f"          产能审计: {audit}")

    # 同构检查：WL 指纹相同 -> 再跑 is_isomorphic 确认
    print("\n== 同构检查（两两比较）==")
    ids = list(fps)
    any_isom = False
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if fps[a]['wl'] == fps[b]['wl']:
                Ga, _ = build_base_graph(maps[a])
                Gb, _ = build_base_graph(maps[b])
                if nx.is_isomorphic(Ga, Gb):
                    print(f"  !! 同构: {a} ~ {b}")
                    any_isom = True
    if not any_isom:
        print("  无同构图（WL 指纹全部互异）")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""EasyWar 地图生成器 v5（产能约束版）
规范（docs/MAP_GUIDE.md）：
  尺寸 <= 15x13（5 人 60 格环等分特例 16x16）
  单据点地块 <= 8（产能 <= 4.1/s，驻军上限 <= 160）
"""
import json

def render(cells, w, h):
    rows = [['.'] * w for _ in range(h)]
    for (x, y), ch in cells.items():
        rows[y][x] = ch
    return [''.join(r) for r in rows]

def mirror180(cells, w, h, cmap):
    """180° 对称补全: (x,y)->(w-1-x, h-1-y)，字符按 cmap 映射。"""
    out = dict(cells)
    for (x, y), ch in list(cells.items()):
        p = (w - 1 - x, h - 1 - y)
        out.setdefault(p, cmap.get(ch, ch))
    return out

# ============================================================
# 1. peanut 花生 · 双环 v2 (1v1, 11x11) —— 小双环共享梁
# ============================================================
P = {}
# 上环带 行1-5 列1-9（9x5 环，内 void 行2-4 列3-7）
for x in range(1, 7):
    P[(x, 1)] = 'a'       # 上环顶左段（A 区）
for x in range(7, 10):
    P[(x, 1)] = 'c'       # 上环顶右段（C 区）
for x in range(1, 4):
    P[(x, 9)] = 'u'       # 下环底左段（U 区）
for x in range(4, 10):
    P[(x, 9)] = 'm'       # 下环底右段（M 区）
for y in range(1, 6):
    P[(1, y)] = 'a' if y <= 4 else 'p'
    P[(9, y)] = 'c' if y <= 4 else 'g'
for y in range(5, 10):
    P[(1, y)] = 'u' if y >= 6 else 'p'
    P[(9, y)] = 'm' if y >= 6 else 'g'
# 共享梁 行5 列2-8
for x in range(2, 9):
    P[(x, 5)] = 'p' if x <= 4 else ('g' if x >= 6 else 'p')
# 据点（180° 对: x'=10-x, y'=10-y）
P[(2, 1)] = 'A'
P[(8, 9)] = 'M'
P[(9, 3)] = 'C'
P[(1, 7)] = 'U'
P[(3, 5)] = 'P'
P[(7, 5)] = 'G'
peanut = mirror180(P, 11, 11, {'a': 'm', 'm': 'a', 'c': 'u', 'u': 'c',
                               'A': 'M', 'M': 'A', 'C': 'U', 'U': 'C',
                               'P': 'G', 'G': 'P', 'p': 'g', 'g': 'p'})
peanut_rows = render(peanut, 11, 11)

# ============================================================
# 2. isles 双岛 · 浅滩 v4 (1v1, 15x13) —— 双桥 + 岛内对位
# ============================================================
I = {}
# 左岛 列1-3 行3-9（3x7=21 格）：P(1,3)、A(1,9)、void (2,6)(3,6)(2,7)
for y in range(3, 10):
    I[(1, y)] = 'p' if y < 6 else 'a'
I[(2, 3)] = 'p'; I[(3, 3)] = 'p'
I[(2, 4)] = 'p'; I[(3, 4)] = 'p'
I[(2, 5)] = 'p'; I[(3, 5)] = 'p'
I[(2, 8)] = 'a'; I[(3, 8)] = 'a'
I[(2, 9)] = 'a'; I[(3, 9)] = 'a'
# void: (2,6)(3,6)(2,7) 留空；列1 行6-7 需连通 -> (1,6)(1,7) = 'p'/'a'
I[(1, 6)] = 'a'
I[(1, 7)] = 'a'
# 桥1 行4 列4-10（C 桥）
for x in range(4, 11):
    I[(x, 4)] = 'c'
# 桥2 行8 列4-10（U 桥）
for x in range(4, 11):
    I[(x, 8)] = 'u'
# 据点
I[(1, 3)] = 'P'
I[(1, 9)] = 'A'
I[(7, 4)] = 'C'
I[(7, 8)] = 'U'
isles = mirror180(I, 15, 13, {'p': 'g', 'g': 'p', 'a': 'm', 'm': 'a', 'c': 'u', 'u': 'c',
                              'P': 'G', 'G': 'P', 'A': 'M', 'M': 'A',
                              'C': 'U', 'U': 'C'})
isles_rows = render(isles, 15, 13)

# ============================================================
# 3. ring 环形 · 赛道 v2 (4人, 15x13) —— 1 宽环，8 据点交替
# ============================================================
R = {}
ring_cells = set()
for x in range(15):
    ring_cells.add((x, 0)); ring_cells.add((x, 12))
for y in range(1, 12):
    ring_cells.add((0, y)); ring_cells.add((14, y))
# 8 据点等距（环 52 格，8 段 6-7 格）：A P M G C X U G2
order = []
order += [(x, 0) for x in range(15)]
order += [(14, y) for y in range(1, 12)]
order += [(x, 12) for x in range(14, -1, -1)]
order += [(0, y) for y in range(11, 0, -1)]
assert len(order) == 52 and len(set(order)) == 52
# 8 段：6,7,6,7,6,7,6,7
cuts = [0, 6, 13, 19, 26, 32, 39, 45, 52]
segs = [order[cuts[i]:cuts[i+1]] for i in range(8)]
bases8 = ['A', 'P', 'M', 'G', 'C', 'X', 'U', 'G']
for i, seg in enumerate(segs):
    owner = bases8[i].lower()
    for cell in seg:
        R[cell] = owner
    R[seg[0]] = bases8[i]
ring_rows = render(R, 15, 13)

# ============================================================
# 4. sixline 六横 · 对垒 v2 (6人, 15x11) —— 压到 15 宽 + 每据点<=8
# ============================================================
S = {}
# 行1 列1-13：a a A a m m m M m c c C c c（A 3、M 4、C 4）
S[(1, 1)] = 'a'; S[(2, 1)] = 'a'; S[(3, 1)] = 'A'
S[(4, 1)] = 'a'; S[(5, 1)] = 'm'; S[(6, 1)] = 'm'; S[(7, 1)] = 'm'
S[(8, 1)] = 'M'; S[(9, 1)] = 'm'
S[(10, 1)] = 'c'; S[(11, 1)] = 'c'; S[(12, 1)] = 'c'; S[(13, 1)] = 'C'
S[(14, 1)] = 'c'
# 行9 镜像
S[(1, 9)] = 'u'; S[(2, 9)] = 'u'; S[(3, 9)] = 'U'
S[(4, 9)] = 'u'; S[(5, 9)] = 'g'; S[(6, 9)] = 'g'; S[(7, 9)] = 'g'
S[(8, 9)] = 'G'; S[(9, 9)] = 'g'
S[(10, 9)] = 'p'; S[(11, 9)] = 'p'; S[(12, 9)] = 'p'; S[(13, 9)] = 'P'
S[(14, 9)] = 'p'
# 行4 中央带上沿：玩家各 2 块 + 竖列 x（X 每座 <= 7 块）
S[(1, 4)] = 'a'; S[(2, 4)] = 'a'
S[(3, 4)] = 'x'
S[(4, 4)] = 'm'; S[(5, 4)] = 'm'
S[(8, 4)] = 'x'
S[(10, 4)] = 'c'; S[(11, 4)] = 'c'
S[(13, 4)] = 'x'
# 行6 镜像
S[(1, 6)] = 'u'; S[(2, 6)] = 'u'
S[(3, 6)] = 'x'
S[(4, 6)] = 'g'; S[(5, 6)] = 'g'
S[(8, 6)] = 'x'
S[(10, 6)] = 'p'; S[(11, 6)] = 'p'
S[(13, 6)] = 'x'
# 中央带 行5 全通
for x in range(1, 15):
    S[(x, 5)] = 'x'
S[(3, 5)] = 'X'; S[(8, 5)] = 'X'; S[(13, 5)] = 'X'
# 竖列
for x, ow in ((3, 'a'), (8, 'm'), (13, 'c')):
    S[(x, 2)] = ow; S[(x, 3)] = ow
    S[(x, 7)] = 'u' if ow == 'a' else ('g' if ow == 'm' else 'p')
    S[(x, 8)] = S[(x, 7)]
six_rows = render(S, 15, 11)

# ============================================================
# 5. pentaring 五星 · 环阵 v3 (5人, 16x16) —— 纯交替环 C10
#    环 60 格 10 段（6 格/段）：5 玩家 5 要塞交替，每据点 5 块
# ============================================================
P5 = {}
W5 = 16
order5 = []
order5 += [(x, 0) for x in range(W5)]
order5 += [(W5 - 1, y) for y in range(1, W5)]
order5 += [(x, W5 - 1) for x in range(W5 - 2, -1, -1)]
order5 += [(0, y) for y in range(W5 - 2, 0, -1)]
assert len(order5) == 60 and len(set(order5)) == 60
segs5 = [order5[i:i+6] for i in range(0, 60, 6)]
seg_owner = ['A', 'P', 'M', 'X', 'C', 'G', 'U', 'X', 'G', 'X']
for i, seg in enumerate(segs5):
    ow = seg_owner[i]
    for cell in seg:
        P5[cell] = ow.lower()
    P5[seg[0]] = ow
pent_rows = render(P5, 16, 16)

OUT = {
    'peanut': peanut_rows,
    'isles': isles_rows,
    'ring': ring_rows,
    'sixline': six_rows,
    'pentaring': pent_rows,
}
print(json.dumps(OUT))

#!/usr/bin/env python3
"""v3 地图生成器 v3：统一「地块循环先行，据点最后覆盖」。"""
import json

def render(cells, w, h):
    rows = [['.'] * w for _ in range(h)]
    for (x, y), ch in cells.items():
        rows[y][x] = ch
    return [''.join(r) for r in rows]

def mirror180(cells, w, h, cmap):
    out = dict(cells)
    for (x, y), ch in list(cells.items()):
        p = (w - 1 - x, h - 1 - y)
        out.setdefault(p, cmap.get(ch, ch))
    return out

# ============================================================
# 1. 花生 · 双环 (1v1, 13x13)
# ============================================================
P = {}
for x in range(1, 12):
    P[(x, 1)] = 'a'
    P[(x, 11)] = 'm'
for y in range(2, 6):
    P[(1, y)] = 'a'
    P[(2, y)] = 'a'
    P[(10, y)] = 'c'
    P[(11, y)] = 'c'
for y in range(7, 11):
    P[(1, y)] = 'u'
    P[(2, y)] = 'u'
    P[(10, y)] = 'm'
    P[(11, y)] = 'm'
for x in (1, 11):
    P[(x, 6)] = 'p'
for x in range(2, 8):
    P[(x, 6)] = 'p'
for x in range(8, 11):
    P[(x, 6)] = 'g'
# 据点最后覆盖
P[(2, 1)] = 'A'
P[(10, 11)] = 'M'
P[(11, 3)] = 'C'
P[(1, 9)] = 'U'
P[(3, 6)] = 'P'
P[(9, 6)] = 'G'
peanut = mirror180(P, 13, 13, {'a': 'm', 'm': 'a', 'c': 'u', 'u': 'c',
                               'A': 'M', 'M': 'A', 'C': 'U', 'U': 'C',
                               'P': 'G', 'G': 'P', 'p': 'g', 'g': 'p'})
peanut_rows = render(peanut, 13, 13)

# ============================================================
# 2. 三星 · 环 (3人, 17x13)：A 扇面双出口 + 双横梁
# ============================================================
S = {}
S[(8, 1)] = 'A'
for y, xs in ((2, range(7, 10)), (3, range(6, 11)), (4, range(5, 12)), (5, range(5, 12))):
    for x in xs:
        S[(x, y)] = 'a'
for x in range(5, 14):
    S[(x, 6)] = 'p'
for x in range(5, 14):
    S[(x, 7)] = 'p'
for x in range(6, 13):
    S[(x, 8)] = 'p'
S[(5, 8)] = 'm'
S[(13, 8)] = 'c'
for x in range(5, 13):
    S[(x, 9)] = 'p' if x <= 11 else 'c'
S[(4, 9)] = 'm'
S[(5, 9)] = 'm'
S[(12, 9)] = 'c'
S[(13, 9)] = 'c'
for x in range(4, 6):
    S[(x, 10)] = 'm'
S[(3, 10)] = 'm'
S[(4, 11)] = 'm'
S[(3, 11)] = 'm'
S[(11, 10)] = 'c'
S[(12, 10)] = 'c'
S[(13, 10)] = 'c'
S[(12, 11)] = 'c'
S[(13, 11)] = 'c'
S[(8, 6)] = 'P'
S[(2, 11)] = 'M'
S[(14, 11)] = 'C'
star_rows = render(S, 17, 13)

# ============================================================
# 3. 双岛 · 浅滩 v3 (1v1, 17x13)
# ============================================================
I = {}
for y in range(2, 11):
    for x in range(1, 5):
        I[(x, y)] = 'p'
for x in range(5, 12):
    I[(x, 4)] = 'c'
for x in range(5, 12):
    I[(x, 8)] = 'u'
for x in range(5, 9):
    I[(x, 6)] = 'a'
for x in range(9, 12):
    I[(x, 6)] = 'm'
I[(4, 4)] = 'p'
I[(4, 6)] = 'p'
I[(1, 2)] = 'P'
I[(1, 10)] = 'A'
I[(8, 4)] = 'C'
I[(8, 8)] = 'U'
isles = mirror180(I, 17, 13, {'p': 'm', 'm': 'p', 'a': 'm', 'c': 'u', 'u': 'c',
                              'P': 'G', 'G': 'P', 'A': 'M', 'M': 'A',
                              'C': 'U', 'U': 'C'})
isles_rows = render(isles, 17, 13)

# ============================================================
# 4. 三足 · 鼎立 v4 (3人, 15x13)：三房 + 双走廊（要塞卡口 + 直通）+ X
# ============================================================
T = {}
# 北房 行1-3 列4-8
for y in (1, 2, 3):
    for x in range(4, 9):
        T[(x, y)] = 'a'
T[(6, 4)] = 'a'
# 左房 行10-12 列1-5 / 右房 列9-13
for y in (10, 11, 12):
    for x in range(1, 6):
        T[(x, y)] = 'm'
    for x in range(9, 14):
        T[(x, y)] = 'c'
# 左走廊：卡口(5,9)U + 直通(6,8)(6,9)(6,10)
T[(6, 8)] = 'u'
T[(5, 8)] = 'u'
T[(6, 9)] = 'u'
T[(6, 10)] = 'u'
T[(5, 10)] = 'u'
# 右走廊：卡口(9,9)G + 直通(8,8)(8,9)(8,10)
T[(8, 8)] = 'g'
T[(9, 8)] = 'g'
T[(8, 9)] = 'g'
T[(8, 10)] = 'g'
T[(9, 10)] = 'g'
# 中央三角 行5-7 列6-8
for y in (5, 6, 7):
    for x in range(6, 9):
        T[(x, y)] = 'x'
# 据点最后覆盖
T[(6, 1)] = 'A'
T[(3, 11)] = 'M'
T[(11, 11)] = 'C'
T[(7, 4)] = 'P'
T[(5, 9)] = 'U'
T[(9, 9)] = 'G'
T[(7, 6)] = 'X'
tripod_rows = render(T, 15, 13)

# ============================================================
# 5. 五星 · 环阵 v3 (5人, 15x15)：环 56 格 5 段 + 中心 X（P 移除）
# ============================================================
R5 = {}
order = []
order += [(x, 0) for x in range(0, 15)]
order += [(14, y) for y in range(1, 15)]
order += [(x, 14) for x in range(13, -1, -1)]
order += [(0, y) for y in range(13, 0, -1)]
assert len(order) == 56 and len(set(order)) == 56
segs = [order[i:i+11] for i in range(0, 44, 11)]
segs.append(order[44:56])
players_ring = ['G', 'A', 'U', 'M', 'C']
for i, seg in enumerate(segs):
    owner = players_ring[i].lower()
    for cell in seg:
        R5[cell] = owner
    R5[seg[0]] = players_ring[i].upper()
R5[(7, 7)] = 'X'
for dx, dy in ((0, -1), (-1, 0), (1, 0), (0, 1)):
    R5[(7 + dx, 7 + dy)] = 'x'
# 五辐条等长（每辐 5 格玩家地 + 岛边 x）
spokes = {
    'G': [(7, 1), (7, 2), (7, 3), (7, 4), (7, 5)],          # N -> 段1 G
    'A': [(9, 7), (10, 7), (11, 7), (12, 7), (13, 7)],      # E -> 段2 A
    'U': [(1, 7), (2, 7), (3, 7), (4, 7), (5, 7)],          # W -> 段5 C 侧
    'M': [(7, 9), (7, 10), (7, 11), (7, 12), (7, 13)],      # S -> 段4 M
    'C': [(6, 6), (5, 5), (4, 4), (3, 3), (2, 2), (1, 1)],  # NW -> 段1 G 侧
}
for owner, cells in spokes.items():
    for cell in cells:
        R5.setdefault(cell, owner.lower())
pent_rows = render(R5, 15, 15)

# ============================================================
# 6. 六横 · 对垒 v3 (6人, 17x11)
# ============================================================
SX = {}
for x in range(1, 16):
    SX[(x, 1)] = 'a' if x <= 4 else ('m' if x <= 9 else 'c')
    SX[(x, 9)] = 'u' if x <= 4 else ('g' if x <= 9 else 'p')
    SX[(x, 4)] = 'a' if x <= 4 else ('m' if x <= 9 else 'c')
    SX[(x, 6)] = 'u' if x <= 4 else ('g' if x <= 9 else 'p')
    SX[(x, 5)] = 'x'
for x in (3, 8, 13):
    SX[(x, 2)] = 'a' if x == 3 else ('m' if x == 8 else 'c')
    SX[(x, 3)] = SX[(x, 2)]
    SX[(x, 7)] = 'u' if x == 3 else ('g' if x == 8 else 'p')
    SX[(x, 8)] = SX[(x, 7)]
SX[(3, 1)] = 'A'; SX[(8, 1)] = 'M'; SX[(13, 1)] = 'C'
SX[(3, 9)] = 'U'; SX[(8, 9)] = 'G'; SX[(13, 9)] = 'P'
SX[(3, 5)] = 'X'; SX[(8, 5)] = 'X'; SX[(13, 5)] = 'X'
six_rows = render(SX, 17, 11)

OUT = {
    'peanut': peanut_rows,
    'starring': star_rows,
    'isles': isles_rows,
    'tripod': tripod_rows,
    'pentaring': pent_rows,
    'sixline': six_rows,
}
print(json.dumps(OUT))

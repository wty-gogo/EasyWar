#!/usr/bin/env python3
"""v3 生成器 v4：tripod=六边形环(C6)，pentaring=60格环完美5等分。"""
import json

def render(cells, w, h):
    rows = [['.'] * w for _ in range(h)]
    for (x, y), ch in cells.items():
        rows[y][x] = ch
    return [''.join(r) for r in rows]

def line(p1, p2):
    """画曼哈顿直线（先竖后横），返回格列表（含两端）。"""
    (x1, y1), (x2, y2) = p1, p2
    pts = []
    if y1 < y2:
        for y in range(y1, y2 + 1):
            pts.append((x1, y))
    else:
        for y in range(y1, y2 - 1, -1):
            pts.append((x1, y))
    if x1 < x2:
        for x in range(x1 + 1, x2 + 1):
            pts.append((x, y2))
    else:
        for x in range(x1 - 1, x2 - 1, -1):
            pts.append((x, y2))
    return pts

# ============================================================
# 7. 三足 · 鼎立 v5 (3人, 15x13)：六边形环 C6
#    顶点: A(7,1) P(2,4) M(2,10) U(7,12) C(12,10) G(12,4)
#    玩家 A/M/C 与要塞 P/U/G 交替，环上等距
# ============================================================
verts = {'A': (7, 1), 'P': (2, 4), 'M': (2, 10), 'U': (7, 12), 'C': (12, 10), 'G': (12, 4)}
order6 = ['A', 'P', 'M', 'U', 'C', 'G']
T = {}
ring_cells = set()
for i in range(6):
    a = verts[order6[i]]
    b = verts[order6[(i + 1) % 6]]
    for c in line(a, b):
        ring_cells.add(c)
# 环格归属：最近顶点据点
def nearest_owner(pos):
    best, bd = None, 1e9
    for name, v in verts.items():
        d = abs(pos[0] - v[0]) + abs(pos[1] - v[1])
        if d < bd:
            best, bd = name, d
    return best
for c in ring_cells:
    own = nearest_owner(c)
    T[c] = own.lower() if own in 'AMC' else own.lower()
# 据点最后覆盖
for name, v in verts.items():
    T[v] = name
# 环内 void 的 4 个角补地块（增加战略性纵深 + 洞数）
# 中心偏下：X? 不加要塞，保持 C6 纯粹
tripod_rows = render(T, 15, 13)

# ============================================================
# 8. 五星 · 环阵 v4 (5人, 16x16)：环 60 格 = 5×12 完美等分
# ============================================================
W = 16
order = []
order += [(x, 0) for x in range(W)]
order += [(W - 1, y) for y in range(1, W)]
order += [(x, W - 1) for x in range(W - 2, -1, -1)]
order += [(0, y) for y in range(W - 2, 0, -1)]
assert len(order) == 60 and len(set(order)) == 60
segs = [order[i:i+12] for i in range(0, 60, 12)]
players_ring = ['G', 'A', 'U', 'M', 'C']
R5 = {}
for i, seg in enumerate(segs):
    owner = players_ring[i].lower()
    for cell in seg:
        R5[cell] = owner
    R5[seg[0]] = players_ring[i].upper()
# 中心岛：X(8,8) + 4 x
R5[(8, 8)] = 'X'
for dx, dy in ((0, -1), (-1, 0), (1, 0), (0, 1)):
    R5[(8 + dx, 8 + dy)] = 'x'
# 五辐条（直线，等长 7 格）到中心岛边
spokes = {
    'G': [(8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7)],       # N
    'M': [(8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (8, 15)], # S
    'A': [(9, 8), (10, 8), (11, 8), (12, 8), (13, 8), (14, 8), (15, 8)], # E
    'C': [(1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (7, 8)],       # W
}
# 注意：NW 辐条会穿过 G/C 段制造跨玩家捷径，故删除；U 玩家借 A 段 E 辐条直连中心
for owner, cells in spokes.items():
    for cell in cells:
        R5.setdefault(cell, owner.lower())
pent_rows = render(R5, 16, 16)

OUT = {'tripod': tripod_rows, 'pentaring': pent_rows}
print(json.dumps(OUT))

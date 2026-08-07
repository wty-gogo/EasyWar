use bevy_app::App;
use easywar_logic::*;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

const CANDIDATES: [&str; 3] = [
    "dual_ladder_1v1.toml",
    "braided_rings_1v1.toml",
    "ring_chord_1v1.toml",
];

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets")
}

fn load_map(path: &Path) -> App {
    let mut app = App::new();
    app.add_plugins(GamePlugin);
    spawn_map_seeded(app.world_mut(), path, &assets_dir().join("subjects"), 42)
        .unwrap_or_else(|error| panic!("加载 {} 失败: {error}", path.display()));
    app
}

fn reachable_without(
    adjacency: &[HashSet<usize>],
    start: usize,
    goal: usize,
    removed: Option<usize>,
) -> bool {
    let mut seen = vec![false; adjacency.len()];
    let mut pending = vec![start];
    seen[start] = true;
    while let Some(current) = pending.pop() {
        if current == goal {
            return true;
        }
        for &next in &adjacency[current] {
            if Some(next) != removed && !seen[next] {
                seen[next] = true;
                pending.push(next);
            }
        }
    }
    false
}

fn assert_candidate_invariants(name: &str) {
    let app = load_map(&assets_dir().join("maps").join(name));
    let lookup = app.world().resource::<GridLookup>().clone();
    let base_list = app.world().resource::<BaseList>().clone();
    assert!(
        lookup.width <= 17 && lookup.height <= 13,
        "{name} 必须能在当前单屏显示"
    );

    let base_cells: Vec<CellIdx> = base_list
        .0
        .iter()
        .map(|entity| {
            lookup
                .cells
                .iter()
                .position(|candidate| candidate == entity)
                .expect("据点必须属于 GridLookup")
        })
        .collect();
    let linked: Vec<Vec<CellIdx>> = base_list
        .0
        .iter()
        .map(|&entity| {
            app.world()
                .get::<Base>(entity)
                .expect("据点缺少 Base")
                .linked
                .clone()
        })
        .collect();
    assert!(
        linked.iter().all(|tiles| (1..=10).contains(&tiles.len())),
        "{name} 每个据点必须拥有 1～10 块关联地块"
    );

    let mut region_of = vec![None; lookup.cells.len()];
    for (region, (&base, tiles)) in base_cells.iter().zip(&linked).enumerate() {
        for cell in std::iter::once(base).chain(tiles.iter().copied()) {
            assert!(
                region_of[cell].replace(region).is_none(),
                "{name} 的格子被多个据点关联"
            );
        }
    }
    for (cell, &entity) in lookup.cells.iter().enumerate() {
        let kind = *app
            .world()
            .get::<CellKind>(entity)
            .expect("格子缺少 CellKind");
        assert_ne!(kind, CellKind::Plain, "{name} 不允许无关联普通地块");
        assert_eq!(
            kind.enterable(),
            region_of[cell].is_some(),
            "{name} 每个可进入格必须且只能属于一个据点"
        );
    }

    let mut adjacency = vec![HashSet::new(); base_cells.len()];
    for (cell, region) in region_of.iter().enumerate() {
        let Some(region) = *region else { continue };
        let (x, y) = lookup.xy(cell);
        let neighbors = [
            x.checked_sub(1).map(|nx| lookup.idx(nx, y)),
            (x + 1 < lookup.width).then_some(lookup.idx(x + 1, y)),
            y.checked_sub(1).map(|ny| lookup.idx(x, ny)),
            (y + 1 < lookup.height).then_some(lookup.idx(x, y + 1)),
        ];
        for next in neighbors.into_iter().flatten() {
            if let Some(other) = region_of[next] {
                if other != region {
                    adjacency[region].insert(other);
                    adjacency[other].insert(region);
                }
            }
        }
    }

    let base_owners: Vec<FactionId> = base_cells
        .iter()
        .map(|&cell| {
            app.world()
                .get::<Owner>(lookup.entity(cell))
                .expect("据点缺少 Owner")
                .0
        })
        .collect();
    let start = base_owners
        .iter()
        .position(|&owner| owner == 1)
        .expect("缺少玩家出生据点");
    let goal = base_owners
        .iter()
        .position(|&owner| owner == 2)
        .expect("缺少 AI 出生据点");
    assert!(
        reachable_without(&adjacency, start, goal, None),
        "{name} 双方据点不可达"
    );
    for neutral in (0..base_cells.len()).filter(|&region| region != start && region != goal) {
        assert!(
            reachable_without(&adjacency, start, goal, Some(neutral)),
            "{name} 的中立据点 {neutral} 是双方绝对必经点"
        );
    }

    let mirror_cell = |cell: CellIdx| {
        let (x, y) = lookup.xy(cell);
        lookup.idx(lookup.width - 1 - x, y)
    };
    let mirror_region: Vec<usize> = base_cells
        .iter()
        .map(|&cell| region_of[mirror_cell(cell)].expect("镜像据点没有对应区域"))
        .collect();
    for (cell, region) in region_of.iter().enumerate() {
        let Some(region) = *region else { continue };
        assert_eq!(
            region_of[mirror_cell(cell)],
            Some(mirror_region[region]),
            "{name} 的关联区域不满足左右镜像"
        );
        let left = app
            .world()
            .get::<Garrison>(lookup.entity(cell))
            .expect("格子缺少 Garrison")
            .cur;
        let right = app
            .world()
            .get::<Garrison>(lookup.entity(mirror_cell(cell)))
            .expect("镜像格缺少 Garrison")
            .cur;
        assert_eq!(left, right, "{name} 的镜像格防御值不一致");
    }
    for (region, &mirror) in mirror_region.iter().enumerate() {
        let expected_owner = match base_owners[region] {
            1 => 2,
            2 => 1,
            owner => owner,
        };
        assert_eq!(
            base_owners[mirror], expected_owner,
            "{name} 的出生位交换不对称"
        );
        assert_eq!(
            linked[region].len(),
            linked[mirror].len(),
            "{name} 的镜像据点地块数不一致"
        );
    }
}

#[test]
fn candidate_maps_satisfy_static_standard() {
    for name in CANDIDATES {
        assert_candidate_invariants(name);
    }
}

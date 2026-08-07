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

fn path_distance(app: &mut App, from: CellIdx, to: CellIdx, faction: FactionId) -> usize {
    find_path(app.world_mut(), from, to, faction)
        .unwrap_or_else(|| panic!("{from} 到 {to} 不可达"))
        .len()
        - 1
}

fn three_faction_candidate(
    name: &str,
    expected_linked: &[usize; 9],
    expected_edges: &[(usize, usize)],
) -> (App, Vec<CellIdx>) {
    let app = load_map(&assets_dir().join("maps").join(name));
    let lookup = app.world().resource::<GridLookup>().clone();
    let bases = app.world().resource::<BaseList>().clone();
    assert_eq!(app.world().resource::<Factions>().0.len(), 3);
    assert!(lookup.width <= 17 && lookup.height <= 13);
    assert_eq!(bases.0.len(), 9);

    let base_cells = bases
        .0
        .iter()
        .map(|entity| {
            lookup
                .cells
                .iter()
                .position(|candidate| candidate == entity)
                .expect("据点必须属于 GridLookup")
        })
        .collect::<Vec<_>>();
    let linked = bases
        .0
        .iter()
        .map(|entity| {
            app.world()
                .get::<Base>(*entity)
                .expect("据点缺少 Base")
                .linked
                .clone()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        linked.iter().map(Vec::len).collect::<Vec<_>>(),
        expected_linked
    );

    let mut region_of = vec![None; lookup.cells.len()];
    for (region, (&base, tiles)) in base_cells.iter().zip(&linked).enumerate() {
        for cell in std::iter::once(base).chain(tiles.iter().copied()) {
            assert!(region_of[cell].replace(region).is_none());
        }
    }
    let region_of_ref = &region_of;
    let actual_edges = region_of
        .iter()
        .enumerate()
        .filter_map(|(cell, region)| region.map(|region| (cell, region)))
        .flat_map(|(cell, region)| {
            let (x, y) = lookup.xy(cell);
            [
                x.checked_sub(1).map(|nx| lookup.idx(nx, y)),
                (x + 1 < lookup.width).then_some(lookup.idx(x + 1, y)),
                y.checked_sub(1).map(|ny| lookup.idx(x, ny)),
                (y + 1 < lookup.height).then_some(lookup.idx(x, y + 1)),
            ]
            .into_iter()
            .flatten()
            .filter_map(move |next| region_of_ref[next].map(|other| (region, other)))
        })
        .filter(|(left, right)| left != right)
        .map(|(left, right)| (left.min(right), left.max(right)))
        .collect::<HashSet<_>>();
    let expected_edges = expected_edges
        .iter()
        .map(|&(left, right)| (left.min(right), left.max(right)))
        .collect::<HashSet<_>>();
    assert_eq!(actual_edges, expected_edges, "{name} 的真实据点图发生变化");
    (app, base_cells)
}

fn distance_signature(
    app: &mut App,
    start: CellIdx,
    targets: impl IntoIterator<Item = CellIdx>,
) -> Vec<usize> {
    let owner = app
        .world()
        .get::<Owner>(app.world().resource::<GridLookup>().entity(start))
        .expect("出生据点缺少 Owner")
        .0;
    let mut distances = targets
        .into_iter()
        .map(|target| path_distance(app, start, target, owner))
        .collect::<Vec<_>>();
    distances.sort_unstable();
    distances
}

#[test]
fn layered_triangle_has_equal_three_faction_opportunity_sets() {
    let edges = [
        (0, 3),
        (1, 3),
        (1, 4),
        (2, 4),
        (2, 5),
        (0, 5),
        (0, 6),
        (1, 7),
        (2, 8),
        (6, 7),
        (7, 8),
        (8, 6),
    ];
    let (mut app, cells) = three_faction_candidate(
        "layered_triangle_3ffa.toml",
        &[7, 7, 7, 6, 6, 6, 7, 7, 7],
        &edges,
    );
    let players = cells[0..3].to_vec();
    let fast = cells[3..6].to_vec();
    let economy = cells[6..9].to_vec();

    for (index, &start) in players.iter().enumerate() {
        assert_eq!(
            distance_signature(
                &mut app,
                start,
                players.iter().copied().filter(|target| *target != start),
            ),
            vec![8, 8],
            "第 {index} 个出生位到对手的格距集合不等价"
        );
        assert_eq!(
            distance_signature(&mut app, start, fast.iter().copied()),
            vec![6, 6, 14],
            "第 {index} 个出生位到快点的机会集合不等价"
        );
        assert_eq!(
            distance_signature(&mut app, start, economy.iter().copied()),
            vec![2, 6, 6],
            "第 {index} 个出生位到经济点的机会集合不等价"
        );
    }
    for index in 0..3 {
        assert_eq!(
            path_distance(&mut app, economy[index], economy[(index + 1) % 3], NEUTRAL),
            4,
            "内圈相邻经济点的换线格距必须相同"
        );
    }
}

#[test]
fn three_leaf_windmill_has_equal_opening_opportunities() {
    let edges = [
        (0, 3),
        (1, 3),
        (1, 4),
        (2, 4),
        (2, 5),
        (0, 5),
        (3, 4),
        (4, 5),
        (5, 3),
        (0, 6),
        (1, 7),
        (2, 8),
    ];
    let (mut app, cells) = three_faction_candidate(
        "three_leaf_windmill_3ffa.toml",
        &[6, 6, 6, 6, 6, 6, 7, 7, 7],
        &edges,
    );
    let players = cells[0..3].to_vec();
    let hubs = cells[3..6].to_vec();
    let leaves = cells[6..9].to_vec();

    for (index, &start) in players.iter().enumerate() {
        assert_eq!(
            distance_signature(
                &mut app,
                start,
                players.iter().copied().filter(|target| *target != start),
            ),
            vec![8, 8]
        );
        assert_eq!(
            distance_signature(&mut app, start, hubs.iter().copied()),
            vec![4, 4, 8],
            "第 {index} 个出生位到共享枢纽的机会集合不等价"
        );
        assert_eq!(
            path_distance(&mut app, start, leaves[index], index as FactionId + 1),
            3,
            "第 {index} 个出生位到本地经济叶子的开局格距不等价"
        );
    }
    for index in 0..3 {
        assert_eq!(
            path_distance(&mut app, hubs[index], hubs[(index + 1) % 3], NEUTRAL),
            6,
            "枢纽三角的换线格距必须相同"
        );
    }
}

#[test]
fn tripod_ring_is_a_symmetric_three_faction_map() {
    let path = assets_dir().join("maps/tripod_ring_3ffa.toml");
    let mut app = load_map(&path);
    let lookup = app.world().resource::<GridLookup>().clone();
    let bases = app.world().resource::<BaseList>().clone();
    let factions = app.world().resource::<Factions>();

    assert_eq!(factions.0.len(), 3, "地图应生成 1 名玩家和 2 名 AI");
    assert_eq!(lookup.width, 17);
    assert_eq!(lookup.height, 9);
    assert!(bases.0.iter().all(|entity| {
        app.world()
            .get::<Base>(*entity)
            .is_some_and(|base| base.linked.len() == 5)
    }));

    let base_cells = bases
        .0
        .iter()
        .map(|entity| {
            lookup
                .cells
                .iter()
                .position(|candidate| candidate == entity)
                .expect("据点必须属于 GridLookup")
        })
        .collect::<Vec<_>>();
    let (starts, neutrals): (Vec<_>, Vec<_>) = base_cells.into_iter().partition(|cell| {
        app.world()
            .get::<Owner>(lookup.entity(*cell))
            .is_some_and(|owner| owner.0 != NEUTRAL)
    });
    assert_eq!(starts.len(), 3);
    assert_eq!(neutrals.len(), 3);

    for (index, &start) in starts.iter().enumerate() {
        let owner = app.world().get::<Owner>(lookup.entity(start)).unwrap().0;
        let mut rival_distances = starts
            .iter()
            .copied()
            .filter(|target| *target != start)
            .map(|target| path_distance(&mut app, start, target, owner))
            .collect::<Vec<_>>();
        rival_distances.sort_unstable();
        assert_eq!(
            rival_distances,
            vec![12, 12],
            "第 {index} 个出生位到两名对手的短路应等长"
        );

        let mut fortress_distances = neutrals
            .iter()
            .copied()
            .map(|target| path_distance(&mut app, start, target, owner))
            .collect::<Vec<_>>();
        fortress_distances.sort_unstable();
        assert_eq!(
            fortress_distances,
            vec![6, 6, 18],
            "第 {index} 个出生位的中立要塞机会集合应相同"
        );
    }
}

#[test]
fn custom_subject_selection_keeps_three_factions_distinct() {
    for name in [
        "tripod_ring_3ffa.toml",
        "layered_triangle_3ffa.toml",
        "three_leaf_windmill_3ffa.toml",
    ] {
        let mut app = App::new();
        app.add_plugins(GamePlugin);
        spawn_map_custom(
            app.world_mut(),
            &assets_dir().join("maps").join(name),
            &assets_dir().join("subjects"),
            Some("physics"),
            Some("math"),
        )
        .unwrap_or_else(|error| panic!("{name} 自选学科后加载失败: {error}"));

        let names = app
            .world()
            .resource::<Factions>()
            .0
            .iter()
            .map(|faction| faction.name.as_str())
            .collect::<HashSet<_>>();
        assert_eq!(names.len(), 3, "{name} 的三个初始阵营必须可区分");
    }
}

#[test]
fn three_faction_ai_controllers_can_expand_concurrently() {
    for name in [
        "tripod_ring_3ffa.toml",
        "layered_triangle_3ffa.toml",
        "three_leaf_windmill_3ffa.toml",
    ] {
        let mut app = load_map(&assets_dir().join("maps").join(name));
        let faction_ids = app
            .world()
            .resource::<Factions>()
            .0
            .iter()
            .map(|faction| faction.id)
            .collect::<Vec<_>>();
        app.world_mut().insert_resource(AiControllers(
            faction_ids
                .into_iter()
                .map(|faction| AiController::seeded(faction, AiParams::normal(), 42))
                .collect(),
        ));
        for _ in 0..(30.0 / SIM_DT) as usize {
            app.world_mut().try_run_schedule(SimTick).unwrap();
        }

        let lookup = app.world().resource::<GridLookup>();
        let occupied_linked_tiles = lookup
            .cells
            .iter()
            .filter(|entity| {
                app.world().get::<CellKind>(**entity) == Some(&CellKind::LinkedTile)
                    && app
                        .world()
                        .get::<Owner>(**entity)
                        .is_some_and(|owner| owner.0 != NEUTRAL)
            })
            .count();
        assert!(
            occupied_linked_tiles >= 3,
            "{name} 的三个 AI 并发决策后应已开始扩张"
        );
    }
}

#[test]
fn three_faction_match_ends_only_after_two_factions_are_eliminated() {
    let mut app = load_map(&assets_dir().join("maps/tripod_ring_3ffa.toml"));
    let bases = app.world().resource::<BaseList>().clone();
    let faction_base = |app: &App, faction| {
        bases
            .0
            .iter()
            .copied()
            .find(|entity| {
                app.world()
                    .get::<Owner>(*entity)
                    .is_some_and(|owner| owner.0 == faction)
            })
            .expect("阵营应有出生据点")
    };

    let third = faction_base(&app, 3);
    app.world_mut().get_mut::<Owner>(third).unwrap().0 = 1;
    app.world_mut().try_run_schedule(SimTick).unwrap();
    assert_eq!(
        app.world().resource::<Winner>().0,
        None,
        "淘汰一方后仍有两个阵营，不应提前结束"
    );

    let second = faction_base(&app, 2);
    app.world_mut().get_mut::<Owner>(second).unwrap().0 = 1;
    app.world_mut().try_run_schedule(SimTick).unwrap();
    assert_eq!(app.world().resource::<Winner>().0, Some(1));
}

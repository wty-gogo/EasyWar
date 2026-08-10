//! 地图与词库加载：TOML 解析后向 World 生成格子/据点实体与基础资源。
//! 生成算法逐行移植自旧 load.rs（固定种子 + 180° 旋转对称成对赋值）。

use crate::components::*;
use bevy_ecs::prelude::*;
use serde::Deserialize;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;

// ---------- TOML 文件结构（与 assets/maps/*.toml 对应） ----------

#[derive(Deserialize)]
struct MapFile {
    map: MapMeta,
    base: Vec<BaseDef>,
    neutral: NeutralDef,
    linked_tile: LinkedTileDef,
    rules: RulesDef,
}

#[derive(Deserialize)]
struct MapMeta {
    #[allow(dead_code)]
    name: String,
    width: usize,
    height: usize,
    layout: Vec<String>,
}

#[derive(Deserialize)]
struct BaseDef {
    subject: String,
    owner: String, // "player" | "ai" | "neutral"（未来可扩展 "ai2"…）
    pos: (usize, usize),
    garrison: f32,
    production_base: f32,
    linked_tiles: Vec<LinkedRef>,
}

#[derive(Deserialize)]
struct LinkedRef {
    pos: (usize, usize),
    // 知识点不再按索引固定：每局从学科词库随机抽取
}

#[derive(Deserialize)]
struct NeutralDef {
    defense_min: f32,
    defense_max: f32,
    beam_defense_min: f32,
    beam_defense_max: f32,
    seed: u64,
    #[serde(default)]
    rotational_symmetry: bool,
    #[serde(default)]
    symmetry: Option<String>,
}

#[derive(Deserialize)]
struct LinkedTileDef {
    defense_min: f32,
    defense_max: f32,
    production_bonus: f32,
}

#[derive(Deserialize)]
struct RulesDef {
    garrison_cap_base: f32,
    garrison_cap_per_tile: f32,
    regen_per_sec: f32,
    squad_interval_sec: f32,
    squad_max_size: f32,
    #[serde(default = "default_squad_growth_garrison_step")]
    squad_growth_garrison_step: f32,
    #[serde(default = "default_squad_soft_cap_garrison")]
    squad_soft_cap_garrison: f32,
    squad_move_sec_per_cell: f32,
}

fn default_squad_growth_garrison_step() -> f32 {
    40.0
}

fn default_squad_soft_cap_garrison() -> f32 {
    120.0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MapSymmetry {
    None,
    Rotational,
    Vertical,
    Horizontal,
}

fn map_symmetry(neutral: &NeutralDef) -> Result<MapSymmetry, String> {
    match neutral.symmetry.as_deref() {
        Some("none") => Ok(MapSymmetry::None),
        Some("rotational") => Ok(MapSymmetry::Rotational),
        Some("vertical") => Ok(MapSymmetry::Vertical),
        Some("horizontal") => Ok(MapSymmetry::Horizontal),
        Some(other) => Err(format!("未知地图对称方式: {other}")),
        None if neutral.rotational_symmetry => Ok(MapSymmetry::Rotational),
        None => Ok(MapSymmetry::None),
    }
}

fn symmetric_index(i: usize, width: usize, height: usize, symmetry: MapSymmetry) -> usize {
    let (x, y) = (i % width, i / width);
    let (sx, sy) = match symmetry {
        MapSymmetry::None => (x, y),
        MapSymmetry::Rotational => (width - 1 - x, height - 1 - y),
        MapSymmetry::Vertical => (width - 1 - x, y),
        MapSymmetry::Horizontal => (x, height - 1 - y),
    };
    sy * width + sx
}

/// 只校验地图数据本身，不访问 ECS；加载和测试共用同一组不变量。
fn validate_map_file(mf: &MapFile) -> Result<MapSymmetry, String> {
    let (width, height) = (mf.map.width, mf.map.height);
    if width == 0 || height == 0 {
        return Err("地图尺寸不能为 0".into());
    }
    if mf.map.layout.len() != height || mf.map.layout.iter().any(|row| row.chars().count() != width)
    {
        return Err("地图 layout 尺寸与 width/height 不符".into());
    }
    if mf
        .map
        .layout
        .iter()
        .flat_map(|row| row.chars())
        .any(|cell| cell != '.' && cell != '#')
    {
        return Err("地图 layout 只允许使用 '.' 与 '#'".into());
    }
    if mf.base.len() < 2 {
        return Err("地图至少需要 2 个据点".into());
    }
    if !mf.rules.squad_growth_garrison_step.is_finite()
        || mf.rules.squad_growth_garrison_step <= 0.0
    {
        return Err("动态波次的驻军增长步长必须大于 0".into());
    }
    if !mf.rules.squad_soft_cap_garrison.is_finite()
        || mf.rules.squad_soft_cap_garrison < mf.rules.squad_growth_garrison_step
    {
        return Err("动态波次的缓增长起点不得小于驻军增长步长".into());
    }

    let enterable: Vec<bool> = mf
        .map
        .layout
        .iter()
        .flat_map(|row| row.chars().map(|cell| cell == '#'))
        .collect();
    let mut claimed_by: Vec<Option<&str>> = vec![None; width * height];
    let index_of = |pos: (usize, usize)| -> Result<usize, String> {
        let (x, y) = pos;
        if x >= width || y >= height {
            Err(format!("坐标越界: [{x}, {y}]"))
        } else {
            Ok(y * width + x)
        }
    };

    for base in &mf.base {
        if !(1..=10).contains(&base.linked_tiles.len()) {
            return Err(format!(
                "据点 {} 的关联地块数必须在 1～10，实际为 {}",
                base.subject,
                base.linked_tiles.len()
            ));
        }
        let positions =
            std::iter::once(base.pos).chain(base.linked_tiles.iter().map(|tile| tile.pos));
        for pos in positions {
            let index = index_of(pos)?;
            if !enterable[index] {
                return Err(format!(
                    "据点 {} 声明了 layout 外的格子: [{}, {}]",
                    base.subject, pos.0, pos.1
                ));
            }
            if let Some(other) = claimed_by[index] {
                return Err(format!(
                    "格子 [{}, {}] 同时关联 {} 与 {}",
                    pos.0, pos.1, other, base.subject
                ));
            }
            claimed_by[index] = Some(&base.subject);
        }
    }

    if let Some(index) = enterable
        .iter()
        .zip(&claimed_by)
        .position(|(&is_enterable, owner)| is_enterable != owner.is_some())
    {
        return Err(format!(
            "格子 [{}, {}] 未且仅未关联一个据点",
            index % width,
            index / width
        ));
    }

    let start = enterable
        .iter()
        .position(|&cell| cell)
        .ok_or_else(|| "地图没有可进入格子".to_string())?;
    let mut seen = vec![false; width * height];
    let mut queue = VecDeque::from([start]);
    seen[start] = true;
    while let Some(index) = queue.pop_front() {
        let (x, y) = (index % width, index / width);
        let adjacent = [
            x.checked_sub(1).map(|nx| y * width + nx),
            (x + 1 < width).then_some(y * width + x + 1),
            y.checked_sub(1).map(|ny| ny * width + x),
            (y + 1 < height).then_some((y + 1) * width + x),
        ];
        for next in adjacent.into_iter().flatten() {
            if enterable[next] && !seen[next] {
                seen[next] = true;
                queue.push_back(next);
            }
        }
    }
    if enterable
        .iter()
        .enumerate()
        .any(|(i, &cell)| cell && !seen[i])
    {
        return Err("地图存在不连通的可进入区域".into());
    }

    let symmetry = map_symmetry(&mf.neutral)?;
    if symmetry != MapSymmetry::None
        && enterable
            .iter()
            .enumerate()
            .any(|(i, &cell)| cell != enterable[symmetric_index(i, width, height, symmetry)])
    {
        return Err(format!("地图占用格不满足声明的 {symmetry:?} 对称"));
    }
    Ok(symmetry)
}

// ---------- 学科词库 ----------

#[derive(Clone, Debug, Deserialize)]
pub struct SubjectDef {
    pub id: String,
    pub name: String,
    pub color: String, // "#RRGGBB"
    pub knowledge_points: Vec<String>,
}

pub fn load_subjects(dir: &Path) -> Result<HashMap<String, SubjectDef>, String> {
    let mut out = HashMap::new();
    let rd = std::fs::read_dir(dir).map_err(|e| format!("读词库目录失败: {e}"))?;
    for entry in rd {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("toml") {
            continue;
        }
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("读 {:?} 失败: {e}", path))?;
        let def: SubjectDef =
            toml::from_str(&text).map_err(|e| format!("解析 {:?} 失败: {e}", path))?;
        out.insert(def.id.clone(), def);
    }
    Ok(out)
}

pub fn parse_hex_color(hex: &str) -> [f32; 4] {
    let h = hex.trim_start_matches('#');
    let r = u8::from_str_radix(&h[0..2], 16).unwrap_or(128) as f32 / 255.0;
    let g = u8::from_str_radix(&h[2..4], 16).unwrap_or(128) as f32 / 255.0;
    let b = u8::from_str_radix(&h[4..6], 16).unwrap_or(128) as f32 / 255.0;
    [r, g, b, 1.0]
}

// ---------- 确定性随机（xorshift64*，避免外部依赖） ----------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    /// [min, max] 内的浮点
    fn range(&mut self, min: f32, max: f32) -> f32 {
        let t = (self.next() % 10000) as f32 / 9999.0;
        min + t * (max - min)
    }
}

// ---------- 生成地图 ----------

/// 返回稳定的阵营顺序：玩家固定为 1，其余 owner 按名称排序。
/// 地图文件中的据点声明顺序不应改变阵营 id 或 AI 提交顺序。
fn faction_owner_names(bases: &[BaseDef]) -> Vec<String> {
    let mut owners = bases
        .iter()
        .filter(|base| base.owner != "neutral")
        .map(|base| base.owner.clone())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    owners.sort_by(|left, right| match (left.as_str(), right.as_str()) {
        ("player", "player") => std::cmp::Ordering::Equal,
        ("player", _) => std::cmp::Ordering::Less,
        (_, "player") => std::cmp::Ordering::Greater,
        _ => left.cmp(right),
    });
    owners
}

/// 开局自选学科后，为所有初始阵营分配互不重复的学科；中立据点再使用剩余学科。
fn assign_custom_subjects(
    map: &mut MapFile,
    subjects: &HashMap<String, SubjectDef>,
    player_subject: Option<&str>,
    ai_subject: Option<&str>,
) -> Result<(), String> {
    if player_subject.is_none() && ai_subject.is_none() {
        return Ok(());
    }

    let mut subject_ids = subjects.keys().cloned().collect::<Vec<_>>();
    subject_ids.sort();
    let mut used = HashSet::new();
    let owner_subjects = faction_owner_names(&map.base)
        .into_iter()
        .map(|owner| {
            let configured = map
                .base
                .iter()
                .find(|base| base.owner == owner)
                .map(|base| base.subject.as_str())
                .ok_or_else(|| format!("阵营 {owner} 没有出生据点"))?;
            let preferred = match owner.as_str() {
                "player" => player_subject.unwrap_or(configured),
                "ai" => ai_subject.unwrap_or(configured),
                _ => configured,
            };
            let chosen = (!used.contains(preferred) && subjects.contains_key(preferred))
                .then(|| preferred.to_string())
                .or_else(|| subject_ids.iter().find(|id| !used.contains(*id)).cloned())
                .ok_or_else(|| "可用学科数量少于参战阵营数量".to_string())?;
            used.insert(chosen.clone());
            Ok((owner, chosen))
        })
        .collect::<Result<HashMap<_, _>, String>>()?;

    map.base
        .iter_mut()
        .filter(|base| base.owner != "neutral")
        .for_each(|base| base.subject = owner_subjects[&base.owner].clone());

    let remaining = subject_ids
        .into_iter()
        .filter(|id| !used.contains(id))
        .collect::<Vec<_>>();
    if remaining.is_empty() && map.base.iter().any(|base| base.owner == "neutral") {
        return Err("没有可分配给中立据点的学科".into());
    }
    map.base
        .iter_mut()
        .filter(|base| base.owner == "neutral")
        .enumerate()
        .for_each(|(index, base)| base.subject = remaining[index % remaining.len()].clone());
    Ok(())
}

pub fn spawn_map(world: &mut World, map_path: &Path, subjects_dir: &Path) -> Result<(), String> {
    spawn_map_inner(world, map_path, subjects_dir, None, None, None)
}

/// 使用指定防御随机种子加载地图，供交换出生位的批量自博弈复用。
pub fn spawn_map_seeded(
    world: &mut World,
    map_path: &Path,
    subjects_dir: &Path,
    defense_seed: u64,
) -> Result<(), String> {
    spawn_map_inner(
        world,
        map_path,
        subjects_dir,
        None,
        None,
        Some(defense_seed),
    )
}

/// 生成地图实体，可覆盖玩家/AI 据点的学科（开局界面选择）；中立要塞自动分配剩余学科
pub fn spawn_map_custom(
    world: &mut World,
    map_path: &Path,
    subjects_dir: &Path,
    player_subject: Option<&str>,
    ai_subject: Option<&str>,
) -> Result<(), String> {
    spawn_map_inner(
        world,
        map_path,
        subjects_dir,
        player_subject,
        ai_subject,
        None,
    )
}

fn spawn_map_inner(
    world: &mut World,
    map_path: &Path,
    subjects_dir: &Path,
    player_subject: Option<&str>,
    ai_subject: Option<&str>,
    defense_seed: Option<u64>,
) -> Result<(), String> {
    let text = std::fs::read_to_string(map_path).map_err(|e| format!("读地图失败: {e}"))?;
    let mut mf: MapFile = toml::from_str(&text).map_err(|e| format!("解析地图失败: {e}"))?;
    let symmetry = validate_map_file(&mf)?;
    let subjects = load_subjects(subjects_dir)?;

    assign_custom_subjects(&mut mf, &subjects, player_subject, ai_subject)?;

    for base in &mf.base {
        let subject = subjects
            .get(&base.subject)
            .ok_or_else(|| format!("未知学科: {}", base.subject))?;
        if subject.knowledge_points.len() < base.linked_tiles.len() {
            return Err(format!(
                "学科 {} 只有 {} 个知识点，无法分配 {} 块关联地块",
                base.subject,
                subject.knowledge_points.len(),
                base.linked_tiles.len()
            ));
        }
    }

    let (w, h) = (mf.map.width, mf.map.height);

    // 阵营分配：player → 1，其余 owner 按名称稳定排序。
    let mut faction_of_owner = HashMap::new();
    faction_of_owner.insert("neutral".into(), NEUTRAL);
    let factions = faction_owner_names(&mf.base)
        .into_iter()
        .enumerate()
        .map(|(index, owner)| {
            let id = (index + 1) as FactionId;
            let base = mf
                .base
                .iter()
                .find(|base| base.owner == owner)
                .expect("阵营名称必定来自某个据点");
            let subject = &subjects[&base.subject];
            faction_of_owner.insert(owner.clone(), id);
            Faction {
                id,
                name: subject.name.clone(),
                color: parse_hex_color(&subject.color),
                is_player: owner == "player",
            }
        })
        .collect::<Vec<_>>();

    // 中梁判定：整行都是 '#' 的行
    let is_beam_row: Vec<bool> = mf
        .map
        .layout
        .iter()
        .map(|r| r.chars().all(|c| c == '#'))
        .collect();

    // 先在稠密数组上跑完旧的确定性生成，再统一生成实体
    let mut kinds = vec![CellKind::Void; w * h];
    let mut owners = vec![NEUTRAL; w * h];
    let mut garrison = vec![0.0f32; w * h];
    let mut garrison_max = vec![0.0f32; w * h];
    let mut labels: Vec<Option<String>> = vec![None; w * h];
    for (i, row) in mf.map.layout.iter().enumerate() {
        for (j, ch) in row.chars().enumerate() {
            if ch == '#' {
                kinds[i * w + j] = CellKind::Plain;
            }
        }
    }
    let idx = |x: usize, y: usize| y * w + x;
    let symmetry_key = |i: usize| {
        if symmetry == MapSymmetry::None {
            i
        } else {
            i.min(symmetric_index(i, w, h, symmetry))
        }
    };

    // 普通地块防御值：固定种子 + 对称成对赋值
    let mut rng = Rng(defense_seed.unwrap_or(mf.neutral.seed) | 1);
    let mut assigned: HashMap<CellIdx, f32> = HashMap::new();
    for i in 0..kinds.len() {
        if kinds[i] != CellKind::Plain {
            continue;
        }
        let key = symmetry_key(i);
        let v = *assigned.entry(key).or_insert_with(|| {
            if is_beam_row[i / w] {
                rng.range(mf.neutral.beam_defense_min, mf.neutral.beam_defense_max)
            } else {
                rng.range(mf.neutral.defense_min, mf.neutral.defense_max)
            }
        });
        let v = (v * 10.0).round() / 10.0;
        garrison[i] = v;
        garrison_max[i] = v;
    }

    // 据点与关联地块
    struct BaseSpawn {
        cell: CellIdx,
        subject_id: String,
        subject_name: String,
        production_base: f32,
        production_bonus_per_tile: f32,
        linked: Vec<CellIdx>,
    }
    // 知识点标签每局随机：系统时间播种的独立 RNG（防御值仍用固定种子，数值可复现）
    let mut label_rng = Rng(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15)
        | 1);

    let mut base_spawns = Vec::new();
    for b in mf.base.iter() {
        let subj = subjects
            .get(&b.subject)
            .ok_or_else(|| format!("未知学科: {}", b.subject))?;
        let ci = idx(b.pos.0, b.pos.1);
        let owner = faction_of_owner[&b.owner];
        kinds[ci] = CellKind::Base;
        owners[ci] = owner;
        garrison[ci] = b.garrison;
        // 中立要塞按"可回防"处理；有主据点驻军无上限、不回防
        garrison_max[ci] = if owner == NEUTRAL {
            b.garrison
        } else {
            f32::INFINITY
        };
        labels[ci] = Some(subj.name.clone());

        // 从词库洗牌后顺序抽取：本据点内知识点名称不重复
        let mut pool = subj.knowledge_points.clone();
        for k in (1..pool.len()).rev() {
            let j = (label_rng.next() % (k as u64 + 1)) as usize;
            pool.swap(j, k);
        }

        let mut linked = Vec::new();
        for (n, lr) in b.linked_tiles.iter().enumerate() {
            let li = idx(lr.pos.0, lr.pos.1);
            let key = symmetry_key(li);
            let v = *assigned.entry(key).or_insert_with(|| {
                rng.range(mf.linked_tile.defense_min, mf.linked_tile.defense_max)
            });
            let v = (v * 10.0).round() / 10.0;
            kinds[li] = CellKind::LinkedTile;
            // 初始场景下除据点外均为中立格子——关联地块也要先抢才有产能加成
            owners[li] = NEUTRAL;
            garrison[li] = v;
            garrison_max[li] = v;
            labels[li] = Some(pool.get(n).cloned().unwrap_or_else(|| "??".into()));
            linked.push(li);
        }

        base_spawns.push(BaseSpawn {
            cell: ci,
            subject_id: subj.id.clone(),
            subject_name: subj.name.clone(),
            production_base: b.production_base,
            production_bonus_per_tile: mf.linked_tile.production_bonus,
            linked,
        });
    }

    // 统一生成格子实体（按 idx 顺序，含虚空格——保证 GridLookup 下标稳定）
    let mut cell_entities = Vec::with_capacity(w * h);
    for i in 0..w * h {
        let e = world
            .spawn((
                kinds[i],
                Owner(owners[i]),
                Garrison {
                    cur: garrison[i],
                    max: garrison_max[i],
                },
                Label(labels[i].clone()),
            ))
            .id();
        cell_entities.push(e);
    }

    // 据点实体追加 Base 组件
    let mut base_entities = Vec::new();
    for bs in base_spawns {
        let e = cell_entities[bs.cell];
        world.entity_mut(e).insert(Base {
            subject_id: bs.subject_id,
            subject_name: bs.subject_name,
            production_base: bs.production_base,
            production_bonus_per_tile: bs.production_bonus_per_tile,
            linked: bs.linked,
        });
        base_entities.push(e);
    }

    world.insert_resource(GridLookup {
        width: w,
        height: h,
        cells: cell_entities,
    });
    world.insert_resource(BaseList(base_entities));
    world.insert_resource(Factions(factions));
    world.insert_resource(Rules {
        garrison_cap_base: mf.rules.garrison_cap_base,
        garrison_cap_per_tile: mf.rules.garrison_cap_per_tile,
        regen_per_sec: mf.rules.regen_per_sec,
        squad_interval_sec: mf.rules.squad_interval_sec,
        squad_max_size: mf.rules.squad_max_size,
        squad_growth_garrison_step: mf.rules.squad_growth_garrison_step,
        squad_soft_cap_garrison: mf.rules.squad_soft_cap_garrison,
        squad_move_sec_per_cell: mf.rules.squad_move_sec_per_cell,
    });
    Ok(())
}

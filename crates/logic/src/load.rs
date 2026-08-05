use crate::model::*;
use serde::Deserialize;
use std::collections::HashMap;
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
    knowledge_index: usize,
}

#[derive(Deserialize)]
struct NeutralDef {
    defense_min: f32,
    defense_max: f32,
    beam_defense_min: f32,
    beam_defense_max: f32,
    seed: u64,
    rotational_symmetry: bool,
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
    squad_move_sec_per_cell: f32,
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
        let text = std::fs::read_to_string(&path).map_err(|e| format!("读 {:?} 失败: {e}", path))?;
        let def: SubjectDef = toml::from_str(&text).map_err(|e| format!("解析 {:?} 失败: {e}", path))?;
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

// ---------- 构建 GameState ----------

pub fn build_game(map_path: &Path, subjects_dir: &Path) -> Result<GameState, String> {
    build_game_custom(map_path, subjects_dir, None, None)
}

/// 构建游戏，可覆盖玩家/AI 据点的学科（开局界面选择）；中立要塞自动分配剩余学科
pub fn build_game_custom(
    map_path: &Path,
    subjects_dir: &Path,
    player_subject: Option<&str>,
    ai_subject: Option<&str>,
) -> Result<GameState, String> {
    let text = std::fs::read_to_string(map_path).map_err(|e| format!("读地图失败: {e}"))?;
    let mut mf: MapFile = toml::from_str(&text).map_err(|e| format!("解析地图失败: {e}"))?;
    let subjects = load_subjects(subjects_dir)?;

    // 学科重排：玩家/AI 用选择的学科，中立据点按顺序分配剩余学科
    if player_subject.is_some() || ai_subject.is_some() {
        let mut used: Vec<String> = Vec::new();
        for b in mf.base.iter_mut() {
            match b.owner.as_str() {
                "player" => {
                    if let Some(s) = player_subject {
                        b.subject = s.into();
                        used.push(s.into());
                    }
                }
                "ai" => {
                    if let Some(s) = ai_subject {
                        b.subject = s.into();
                        used.push(s.into());
                    }
                }
                _ => {}
            }
        }
        let mut remaining: Vec<String> = subjects
            .keys()
            .filter(|id| !used.contains(id))
            .cloned()
            .collect();
        remaining.sort();
        let mut i = 0;
        for b in mf.base.iter_mut() {
            if b.owner == "neutral" && !remaining.is_empty() {
                b.subject = remaining[i % remaining.len()].clone();
                i += 1;
            }
        }
    }

    let (w, h) = (mf.map.width, mf.map.height);
    if mf.map.layout.len() != h || mf.map.layout.iter().any(|r| r.chars().count() != w) {
        return Err("地图 layout 尺寸与 width/height 不符".into());
    }

    // 阵营分配：player → 1，ai → 2（按出现顺序往后）
    let mut faction_of_owner: HashMap<String, FactionId> = HashMap::new();
    faction_of_owner.insert("neutral".into(), NEUTRAL);
    let mut factions: Vec<Faction> = Vec::new();
    for b in &mf.base {
        if b.owner == "neutral" || faction_of_owner.contains_key(&b.owner) {
            continue;
        }
        let id = faction_of_owner.len() as FactionId;
        faction_of_owner.insert(b.owner.clone(), id);
        let subj = subjects
            .get(&b.subject)
            .ok_or_else(|| format!("未知学科: {}", b.subject))?;
        factions.push(Faction {
            id,
            name: subj.name.clone(),
            color: parse_hex_color(&subj.color),
            is_player: b.owner == "player",
        });
    }

    // 中梁判定：整行都是 '#' 的行
    let is_beam_row: Vec<bool> = mf
        .map
        .layout
        .iter()
        .map(|r| r.chars().all(|c| c == '#'))
        .collect();

    // 基础格子
    let mut cells = Vec::with_capacity(w * h);
    for row in mf.map.layout.iter() {
        for ch in row.chars() {
            let kind = if ch == '#' { CellKind::Plain } else { CellKind::Void };
            cells.push(Cell {
                kind,
                owner: NEUTRAL,
                garrison: 0.0,
                garrison_max: 0.0,
                label: None,
            });
        }
    }
    let idx = |x: usize, y: usize| y * w + x;
    let sym = |i: usize| (h - 1 - i / w) * w + (w - 1 - i % w); // 180° 旋转对称格

    // 普通地块防御值：固定种子 + 对称成对赋值
    let mut rng = Rng(mf.neutral.seed | 1);
    let mut assigned: HashMap<CellIdx, f32> = HashMap::new();
    for i in 0..cells.len() {
        if cells[i].kind != CellKind::Plain {
            continue;
        }
        let key = if mf.neutral.rotational_symmetry { i.min(sym(i)) } else { i };
        let v = *assigned.entry(key).or_insert_with(|| {
            if is_beam_row[i / w] {
                rng.range(mf.neutral.beam_defense_min, mf.neutral.beam_defense_max)
            } else {
                rng.range(mf.neutral.defense_min, mf.neutral.defense_max)
            }
        });
        // 取整到 1 位小数，显示干净
        let v = (v * 10.0).round() / 10.0;
        cells[i].garrison = v;
        cells[i].garrison_max = v;
    }

    // 据点与关联地块
    let mut bases = Vec::new();
    let mut base_index = HashMap::new();
    for (bi, b) in mf.base.iter().enumerate() {
        let subj = subjects
            .get(&b.subject)
            .ok_or_else(|| format!("未知学科: {}", b.subject))?;
        let ci = idx(b.pos.0, b.pos.1);
        let owner = faction_of_owner[&b.owner];
        cells[ci].kind = CellKind::Base;
        cells[ci].owner = owner;
        cells[ci].garrison = b.garrison;
        // 中立要塞按"可回防"处理；有主据点驻军无上限、不回防
        cells[ci].garrison_max = if owner == NEUTRAL { b.garrison } else { f32::INFINITY };
        cells[ci].label = Some(subj.name.clone());
        base_index.insert(ci, bi);

        let mut linked = Vec::new();
        for lr in &b.linked_tiles {
            let li = idx(lr.pos.0, lr.pos.1);
            let key = if mf.neutral.rotational_symmetry { li.min(sym(li)) } else { li };
            let v = *assigned.entry(key).or_insert_with(|| {
                rng.range(mf.linked_tile.defense_min, mf.linked_tile.defense_max)
            });
            let v = (v * 10.0).round() / 10.0;
            cells[li].kind = CellKind::LinkedTile;
            // 初始场景下除据点外均为中立格子——关联地块也要先抢才有产能加成
            cells[li].owner = NEUTRAL;
            cells[li].garrison = v;
            cells[li].garrison_max = v;
            cells[li].label = Some(
                subj.knowledge_points
                    .get(lr.knowledge_index)
                    .cloned()
                    .unwrap_or_else(|| "??".into()),
            );
            linked.push(li);
        }

        bases.push(Base {
            cell: ci,
            subject_id: subj.id.clone(),
            subject_name: subj.name.clone(),
            production_base: b.production_base,
            production_bonus_per_tile: mf.linked_tile.production_bonus,
            linked,
        });
    }

    Ok(GameState {
        width: w,
        height: h,
        cells,
        bases,
        base_index,
        factions,
        squads: Vec::new(),
        streams: Vec::new(),
        rules: Rules {
            garrison_cap_base: mf.rules.garrison_cap_base,
            garrison_cap_per_tile: mf.rules.garrison_cap_per_tile,
            regen_per_sec: mf.rules.regen_per_sec,
            squad_interval_sec: mf.rules.squad_interval_sec,
            squad_max_size: mf.rules.squad_max_size,
            squad_move_sec_per_cell: mf.rules.squad_move_sec_per_cell,
        },
        time: 0.0,
        winner: None,
    })
}

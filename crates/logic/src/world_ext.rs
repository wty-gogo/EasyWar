//! World 存取辅助：Squad/Stream 实体按 seq 排序装卸。
//! 排序保证迭代顺序 = 生成顺序（确定性），与旧 Vec 语义一致。

use crate::components::*;
use bevy_ecs::prelude::*;

pub(crate) fn load_squads(world: &mut World) -> Vec<(Entity, Squad)> {
    let mut q = world.query::<(Entity, &Squad)>();
    let mut out: Vec<(Entity, Squad)> = q.iter(world).map(|(e, s)| (e, s.clone())).collect();
    out.sort_by_key(|(_, s)| s.seq);
    out
}

pub(crate) fn load_streams(world: &mut World) -> Vec<(Entity, Stream)> {
    let mut q = world.query::<(Entity, &Stream)>();
    let mut out: Vec<(Entity, Stream)> = q.iter(world).map(|(e, s)| (e, s.clone())).collect();
    out.sort_by_key(|(_, s)| s.seq);
    out
}

/// 逐字段写回（仅在变化时触发变更检测）
pub(crate) fn write_squad(world: &mut World, entity: Entity, squad: &Squad) {
    let mut em = world.entity_mut(entity);
    if let Some(mut c) = em.get_mut::<Squad>() {
        if *c != *squad {
            *c = squad.clone();
        }
    }
}

pub(crate) fn write_stream(world: &mut World, entity: Entity, stream: &Stream) {
    let mut em = world.entity_mut(entity);
    if let Some(mut c) = em.get_mut::<Stream>() {
        if *c != *stream {
            *c = stream.clone();
        }
    }
}

//! 无头冒烟入口：MinimalPlugins 都不需要——App + GamePlugin，手动驱动 SimTick。
//! 用法：`cargo run -p easywar-logic --bin headless -- [秒数]`

use bevy_app::App;
use easywar_logic::*;
use std::path::PathBuf;

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets")
}

fn main() {
    let secs: f32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(600.0);

    let mut app = App::new();
    app.add_plugins(GamePlugin);
    spawn_map(
        app.world_mut(),
        &assets_dir().join("maps/h_1v1.toml"),
        &assets_dir().join("subjects"),
    )
    .expect("地图加载失败");
    app.world_mut().insert_resource(AiControllers(vec![
        AiController::new(1, AiParams::normal()),
        AiController::new(2, AiParams::normal()),
    ]));

    {
        let world = app.world_mut();
        let lookup = world.resource::<GridLookup>();
        let bases = world.resource::<BaseList>();
        println!(
            "[headless] 地图加载成功：{}x{}，据点 {} 个；AI vs AI 开打",
            lookup.width,
            lookup.height,
            bases.0.len()
        );
    }

    let steps = (secs / SIM_DT) as usize;
    let report_every = (60.0 / SIM_DT) as usize;
    for i in 0..steps {
        app.world_mut()
            .try_run_schedule(SimTick)
            .expect("SimTick 未注册");
        if i % report_every == 0 {
            let world = app.world_mut();
            let clock = world.resource::<GameClock>().time;
            let winner = world.resource::<Winner>().0;
            let t1 = total_troops(world, 1);
            let t2 = total_troops(world, 2);
            let mut q = world.query::<&Squad>();
            let squads = q.iter(world).count();
            println!(
                "[headless] t={:>5.1}s 语文总兵 {:>6.1} 数学总兵 {:>6.1} 小队 {} 胜者 {:?}",
                clock, t1, t2, squads, winner
            );
        }
        if app.world().resource::<Winner>().0.is_some() {
            break;
        }
    }
    let world = app.world_mut();
    println!(
        "[headless] 结束：t={:.1}s winner={:?}",
        world.resource::<GameClock>().time,
        world.resource::<Winner>().0
    );
}

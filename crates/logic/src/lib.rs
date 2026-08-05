//! EasyWar 逻辑层：纯逻辑，不依赖任何渲染/引擎 crate。
//! 可无头运行、可变速 tick —— 服务于游戏本体、AI 模拟与未来的 RL 训练。

pub mod ai;
pub mod load;
pub mod model;
pub mod sim;

pub use ai::{AiCommand, AiController, AiParams};
pub use load::{build_game, build_game_custom, load_subjects, parse_hex_color, SubjectDef};
pub use model::*;

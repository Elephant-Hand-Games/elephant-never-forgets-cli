pub mod chunker;
pub mod cli;
pub mod config;
pub mod db;
pub mod discovery;
pub mod embed;
pub mod errors;
pub mod extract;
pub mod index;
pub mod models;
#[cfg(feature = "native-candle")]
mod native_candle;
#[cfg(feature = "native-candle")]
mod native_gemma3;
pub mod output;
pub mod providers;
pub mod ranking;
pub mod search;
pub mod update;

pub fn run() -> anyhow::Result<()> {
    cli::run()
}

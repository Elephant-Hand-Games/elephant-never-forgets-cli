use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::{config, index, models, output, search};

#[derive(Debug, Parser)]
#[command(
    name = "enf",
    version,
    about = "Elephant Never Forgets: native-first semantic search for local projects",
    long_about = "Elephant Never Forgets indexes local docs, repos, directories, and agent workflow files into SQLite for semantic and keyword search."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Init(InitArgs),
    Models(ModelsArgs),
    Index(IndexArgs),
    Search(SearchArgs),
    Retrieve(SearchArgs),
    Status(StatusArgs),
    Doctor(DoctorArgs),
    Ci(CiArgs),
}

#[derive(Debug, Args)]
pub struct InitArgs {
    #[arg(long)]
    pub db: bool,
    #[arg(long)]
    pub native_embed: bool,
    #[arg(long, alias = "local-embed")]
    pub local_embed: bool,
    #[arg(long, value_enum)]
    pub provider: Option<ProviderArg>,
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long, value_enum)]
    pub variant: Option<ModelVariantArg>,
    #[arg(long, value_enum)]
    pub model_cache: Option<ModelCacheArg>,
    #[arg(long)]
    pub install_models: bool,
    #[arg(long)]
    pub index: bool,
    #[arg(long)]
    pub force: bool,
}

#[derive(Debug, Args)]
pub struct ModelsArgs {
    #[command(subcommand)]
    pub command: ModelsCommand,
}

#[derive(Debug, Subcommand)]
pub enum ModelsCommand {
    List(JsonArgs),
    Current(JsonArgs),
    Install(ModelInstallArgs),
    CachePath(JsonArgs),
    Gc(JsonArgs),
}

#[derive(Debug, Args)]
pub struct ModelInstallArgs {
    pub model: Option<String>,
    #[arg(long, value_enum)]
    pub variant: Option<ModelVariantArg>,
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args)]
pub struct IndexArgs {
    #[arg(default_value = ".")]
    pub path: PathBuf,
    #[arg(long)]
    pub reembed: bool,
    #[arg(long)]
    pub changed_only: bool,
    #[arg(long)]
    pub install_models: bool,
    #[command(flatten)]
    pub provider: ProviderOverrideArgs,
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args, Clone)]
pub struct SearchArgs {
    pub query: String,
    #[arg(long, value_enum)]
    pub mode: Option<SearchModeArg>,
    #[arg(long, value_enum)]
    pub level: Option<SearchLevelArg>,
    #[arg(long)]
    pub limit: Option<usize>,
    #[arg(long)]
    pub cached_query_only: bool,
    #[command(flatten)]
    pub provider: ProviderOverrideArgs,
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args)]
pub struct StatusArgs {
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args)]
pub struct DoctorArgs {
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args)]
pub struct CiArgs {
    #[arg(long)]
    pub no_embed: bool,
    #[arg(long)]
    pub install_models: bool,
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args, Clone)]
pub struct ProviderOverrideArgs {
    #[arg(long, value_enum)]
    pub provider: Option<ProviderArg>,
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long, value_enum)]
    pub variant: Option<ModelVariantArg>,
    #[arg(long)]
    pub endpoint: Option<String>,
    #[arg(long)]
    pub api_key_env: Option<String>,
    #[arg(long)]
    pub dimensions: Option<usize>,
}

#[derive(Debug, Args)]
pub struct JsonArgs {
    #[arg(long)]
    pub json: bool,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum ProviderArg {
    Native,
    Ollama,
    Openai,
    OpenaiCompatible,
    Http,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum ModelVariantArg {
    Quantized,
    Full,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum ModelCacheArg {
    Global,
    Project,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum SearchModeArg {
    Hybrid,
    Vector,
    Keyword,
}

#[derive(Clone, Debug, ValueEnum)]
pub enum SearchLevelArg {
    Chunk,
    File,
    Both,
}

pub fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Init(args) => config::init(args),
        Command::Models(args) => models::run(args),
        Command::Index(args) => index::run(args),
        Command::Search(args) => search::run(args, false),
        Command::Retrieve(args) => search::run(args, true),
        Command::Status(args) => output::status(args),
        Command::Doctor(args) => output::doctor(args),
        Command::Ci(args) => output::ci(args),
    }
}

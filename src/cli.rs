use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::{config, index, models, output, search, setup, update};

#[derive(Debug, Parser)]
#[command(
    name = "enf",
    version,
    about = "Semantic search and RAG retrieval for local projects",
    long_about = "Elephant Never Forgets\n\nSemantic search and RAG retrieval for local projects.\n\nStart here:\n  enf init                 Set up this project interactively\n  enf index .              Index the current directory\n  enf search \"query\"       Search indexed files\n\nCommon workflows:\n  enf init --yes --index   Set up and index now\n  enf setup                Change provider, model, reranker, or images\n  enf retrieve \"query\"     Return JSON chunks for RAG/agents\n  enf doctor               Diagnose setup problems",
    after_help = "Examples:\n  enf init --preset local --index --yes\n  enf search \"save system\" --filetype md\n  enf retrieve \"combat formulas\" --limit 12\n  enf setup reranker\n\nMore help:\n  enf help init\n  enf help setup\n  enf help search"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Init(InitArgs),
    Setup(SetupArgs),
    Config(ConfigArgs),
    Models(ModelsArgs),
    Index(IndexArgs),
    #[command(hide = true)]
    Add(IndexArgs),
    #[command(hide = true)]
    Remove(RemoveArgs),
    #[command(alias = "find")]
    Search(SearchArgs),
    Retrieve(SearchArgs),
    Status(StatusArgs),
    Doctor(DoctorArgs),
    #[command(hide = true)]
    Ci(CiArgs),
    Update(UpdateArgs),
}

#[derive(Debug, Args)]
/// Configure a new project database and embedding profile.
///
/// Gemma is recommended for code-oriented documents, while Nomic is often a better
/// default for larger documents and no-chunking prose workflows.
pub struct InitArgs {
    #[arg(long, default_value = "sqlite", default_missing_value = "sqlite", num_args = 0..=1, hide_possible_values = true, hide = true)]
    pub db: DbArg,
    #[arg(long)]
    pub no_db: bool,
    #[arg(long, value_enum)]
    pub preset: Option<PresetArg>,
    #[arg(long)]
    pub native_embed: bool,
    #[arg(long, alias = "local-embed")]
    pub local_embed: bool,
    #[arg(long, value_enum)]
    pub provider: Option<ProviderArg>,
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long, value_enum)]
    pub chunking: Option<ChunkingModeArg>,
    #[arg(long, value_enum)]
    pub variant: Option<ModelVariantArg>,
    #[arg(long)]
    pub endpoint: Option<String>,
    #[arg(long)]
    pub api_key_env: Option<String>,
    #[arg(long)]
    pub dimensions: Option<usize>,
    #[arg(long, value_enum)]
    pub fallback_provider: Option<ProviderArg>,
    #[arg(long)]
    pub fallback_endpoint: Option<String>,
    #[arg(long)]
    pub fallback_api_key_env: Option<String>,
    #[arg(long, value_enum)]
    pub model_cache: Option<ModelCacheArg>,
    #[arg(long, help = "Ask for profile settings interactively")]
    pub interactive: bool,
    #[arg(
        long,
        help = "Never prompt; fail or use supplied/default flags instead"
    )]
    pub no_input: bool,
    #[arg(short = 'y', long, help = "Accept safe defaults and confirmations")]
    pub yes: bool,
    #[arg(long)]
    pub install_models: bool,
    #[arg(long)]
    pub index: bool,
    #[arg(long)]
    pub force: bool,
    #[arg(long)]
    pub dry_run: bool,
}

#[derive(Debug, Args)]
pub struct SetupArgs {
    #[command(subcommand)]
    pub command: Option<SetupCommand>,
}

#[derive(Debug, Subcommand)]
pub enum SetupCommand {
    Presets(JsonArgs),
    Preset(PresetDetailArgs),
    Use(SetupUseArgs),
    Reranker(SetupRerankerArgs),
    Images(SetupImagesArgs),
    Fallback(SetupFallbackArgs),
    Search(SetupSearchArgs),
}

#[derive(Debug, Args)]
pub struct PresetDetailArgs {
    #[arg(value_enum)]
    pub preset: PresetArg,
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args)]
pub struct SetupUseArgs {
    #[arg(value_enum)]
    pub preset: PresetArg,
    #[arg(long)]
    pub endpoint: Option<String>,
    #[arg(long)]
    pub api_key_env: Option<String>,
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long)]
    pub dimensions: Option<usize>,
    #[arg(short = 'y', long)]
    pub yes: bool,
    #[arg(long)]
    pub dry_run: bool,
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args)]
pub struct SetupRerankerArgs {
    #[arg(long)]
    pub endpoint: Option<String>,
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long)]
    pub candidate_limit: Option<usize>,
    #[arg(long)]
    pub off: bool,
    #[arg(long)]
    pub schema: bool,
    #[arg(short = 'y', long)]
    pub yes: bool,
    #[arg(long)]
    pub dry_run: bool,
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args)]
pub struct SetupImagesArgs {
    #[arg(long)]
    pub endpoint: Option<String>,
    #[arg(long)]
    pub query_endpoint: Option<String>,
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long)]
    pub dimensions: Option<usize>,
    #[arg(long)]
    pub off: bool,
    #[arg(short = 'y', long)]
    pub yes: bool,
    #[arg(long)]
    pub dry_run: bool,
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args)]
pub struct SetupFallbackArgs {
    #[arg(long, value_enum)]
    pub provider: Option<ProviderArg>,
    #[arg(long)]
    pub endpoint: Option<String>,
    #[arg(long)]
    pub api_key_env: Option<String>,
    #[arg(long)]
    pub off: bool,
    #[arg(long)]
    pub dry_run: bool,
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args)]
pub struct SetupSearchArgs {
    #[arg(long, value_enum)]
    pub mode: Option<SearchModeArg>,
    #[arg(long, value_enum)]
    pub level: Option<SearchLevelArg>,
    #[arg(long)]
    pub limit: Option<usize>,
    #[arg(long)]
    pub dry_run: bool,
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args)]
pub struct ConfigArgs {
    #[command(subcommand)]
    pub command: Option<ConfigCommand>,
}

#[derive(Debug, Subcommand)]
pub enum ConfigCommand {
    Path(JsonArgs),
    Show(ConfigShowArgs),
    Explain(ConfigExplainArgs),
    Validate(JsonArgs),
    Edit,
    Set(ConfigSetArgs),
    Diff(ConfigDiffArgs),
    Doctor(JsonArgs),
}

#[derive(Debug, Args)]
pub struct ConfigShowArgs {
    pub section: Option<String>,
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args)]
pub struct ConfigExplainArgs {
    pub key: Option<String>,
}

#[derive(Debug, Args)]
pub struct ConfigSetArgs {
    pub key: String,
    pub value: String,
    #[arg(long)]
    pub dry_run: bool,
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Args)]
pub struct ConfigDiffArgs {
    #[arg(long, value_enum)]
    pub preset: PresetArg,
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
    Path(JsonArgs),
    Clean(ModelGcArgs),
    #[command(hide = true)]
    CachePath(JsonArgs),
    #[command(hide = true)]
    Gc(ModelGcArgs),
}

#[derive(Debug, Args)]
pub struct ModelInstallArgs {
    pub model: Option<String>,
    #[arg(long, value_enum)]
    pub variant: Option<ModelVariantArg>,
    #[arg(long)]
    pub json: bool,
    #[arg(long)]
    pub dry_run: bool,
}

#[derive(Debug, Args)]
pub struct ModelGcArgs {
    #[arg(long)]
    pub json: bool,
    #[arg(long)]
    pub dry_run: bool,
}

#[derive(Debug, Args)]
pub struct IndexArgs {
    #[arg(default_value = ".")]
    pub path: PathBuf,
    #[arg(long)]
    pub reembed: bool,
    #[arg(long = "changed", alias = "changed-only")]
    pub changed_only: bool,
    #[arg(long)]
    pub install_models: bool,
    #[arg(long = "no-embeddings", alias = "no-embed")]
    pub no_embed: bool,
    #[command(flatten)]
    pub provider: ProviderOverrideArgs,
    #[arg(long)]
    pub json: bool,
    #[arg(long)]
    pub dry_run: bool,
}

#[derive(Debug, Args)]
pub struct RemoveArgs {
    pub path: PathBuf,
    #[arg(long)]
    pub dry_run: bool,
}

#[derive(Debug, Args, Clone)]
pub struct SearchArgs {
    pub query: String,
    #[arg(long, value_enum)]
    pub mode: Option<SearchModeArg>,
    #[arg(long, value_enum)]
    pub level: Option<SearchLevelArg>,
    #[arg(long, value_enum, default_value = "all")]
    pub kind: SearchKindArg,
    #[arg(long = "filetype")]
    pub filetypes: Vec<String>,
    #[arg(long = "path")]
    pub paths: Vec<String>,
    #[arg(long)]
    pub limit: Option<usize>,
    #[arg(long)]
    pub cached_query_only: bool,
    #[arg(long)]
    pub compact: bool,
    #[arg(long)]
    pub full: bool,
    #[arg(long)]
    pub explain: bool,
    #[arg(long)]
    pub jsonl: bool,
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
    pub ci: bool,
    #[arg(long)]
    pub fix: bool,
    #[arg(short = 'y', long)]
    pub yes: bool,
    #[arg(long)]
    pub check: Option<String>,
    #[arg(long = "no-embeddings", alias = "no-embed")]
    pub no_embeddings: bool,
    #[arg(long)]
    pub install_models: bool,
    #[arg(long)]
    pub dry_run: bool,
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
    #[arg(long)]
    pub dry_run: bool,
}

#[derive(Debug, Args)]
pub struct UpdateArgs {
    #[arg(long)]
    pub version: Option<String>,
    #[arg(long)]
    pub install_dir: Option<String>,
    #[arg(long)]
    pub method: Option<String>,
    #[arg(long)]
    pub repo: Option<String>,
    #[arg(long)]
    pub dry_run: bool,
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

#[derive(Clone, Debug, ValueEnum, PartialEq, Eq)]
pub enum PresetArg {
    Local,
    Code,
    Docs,
    Ollama,
    Openai,
    Custom,
    Keyword,
}

#[derive(Clone, Debug, ValueEnum, PartialEq, Eq)]
pub enum DbArg {
    Sqlite,
    False,
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

#[derive(Clone, Debug, ValueEnum, PartialEq, Eq)]
pub enum SearchKindArg {
    All,
    Text,
    Image,
}

#[derive(Clone, Debug, ValueEnum, PartialEq, Eq)]
pub enum ChunkingModeArg {
    Smart,
    LineWindow,
    Off,
}

pub fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Init(args) => config::init(args),
        Command::Setup(args) => setup::run_setup(args),
        Command::Config(args) => setup::run_config(args),
        Command::Models(args) => models::run(args),
        Command::Index(args) => index::run(args),
        Command::Add(args) => index::run_add(args),
        Command::Remove(args) => index::run_remove(args),
        Command::Search(args) => search::run(args, false),
        Command::Retrieve(args) => search::run(args, true),
        Command::Status(args) => output::status(args),
        Command::Doctor(args) => output::doctor(args),
        Command::Ci(args) => output::ci(args),
        Command::Update(args) => update::run(args),
    }
}

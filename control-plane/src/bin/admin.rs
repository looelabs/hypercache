use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use hypercache_control_plane::config::Config;
use hypercache_control_plane::controller;
use hypercache_control_plane::fleet;

#[derive(Parser, Debug)]
#[command(
    name = "hypercache-admin",
    author = "Looe Labs",
    version = env!("CARGO_PKG_VERSION"),
    about = "HyperCache Central Cluster Controller & Fleet Administration Suite",
    long_about = "Dedicated administrative CLI for managing multi-node HyperCache AI clusters. Provides cluster topology visibility, node health monitoring, and parallel fleet-wide model pre-loading and eviction."
)]
struct AdminCli {
    /// Path to TOML configuration file (e.g. -c /path/to/hypercache-controller.toml)
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    /// Controller API endpoint URL (e.g. http://127.0.0.1:9120)
    #[arg(long, global = true, env = "HYPERCACHE_CONTROLLER_URL")]
    controller: Option<String>,

    /// Cluster Admin Authentication Token (or via HYPERCACHE_AUTH_TOKEN env)
    #[arg(long, global = true, env = "HYPERCACHE_AUTH_TOKEN")]
    token: Option<String>,

    #[command(subcommand)]
    command: AdminCommands,
}

#[derive(Subcommand, Debug)]
enum AdminCommands {
    /// Start the Central HyperCache Cluster Controller / Fleet Coordinator daemon
    Controller {
        /// Network interface IP address to bind the controller service on (e.g. 10.0.0.10 or 0.0.0.0)
        #[arg(long, short = 'b')]
        bind: Option<String>,

        /// Port to bind the controller service on (default: 9120)
        #[arg(long, short = 'p')]
        port: Option<u16>,
    },

    /// Inspect active cluster topology, storage capacity, and node status
    Nodes {
        /// Output node inventory in JSON format
        #[arg(long)]
        json: bool,
    },

    /// Pre-load / pre-warm model checkpoints across the cluster in parallel
    Preload {
        /// URI of model checkpoint to pre-load (e.g. s3://bucket/model.safetensors or file:///mnt/nfs/model.bin)
        uri: String,

        /// Target node IDs (comma-separated, e.g. --nodes "node-01,node-02" or --nodes all)
        #[arg(long, default_value = "all")]
        nodes: String,

        /// Target nodes by label selector (e.g. --selector "gpu=h100,pool=training")
        #[arg(long)]
        selector: Option<String>,
    },

    /// Pre-load multiple shards or an entire dataset across the cluster with optional worker sharding
    PreloadBatch {
        /// S3 URI prefix of dataset shards (e.g. s3://bucket/datasets/c4/ or hypercache://datasets/tokens/)
        #[arg(long)]
        prefix: Option<String>,

        /// Comma-separated list of explicit URIs to pre-load
        #[arg(long, value_delimiter = ',')]
        uris: Vec<String>,

        /// Automatically partition/shard dataset objects evenly among target GPU workers (i % N)
        #[arg(long)]
        sharded: bool,

        /// Target node IDs (comma-separated, e.g. --nodes "node-01,node-02" or --nodes all)
        #[arg(long, default_value = "all")]
        nodes: String,

        /// Target nodes by label selector (e.g. --selector "gpu=h100,pool=training")
        #[arg(long)]
        selector: Option<String>,
    },

    /// Evict / remove a cached model from GPU nodes across the cluster in parallel
    Evict {
        /// URI of model checkpoint to evict
        uri: String,

        /// Target node IDs (comma-separated, e.g. --nodes "node-01,node-02" or --nodes all)
        #[arg(long, default_value = "all")]
        nodes: String,

        /// Target nodes by label selector (e.g. --selector "gpu=h100,pool=training")
        #[arg(long)]
        selector: Option<String>,
    },

    /// Wipe cached memory and reset SSD block storage across all nodes in the cluster
    Purge {
        /// Target node IDs (comma-separated, e.g. --nodes "node-01,node-02" or --nodes all)
        #[arg(long, default_value = "all")]
        nodes: String,

        /// Target nodes by label selector (e.g. --selector "gpu=h100,pool=training")
        #[arg(long)]
        selector: Option<String>,
    },

    /// Display live telemetry metrics and hit ratio counters across the cluster
    Status,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "hypercache=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = AdminCli::parse();

    let config = if let Some(ref p) = cli.config {
        if p.exists() {
            let canonical = p.canonicalize().unwrap_or_else(|_| p.clone());
            tracing::info!("Loaded configuration from: {}", canonical.display());
            Config::load_from_file(p).unwrap_or_default()
        } else {
            Config::default()
        }
    } else {
        Config::default()
    };

    let controller_url = cli.controller.or(config.controller_url).unwrap_or_else(|| {
        let host = if config.controller_bind_address == "0.0.0.0" {
            "127.0.0.1".to_string()
        } else {
            config.controller_bind_address.clone()
        };
        format!("http://{}:{}", host, config.controller_port)
    });

    let auth_token = cli.token.or(config.cluster_auth_token);

    match cli.command {
        AdminCommands::Controller { bind, port } => {
            let path = match cli.config {
                Some(p) => {
                    if !p.exists() {
                        eprintln!("====================================================================================================");
                        eprintln!("Error: Configuration file not found at {:?}", p);
                        eprintln!("====================================================================================================");
                        eprintln!("Please specify a valid configuration file using -c <path>:");
                        eprintln!("  ./hypercache-admin -c /path/to/hypercache-controller.toml controller");
                        eprintln!("====================================================================================================");
                        return Err(anyhow::anyhow!("Configuration file {:?} does not exist", p));
                    }
                    p
                }
                None => {
                    eprintln!("====================================================================================================");
                    eprintln!("Error: No configuration file specified!");
                    eprintln!("====================================================================================================");
                    eprintln!("Please specify the configuration file using -c <path>:");
                    eprintln!("  ./hypercache-admin -c /path/to/hypercache-controller.toml controller");
                    eprintln!("====================================================================================================");
                    return Err(anyhow::anyhow!(
                        "Missing required configuration file. Please provide -c <path>."
                    ));
                }
            };
            let canonical_path = path.canonicalize().unwrap_or_else(|_| path.clone());
            tracing::info!("Loaded controller configuration from: {}", canonical_path.display());
            let config = Config::load_from_file(&path)?;
            let bind_addr = bind.unwrap_or_else(|| config.controller_bind_address.clone());
            let p = port.unwrap_or(config.controller_port);
            controller::run_controller_server_with_config(&bind_addr, p, auth_token, config).await
        }
        AdminCommands::Nodes { json } => fleet::run_fleet_nodes(&controller_url, auth_token, json).await,
        AdminCommands::Preload { uri, nodes, selector } => {
            fleet::run_fleet_preload(&controller_url, auth_token, &uri, Some(nodes), selector).await
        }
        AdminCommands::PreloadBatch {
            prefix,
            uris,
            sharded,
            nodes,
            selector,
        } => {
            fleet::run_fleet_preload_batch(
                &controller_url,
                auth_token,
                prefix,
                uris,
                Some(nodes),
                selector,
                sharded,
            )
            .await
        }
        AdminCommands::Evict { uri, nodes, selector } => {
            fleet::run_fleet_evict(&controller_url, auth_token, &uri, Some(nodes), selector).await
        }
        AdminCommands::Purge { nodes, selector } => {
            fleet::run_fleet_purge(&controller_url, auth_token, Some(nodes), selector).await
        }
        AdminCommands::Status => fleet::run_fleet_nodes(&controller_url, auth_token, false).await,
    }
}

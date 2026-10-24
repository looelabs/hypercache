mod agent_server;
mod bench;
mod check;
mod config;
#[allow(dead_code)]
mod controller;
mod discovery;
mod ingestion;
mod metadata;
mod protocol;
mod ram;
mod server;
mod telemetry;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use fs2::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use bench::S3BenchConfig;
use config::Config;
use discovery::DeviceDetector;
use ingestion::IngestionEngine;
use metadata::MetadataStore;
use protocol::{Request, Response, ResponseStatus};
use server::Server;
use telemetry::TelemetryCollector;

#[derive(Parser, Debug)]
#[command(
    name = "hypercache",
    author = "HyperCache Team",
    version = "0.9.0-beta",
    about = "HyperCache: High-Performance Host-Based AI Storage & Cache Acceleration Engine",
    long_about = "HyperCache is a host-based caching engine designed to accelerate AI model loading (PyTorch, Safetensors, HuggingFace) by striping raw NVMe/SSD block devices and serving directly via zero-copy DMA / GPUDirect Storage."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Path to TOML configuration file (e.g. -c /path/to/hypercache-agent.toml)
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    /// Explicit block devices to use (comma-separated, e.g. --devices /dev/vdc,/dev/vdd). If not specified, 'auto' discovery mode is used.
    #[arg(short, long, env = "HYPERCACHE_DEVICES", global = true, value_delimiter = ',')]
    devices: Option<Vec<String>>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start the HyperCache Control Plane Daemon
    Start {
        /// Enable Prometheus HTTP metrics exporter (off by default)
        #[arg(long)]
        metrics: bool,

        /// Custom port for Prometheus metrics (implies --metrics)
        #[arg(long, env = "METRICS_PORT")]
        metrics_port: Option<u16>,

        /// Enable Agent Fleet HTTP API on this node (off by default unless controller_url is set)
        #[arg(long)]
        agent: bool,

        /// Custom port for Agent Fleet API (implies --agent)
        #[arg(long, env = "AGENT_PORT")]
        agent_port: Option<u16>,

        /// Explicitly enable Tier 1 Host RAM cache (off by default, can also be enabled in config)
        #[arg(long)]
        ram: bool,

        /// Override RAM cache capacity in bytes (e.g. 34359738368 for 32 GiB)
        #[arg(long, env = "RAM_CAPACITY_BYTES")]
        ram_capacity_bytes: Option<u64>,
    },

    /// Run the native end-to-end performance and optional live S3 comparison benchmark
    Bench {
        #[arg(long)]
        uri: Option<String>,

        #[arg(long, default_value_t = 100)]
        size_mb: u64,

        #[arg(long, default_value_t = 3)]
        iterations: u32,

        /// Live S3 Endpoint URL to benchmark against (e.g. https://s3.us-east-1.amazonaws.com)
        #[arg(long, env = "S3_ENDPOINT")]
        s3_endpoint: Option<String>,

        /// Target S3 Bucket Name (e.g. my-model-bucket)
        #[arg(long, env = "S3_BUCKET")]
        bucket: Option<String>,

        /// S3 Access Key ID (or via AWS_ACCESS_KEY_ID env)
        #[arg(long, env = "AWS_ACCESS_KEY_ID")]
        access_key: Option<String>,

        /// S3 Secret Access Key (or via AWS_SECRET_ACCESS_KEY env)
        #[arg(long, env = "AWS_SECRET_ACCESS_KEY")]
        secret_key: Option<String>,

        /// AWS / S3 Region (default: us-east-1)
        #[arg(long, env = "AWS_DEFAULT_REGION", default_value = "us-east-1")]
        region: String,

        /// Target storage tier for benchmark ('ram', 'nvme', or 'auto')
        #[arg(long, default_value = "nvme")]
        tier: String,
    },

    /// Inspect active storage pool, discovered NVMe/SSDs, and safety exclusions
    Inspect,

    /// Run pre-flight host, storage pool, shared memory, and configuration readiness checks
    #[command(aliases = ["doctor", "preflight"])]
    Check {
        /// Output check report in JSON format
        #[arg(long)]
        json: bool,

        /// Include Tier 1 Host RAM cache enabled override in readiness evaluation
        #[arg(long)]
        ram: bool,
    },

    /// List all cached objects, sizes, chunk counts, and striped drive locations
    #[command(alias = "list")]
    Ls {
        /// Output cached inventory in JSON format
        #[arg(long)]
        json: bool,
    },

    /// Evict/remove a specific cached model object from memory and NVMe persistence
    #[command(alias = "evict")]
    Rm {
        /// Target object URI (e.g. hypercache://models/llama-3/weights.safetensors)
        uri: String,
    },

    /// Refresh/re-ingest a model from authoritative S3, updating the local striped SSD cache
    Refresh {
        /// Target object URI to refresh (e.g. hypercache://models/llama-3/weights.safetensors)
        uri: String,
    },

    /// Pre-load / pre-warm model checkpoints from S3 or NFS into the local NVMe or RAM storage pool
    #[command(alias = "pull")]
    Preload {
        /// Target object URI to pre-load (e.g. hypercache://s3/bucket/models/llama-3/weights.safetensors)
        uri: Option<String>,

        /// Path to a text file containing multiple model URIs to pre-load (one per line)
        #[arg(short, long)]
        file: Option<PathBuf>,

        /// Target storage tier ('ram', 'nvme', or 'auto')
        #[arg(long, default_value = "auto")]
        tier: String,

        /// Pin pre-loaded objects in cache so they are never evicted by W-TinyLFU
        #[arg(long)]
        pin: bool,
    },

    /// Purge all cached models from memory, NVMe persistence, and reset drive allocation
    #[command(alias = "clean")]
    Purge,

    /// Query the status and live performance metrics of the running daemon
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

    let cli = Cli::parse();

    match cli.command.unwrap_or(Commands::Start {
        metrics: false,
        metrics_port: None,
        agent: false,
        agent_port: None,
        ram: false,
        ram_capacity_bytes: None,
    }) {
        Commands::Start {
            metrics,
            metrics_port,
            agent,
            agent_port,
            ram,
            ram_capacity_bytes,
        } => {
            run_daemon(
                cli.config,
                cli.devices,
                metrics,
                metrics_port,
                agent,
                agent_port,
                ram,
                ram_capacity_bytes,
            )
            .await
        }
        Commands::Bench {
            uri,
            size_mb,
            iterations,
            s3_endpoint,
            bucket,
            access_key,
            secret_key,
            region,
            tier,
        } => {
            let sock = config::probe_active_socket_path().to_string_lossy().to_string();
            let s3_config = s3_endpoint.map(|ep| S3BenchConfig {
                endpoint: ep,
                bucket,
                access_key,
                secret_key,
                region,
            });
            let target_tier = match tier.to_lowercase().as_str() {
                "ram" | "shm" => Some(protocol::StorageTier::Ram),
                "nvme" | "ssd" | "disk" => Some(protocol::StorageTier::Nvme),
                _ => None,
            };
            bench::run_benchmark(&sock, uri, size_mb, iterations, s3_config, target_tier).await
        }
        Commands::Inspect => {
            run_inspect(cli.config, cli.devices).await
        }
        Commands::Check { json, ram } => {
            run_check(cli.config, cli.devices, json, ram).await
        }
        Commands::Ls { json } => {
            let sock = config::probe_active_socket_path().to_string_lossy().to_string();
            run_ls(&sock, json).await
        }
        Commands::Rm { uri } => {
            let sock = config::probe_active_socket_path().to_string_lossy().to_string();
            run_rm(&sock, &uri).await
        }
        Commands::Refresh { uri } => {
            let sock = config::probe_active_socket_path().to_string_lossy().to_string();
            run_refresh(&sock, &uri).await
        }
        Commands::Preload { uri, file, tier, pin } => {
            let sock = config::probe_active_socket_path().to_string_lossy().to_string();
            let target_tier = match tier.to_lowercase().as_str() {
                "ram" | "shm" => Some(protocol::StorageTier::Ram),
                "nvme" | "ssd" | "disk" => Some(protocol::StorageTier::Nvme),
                _ => None,
            };
            run_preload(&sock, uri, file, target_tier, pin).await
        }
        Commands::Purge => {
            let sock = config::probe_active_socket_path().to_string_lossy().to_string();
            run_purge(&sock).await
        }
        Commands::Status => {
            let sock = config::probe_active_socket_path().to_string_lossy().to_string();
            run_status(&sock).await
        }
    }
}

async fn run_daemon(
    config_path: Option<PathBuf>,
    explicit_devices: Option<Vec<String>>,
    enable_metrics_flag: bool,
    custom_metrics_port: Option<u16>,
    enable_agent_flag: bool,
    custom_agent_port: Option<u16>,
    enable_ram_flag: bool,
    custom_ram_capacity: Option<u64>,
) -> Result<()> {
    println!("============================================================");
    println!("        HYPERCACHE HIGH-SPEED CONTROL PLANE DAEMON          ");
    println!("============================================================");

    let path = match config_path {
        Some(p) => {
            if !p.exists() {
                eprintln!("============================================================");
                eprintln!("Error: Configuration file not found at {:?}", p);
                eprintln!("============================================================");
                eprintln!("Please specify a valid configuration file using -c <path>:");
                eprintln!("  ./hypercache -c /path/to/hypercache-agent.toml start");
                eprintln!("============================================================");
                return Err(anyhow::anyhow!("Configuration file {:?} does not exist", p));
            }
            p
        }
        None => {
            eprintln!("============================================================");
            eprintln!("Error: No configuration file specified!");
            eprintln!("============================================================");
            eprintln!("Please specify the configuration file using -c <path>:");
            eprintln!("  ./hypercache -c /path/to/hypercache-agent.toml start");
            eprintln!("============================================================");
            return Err(anyhow::anyhow!("Missing required configuration file. Please provide -c <path>."));
        }
    };

    let canonical_path = path.canonicalize().unwrap_or_else(|_| path.clone());
    info!("Loaded configuration from: {}", canonical_path.display());

    let socket_path = config::get_socket_path();
    let lock_path = config::get_lock_path();

    if let Some(parent) = lock_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Some(parent) = socket_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let mut config = Config::load_from_file(&path)?;
    config.socket_path = socket_path.to_string_lossy().to_string();

    // Apply CLI overrides for RAM caching if provided
    if enable_ram_flag {
        config.ram_cache_enabled = true;
    }
    if let Some(cap) = custom_ram_capacity {
        config.ram_cache_capacity_bytes = Some(cap);
    }

    // 1. Host-wide exclusive storage pool lock (prevents multiple daemons on same machine from corrupting NVMe pool)
    let lock_file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .with_context(|| format!("Failed to open or create lockfile at {:?}", lock_path))?;

    if lock_file.try_lock_exclusive().is_err() {
        eprintln!("============================================================");
        eprintln!("Error: Another HyperCache daemon is already running on this host!");
        eprintln!("============================================================");
        eprintln!("Only one HyperCache daemon may manage the host's block storage pool.");
        eprintln!("- To check running processes: ps aux | grep hypercache");
        eprintln!("- To stop running daemon:     pkill hypercache");
        eprintln!("============================================================");
        return Err(anyhow::anyhow!(
            "Host storage lock {:?} is held by another active process",
            lock_path
        ));
    }

    // 2. Configured Unix Domain Socket connectivity check
    if socket_path.exists() {
        match UnixStream::connect(&socket_path).await {
            Ok(mut stream) => {
                let probe_req = serde_json::to_string(&protocol::Request::Ping)? + "\n";
                if stream.write_all(probe_req.as_bytes()).await.is_ok() {
                    eprintln!("============================================================");
                    eprintln!("Error: HyperCache daemon is already running on this node!");
                    eprintln!("============================================================");
                    eprintln!("Active daemon socket: {}", socket_path.display());
                    eprintln!("- To check daemon status:  ./hypercache status");
                    eprintln!("- To list cached models:   ./hypercache ls");
                    eprintln!("- To stop running daemon:  pkill hypercache");
                    eprintln!("============================================================");
                    return Err(anyhow::anyhow!(
                        "HyperCache daemon is already running on socket {:?}",
                        socket_path
                    ));
                }
            }
            Err(_) => {
                info!("Found inactive socket file at {:?}. Cleaning up stale socket.", socket_path);
                let _ = std::fs::remove_file(&socket_path);
            }
        }
    }

    if let Some(devs) = explicit_devices {
        if !devs.is_empty() {
            config.device_discovery_mode = "manual".to_string();
            config.block_devices = devs;
        }
    }

    let metrics_enabled = enable_metrics_flag || custom_metrics_port.is_some() || config.metrics_enabled;
    let metrics_port = custom_metrics_port.unwrap_or(config.metrics_port);

    let agent_enabled = enable_agent_flag || custom_agent_port.is_some() || config.agent_enabled || config.controller_url.is_some();
    let agent_port = custom_agent_port.unwrap_or(config.agent_http_port);
    let agent_bind = config.agent_bind_address.clone();

    // Dynamic SSD / NVMe Device Auto-Discovery
    let detected_devices = DeviceDetector::detect_devices(&config);
    println!("------------------------------------------------------------");
    if detected_devices.is_empty() {
        println!("No secondary raw NVMe block devices detected (OS root drive protected).");
        if config.ram_cache_enabled {
            println!("Operating with Tier 1 (Host RAM / POSIX /dev/shm) active.");
        }
    } else {
        println!("Discovered {} Active Block Device(s) in Striping Pool:", detected_devices.len());
        for (i, dev) in detected_devices.iter().enumerate() {
            let size_str = if dev.size_bytes >= 1024 * 1024 * 1024 * 1024 {
                format!("{:.2} TiB", dev.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0 * 1024.0))
            } else {
                format!("{:.2} GiB", dev.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0))
            };
            let kind_str = if dev.is_virtual {
                if dev.is_partition { "Virtual Partition" } else { "Virtual Disk" }
            } else if dev.is_nvme {
                if dev.is_partition { "NVMe Partition" } else { "NVMe Whole Disk" }
            } else {
                "Block Device"
            };
            println!(
                "  [{:02}] {:<22} | Type: {:<18} | Size: {:>9} | Model: {}",
                i,
                dev.path,
                kind_str,
                size_str,
                dev.model.as_deref().unwrap_or("Generic")
            );
        }
    }
    println!("------------------------------------------------------------");

    // Assign discovered devices to config
    config.block_devices = detected_devices.iter().map(|d| d.path.clone()).collect();

    let total_storage_bytes: u64 = detected_devices.iter().map(|d| d.size_bytes).sum();
    let ram_cap = if config.ram_cache_enabled {
        config.ram_cache_capacity_bytes.unwrap_or(32 * 1024 * 1024 * 1024)
    } else {
        0
    };
    let combined_storage_bytes = total_storage_bytes + ram_cap;
    let max_capacity = config
        .max_cache_capacity_bytes
        .unwrap_or(combined_storage_bytes.max(32 * 1024 * 1024 * 1024));

    let telemetry = TelemetryCollector::new(&config.metrics_bind_address, metrics_port, metrics_enabled);
    if combined_storage_bytes > 0 {
        telemetry.state.storage_capacity_bytes.store(combined_storage_bytes, std::sync::atomic::Ordering::Relaxed);
    }
    telemetry.state.active_devices_count.store(detected_devices.len() as u64, std::sync::atomic::Ordering::Relaxed);

    let persistence_path = if config.metadata_persistence_enabled {
        let resolved = config::resolve_metadata_persistence_path(&config.metadata_persistence_path);
        config.metadata_persistence_path = resolved.to_string_lossy().to_string();
        Some(resolved)
    } else {
        None
    };

    let t_boot = Instant::now();
    let (metadata, restored_info) = if let Some(ref ppath) = persistence_path {
        if let Ok(Some((store, count, total_bytes))) = MetadataStore::load_from_disk(ppath, max_capacity, Some(telemetry.state.clone())).await {
            (store, Some((count, total_bytes)))
        } else {
            (MetadataStore::new(max_capacity, Some(telemetry.state.clone()), persistence_path.clone()), None)
        }
    } else {
        (MetadataStore::new(max_capacity, Some(telemetry.state.clone()), None), None)
    };
    let boot_dur = t_boot.elapsed();
    metadata
        .register_device_capacities(&detected_devices)
        .await;

    if let Some((count, total_bytes)) = restored_info {
        let size_str = if total_bytes >= 1024 * 1024 * 1024 {
            format!("{:.2} GiB", total_bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        } else {
            format!("{:.2} MiB", total_bytes as f64 / (1024.0 * 1024.0))
        };
        println!("Warm-Boot Recovery: Restored {} cached model(s) ({}) from NVMe manifest in {:.2} ms.", count, size_str, boot_dur.as_secs_f64() * 1000.0);
    }

    let ram_pool = if config.ram_cache_enabled {
        let shm_path = config::get_ram_pool_path(&config);
        let pool = Arc::new(ram::RamMemoryPool::new(shm_path, ram_cap));
        metadata.set_ram_pool(pool.clone()).await;
        Some(pool)
    } else {
        None
    };

    let ingestion = IngestionEngine::new(config.clone(), metadata.clone(), ram_pool.clone());

    // Start Node Agent HTTP server for fleet coordinator orchestration if enabled
    if agent_enabled {
        let agent_ctx = agent_server::AgentContext {
            config: config.clone(),
            metadata: metadata.clone(),
            ingestion: ingestion.clone(),
            discovered_devices: detected_devices.clone(),
        };
        if let Err(e) = agent_server::start_agent_http_server(agent_ctx, agent_port).await {
            warn!("Failed to start Node Agent HTTP server on port {}: {}", agent_port, e);
        }
    }

    let server = Arc::new(Server::new(
        config.clone(),
        metadata,
        ingestion,
        telemetry,
        detected_devices,
    ));

    println!("HyperCache Control Plane Daemon started successfully.");
    println!("   - Config Loaded:      {}", canonical_path.display());
    println!("   - Unix Domain Socket: {}", config.socket_path);
    if config.ram_cache_enabled {
        if let Some(ref pool) = ram_pool {
            let cap_bytes: u64 = pool.get_capacity_bytes();
            let cap_str = format!("{:.2} GiB", cap_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
            println!("   - Tier 1 (Host RAM):  ACTIVE (Capacity: {}, Path: {})", cap_str, pool.get_shm_path().display());
        }
    } else {
        println!("   - Tier 1 (Host RAM):  DISABLED");
    }
    if metrics_enabled {
        println!("   - Prometheus Exporter: http://{}:{}/metrics", config.metrics_bind_address, metrics_port);
    } else {
        println!("   - Prometheus Exporter: DISABLED (Off by default, enable via --metrics or config)");
    }
    if agent_enabled {
        println!("   - Agent Fleet API:    http://{}:{}", agent_bind, agent_port);
    } else {
        println!("   - Agent Fleet API:    DISABLED (Off by default, enable via --agent or controller_url)");
    }
    if let Some(ref curl) = config.controller_url {
        println!("   - Cluster Controller: {} (Heartbeat & Fleet Registration Active)", curl);
    }
    if config.metadata_persistence_enabled {
        println!("   - NVMe Persistence:   {} (Async Auto-Sync Active)", config.metadata_persistence_path);
    }
    println!("Press Ctrl+C to stop.");

    tokio::select! {
        res = server.clone().run() => {
            if let Err(e) = res {
                tracing::error!("Server terminated with error: {}", e);
            }
        }
        _ = tokio::signal::ctrl_c() => {
            println!("\nReceived shutdown signal. Flushing metadata manifest and stopping cleanly...");
            let _ = server.metadata().flush_to_disk().await;
        }
    }

    let sock_path = Path::new(&config.socket_path);
    if sock_path.exists() {
        let _ = std::fs::remove_file(sock_path);
    }

    println!("HyperCache Control Plane stopped cleanly.");
    Ok(())
}

async fn run_inspect(config_path: Option<PathBuf>, explicit_devices: Option<Vec<String>>) -> Result<()> {
    println!("============================================================");
    println!("         HYPERCACHE STORAGE POOL & HARDWARE INSPECTOR       ");
    println!("============================================================");

    let mut config = if let Some(p) = config_path {
        if !p.exists() {
            eprintln!("============================================================");
            eprintln!("Error: Configuration file not found at {:?}", p);
            eprintln!("============================================================");
            eprintln!("Please specify a valid configuration file using -c <path>:");
            eprintln!("  ./hypercache -c /path/to/hypercache-agent.toml inspect");
            eprintln!("============================================================");
            return Err(anyhow::anyhow!("Configuration file {:?} does not exist", p));
        }
        let canonical_path = p.canonicalize().unwrap_or_else(|_| p.clone());
        info!("Loaded configuration from: {}", canonical_path.display());
        Config::load_from_file(&p)?
    } else {
        info!("Running inspector with default discovery rules.");
        Config::default()
    };

    if let Some(devs) = explicit_devices {
        if !devs.is_empty() {
            config.device_discovery_mode = "manual".to_string();
            config.block_devices = devs;
        }
    }

    let devices = DeviceDetector::detect_devices(&config);
    println!("Active Striped Devices: {}", devices.len());
    let mut total_bytes: u64 = 0;
    for (i, dev) in devices.iter().enumerate() {
        total_bytes += dev.size_bytes;
        let size_str = if dev.size_bytes >= 1024 * 1024 * 1024 * 1024 {
            format!("{:.2} TiB", dev.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0 * 1024.0))
        } else {
            format!("{:.2} GiB", dev.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        };
        let numa_str = dev.numa_node.map(|n| format!("NUMA: {}", n)).unwrap_or_else(|| "NUMA: N/A".to_string());
        println!("  [{:02}] {:<20} | Size: {:>9} | {:<9} | NVMe: {:<5} | Model: {}",
            i, dev.path, size_str, numa_str, dev.is_nvme, dev.model.as_deref().unwrap_or("Generic"));
    }
    println!("------------------------------------------------------------");
    println!("Total Aggregate Pool Capacity: {:.2} TiB ({:.2} GiB)",
        total_bytes as f64 / (1024.0 * 1024.0 * 1024.0 * 1024.0),
        total_bytes as f64 / (1024.0 * 1024.0));
    println!("Chunk Size:                    {:.2} MiB", config.chunk_size_bytes as f64 / (1024.0 * 1024.0));
    println!("Safety Exclusions Active:      Mounted Root Tree + Superblock Filesystem Signatures");
    println!("============================================================");
    Ok(())
}

async fn run_check(
    config_path: Option<PathBuf>,
    explicit_devices: Option<Vec<String>>,
    json: bool,
    enable_ram_flag: bool,
) -> Result<()> {
    let report = check::evaluate_host(config_path, explicit_devices, enable_ram_flag)?;

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        if !report.ready {
            std::process::exit(1);
        }
        return Ok(());
    }

    println!("==========================================================================================");
    println!("                      HYPERCACHE HOST & ENVIRONMENT READINESS CHECK                       ");
    println!("==========================================================================================");
    for item in &report.checks {
        println!(
            "  [{:<4}] {:<12} | {:<32} | {}",
            item.status, item.category, item.name, item.detail
        );
        if let Some(ref rem) = item.remediation {
            println!("         -> Action: {}", rem);
        }
    }
    println!("------------------------------------------------------------------------------------------");
    println!(
        "Summary: {} Passed, {} Warning(s), {} Failed | Host Status: {}",
        report.pass_count,
        report.warn_count,
        report.fail_count,
        if report.ready {
            "READY TO RUN HYPERCACHE"
        } else {
            "NOT READY (Action Required)"
        }
    );
    println!("==========================================================================================");

    if !report.ready {
        std::process::exit(1);
    }
    Ok(())
}

async fn run_status(socket_path: &str) -> Result<()> {
    println!("============================================================");
    println!("              HYPERCACHE LIVE DAEMON STATUS                 ");
    println!("============================================================");

    let mut stream = UnixStream::connect(socket_path)
        .await
        .with_context(|| format!("HyperCache daemon is not running at {}.", socket_path))?;

    let (reader, mut writer) = stream.split();
    let mut reader = BufReader::new(reader);

    // 1. Send PING
    let ping_req = serde_json::to_string(&Request::Ping)? + "\n";
    writer.write_all(ping_req.as_bytes()).await?;
    writer.flush().await?;

    let mut line = String::new();
    reader.read_line(&mut line).await?;
    let resp: Response = serde_json::from_str(line.trim())?;

    if resp.status == ResponseStatus::Pong {
        println!("Daemon State:        ONLINE (UDS: {})", socket_path);
    } else {
        println!("Daemon State:        UNHEALTHY ({:?})", resp);
    }

    // 2. Hardware Concurrency
    let detected_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let target_workers = (detected_cores * 4).clamp(16, 64);
    println!("Hardware Queue:      {} Worker Threads (across {} vCPUs)", target_workers, detected_cores);

    // 3. Query Active Devices
    line.clear();
    let dev_req = serde_json::to_string(&Request::GetDevices)? + "\n";
    writer.write_all(dev_req.as_bytes()).await?;
    writer.flush().await?;

    reader.read_line(&mut line).await?;
    let dev_resp: Response = serde_json::from_str(line.trim())?;
    let mut total_pool_bytes: u64 = 0;
    if let Some(devices) = dev_resp.devices {
        println!("Striped Devices:     {} Active Disk(s)", devices.len());
        for (i, d) in devices.iter().enumerate() {
            total_pool_bytes += d.size_bytes;
            let numa_str = d.numa_node.map(|n| format!("NUMA: {}", n)).unwrap_or_else(|| "NUMA: N/A".to_string());
            println!("  [{:02}] {:<18} | {:>9} | {:<9} | {}", i, d.path, format_bytes(d.size_bytes), numa_str, d.model.as_deref().unwrap_or("Generic"));
        }
        println!("Total Storage Pool:  {}", format_bytes(total_pool_bytes));
    }

    // 4. Query Cache Inventory
    line.clear();
    let list_req = serde_json::to_string(&Request::List)? + "\n";
    writer.write_all(list_req.as_bytes()).await?;
    writer.flush().await?;

    reader.read_line(&mut line).await?;
    let list_resp: Response = serde_json::from_str(line.trim())?;
    let cached_objs = list_resp.objects.unwrap_or_default();
    let used_bytes = list_resp.total_bytes_used.unwrap_or(0);
    let max_cap = if total_pool_bytes > 0 { total_pool_bytes } else { list_resp.max_capacity_bytes.unwrap_or(1) };
    let pct = if max_cap > 0 { (used_bytes as f64 / max_cap as f64) * 100.0 } else { 0.0 };

    println!("Cached Models:       {} object(s) cached", cached_objs.len());
    println!("Cache Footprint:     {} / {} ({:.2}% utilized)", format_bytes(used_bytes), format_bytes(max_cap), pct);
    println!("Metrics Endpoint:    http://127.0.0.1:9110/metrics");
    println!("============================================================");
    Ok(())
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 * 1024 {
        format!("{:.2} TiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

async fn run_ls(socket_path: &str, json: bool) -> Result<()> {
    let mut stream = UnixStream::connect(socket_path)
        .await
        .with_context(|| format!("HyperCache daemon is not running at {}.", socket_path))?;

    let (reader, mut writer) = stream.split();
    let mut reader = BufReader::new(reader);

    let req = serde_json::to_string(&Request::List)? + "\n";
    writer.write_all(req.as_bytes()).await?;
    writer.flush().await?;

    let mut line = String::new();
    reader.read_line(&mut line).await?;
    let resp: Response = serde_json::from_str(line.trim())?;

    if json {
        println!("{}", serde_json::to_string_pretty(&resp)?);
        return Ok(());
    }

    println!("====================================================================================================");
    println!("                                HYPERCACHE CACHED OBJECTS & INVENTORY                               ");
    println!("====================================================================================================");

    let objects = resp.objects.unwrap_or_default();
    let total_used = resp.total_bytes_used.unwrap_or(0);
    let max_cap = resp.max_capacity_bytes.unwrap_or(1);

    if objects.is_empty() {
        println!("  (No model checkpoints currently cached in memory or NVMe)");
    } else {
        println!("{:<50} | {:>10} | {:>6} | {:<6} | {:<19} | {}",
            "URI", "SIZE", "CHUNKS", "TIER", "INGESTED AT", "STRIPED TARGETS");
        println!("{}", "-".repeat(110));

        for obj in &objects {
            let date_str = chrono::DateTime::from_timestamp(obj.created_at, 0)
                .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
                .unwrap_or_else(|| "N/A".to_string());

            let tier_str = match obj.tier {
                protocol::StorageTier::Ram => "RAM",
                protocol::StorageTier::Nvme => "NVMe",
            };

            let drives_str = obj.devices.join(", ");

            println!("{:<50} | {:>10} | {:>6} | {:<6} | {:<19} | {}",
                if obj.uri.len() > 50 { format!("...{}", &obj.uri[obj.uri.len() - 47..]) } else { obj.uri.clone() },
                format_bytes(obj.object_size),
                obj.chunk_count,
                tier_str,
                date_str,
                drives_str
            );
        }
    }

    println!("----------------------------------------------------------------------------------------------------");
    let pct = if max_cap > 0 { (total_used as f64 / max_cap as f64) * 100.0 } else { 0.0 };
    println!("Total Cached Objects: {} | Active Cache Footprint: {} / {} ({:.2}% Utilized)",
        objects.len(), format_bytes(total_used), format_bytes(max_cap), pct);
    println!("====================================================================================================");
    Ok(())
}

async fn run_rm(socket_path: &str, uri: &str) -> Result<()> {
    let mut stream = UnixStream::connect(socket_path)
        .await
        .with_context(|| format!("HyperCache daemon is not running at {}.", socket_path))?;

    let (reader, mut writer) = stream.split();
    let mut reader = BufReader::new(reader);

    let req = serde_json::to_string(&Request::Evict { uri: uri.to_string() })? + "\n";
    writer.write_all(req.as_bytes()).await?;
    writer.flush().await?;

    let mut line = String::new();
    reader.read_line(&mut line).await?;
    let resp: Response = serde_json::from_str(line.trim())?;

    if resp.status == ResponseStatus::Evicted {
        let freed = resp.freed_bytes.unwrap_or(0);
        println!("Successfully evicted '{}' ({} freed from cache index).", uri, format_bytes(freed));
    } else {
        let err = resp.error_message.unwrap_or_else(|| "Unknown error".to_string());
        println!("Failed to evict '{}': {}", uri, err);
    }
    Ok(())
}

async fn run_purge(socket_path: &str) -> Result<()> {
    let mut stream = UnixStream::connect(socket_path)
        .await
        .with_context(|| format!("HyperCache daemon is not running at {}.", socket_path))?;

    let (reader, mut writer) = stream.split();
    let mut reader = BufReader::new(reader);

    let req = serde_json::to_string(&Request::Purge)? + "\n";
    writer.write_all(req.as_bytes()).await?;
    writer.flush().await?;

    let mut line = String::new();
    reader.read_line(&mut line).await?;
    let resp: Response = serde_json::from_str(line.trim())?;

    if resp.status == ResponseStatus::Purged {
        let count = resp.purged_count.unwrap_or(0);
        let freed = resp.freed_bytes.unwrap_or(0);
        println!("Successfully purged {} cached model(s) ({} freed from storage pool).", count, format_bytes(freed));
        println!("  -> Drive allocation pointers reset to initial safety offsets.");
    } else {
        let err = resp.error_message.unwrap_or_else(|| "Unknown error".to_string());
        println!("Failed to purge cache: {}", err);
    }
    Ok(())
}

async fn run_refresh(socket_path: &str, uri: &str) -> Result<()> {
    println!("Refreshing model checkpoint '{}'...", uri);

    // 1. Evict old copy
    let mut stream = UnixStream::connect(socket_path)
        .await
        .with_context(|| format!("HyperCache daemon is not running at {}.", socket_path))?;

    let (reader, mut writer) = stream.split();
    let mut reader = BufReader::new(reader);

    let evict_req = serde_json::to_string(&Request::Evict { uri: uri.to_string() })? + "\n";
    writer.write_all(evict_req.as_bytes()).await?;
    writer.flush().await?;

    let mut line = String::new();
    reader.read_line(&mut line).await?;
    println!("  -> Evicted old local cache metadata copy.");

    // 2. Trigger fresh re-ingestion
    line.clear();
    let t_start = Instant::now();
    let ingest_req = serde_json::to_string(&Request::Locate {
        uri: uri.to_string(),
        auto_ingest: true,
        tier: None,
        pinned: false,
    })? + "\n";
    writer.write_all(ingest_req.as_bytes()).await?;
    writer.flush().await?;

    reader.read_line(&mut line).await?;
    let resp: Response = serde_json::from_str(line.trim())?;
    let dur = t_start.elapsed();

    if resp.status == ResponseStatus::Hit {
        let size = resp.object_size.unwrap_or(0);
        let chunks = resp.chunk_layout.map(|c| c.len()).unwrap_or(0);
        let dur_s = dur.as_secs_f64();
        let bw = if dur_s > 0.0 { (size as f64 / (1024.0 * 1024.0)) / dur_s } else { 0.0 };

        println!("  -> Freshly ingested {} ({} chunks) from S3 across striped NVMe in {:.3} s ({:.2} MB/s).",
            format_bytes(size), chunks, dur_s, bw);
        println!("Model '{}' successfully refreshed and hot-cached in memory.", uri);
    } else {
        let err = resp.error_message.unwrap_or_else(|| "Unknown error".to_string());
        println!("Failed to re-ingest '{}': {}", uri, err);
    }

    Ok(())
}

async fn run_preload(socket_path: &str, uri: Option<String>, file_path: Option<PathBuf>, target_tier: Option<protocol::StorageTier>, pinned: bool) -> Result<()> {
    let mut targets = Vec::new();

    if let Some(u) = uri {
        targets.push(u);
    }

    if let Some(ref fp) = file_path {
        let content = std::fs::read_to_string(fp)
            .with_context(|| format!("Failed to read preload manifest file at {:?}", fp))?;
        for line in content.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() && !trimmed.starts_with('#') {
                targets.push(trimmed.to_string());
            }
        }
    }

    if targets.is_empty() {
        anyhow::bail!("Please specify a model URI to pre-load (e.g. ./hypercache preload <URI>) or provide a file list via --file <path>.");
    }

    let tier_label = match target_tier {
        Some(protocol::StorageTier::Ram) => "Tier 1: Host RAM Shared Memory",
        Some(protocol::StorageTier::Nvme) => "Tier 2: Striped NVMe Pool",
        None => "Auto-Selected Tier (RAM if capacity permits, else NVMe)",
    };

    println!("====================================================================================================");
    println!("                               HYPERCACHE PROACTIVE CACHE PRE-LOADER                                ");
    println!("====================================================================================================");
    println!("Target Items to Pre-Warm: {}", targets.len());
    println!("Target Storage Tier:      {}", tier_label);
    println!("Eviction Protection:      {}", if pinned { "PINNED (Protected from W-TinyLFU Eviction)" } else { "Standard W-TinyLFU" });

    let mut total_bytes_ingested: u64 = 0;
    let t_all_start = Instant::now();

    for (idx, target_uri) in targets.iter().enumerate() {
        if targets.len() > 1 {
            println!("\n[{}/{}] Pre-loading '{}'...", idx + 1, targets.len(), target_uri);
        } else {
            println!("Pre-loading checkpoint '{}'...", target_uri);
        }

        let mut stream = UnixStream::connect(socket_path)
            .await
            .with_context(|| format!("HyperCache daemon is not running at {}.", socket_path))?;

        let (reader, mut writer) = stream.split();
        let mut reader = BufReader::new(reader);

        let t_start = Instant::now();
        let ingest_req = serde_json::to_string(&Request::Locate {
            uri: target_uri.clone(),
            auto_ingest: true,
            tier: target_tier,
            pinned,
        })? + "\n";
        writer.write_all(ingest_req.as_bytes()).await?;
        writer.flush().await?;

        let mut line = String::new();
        reader.read_line(&mut line).await?;
        let resp: Response = serde_json::from_str(line.trim())?;
        let dur = t_start.elapsed();

        if resp.status == ResponseStatus::Hit {
            let size = resp.object_size.unwrap_or(0);
            let chunks = resp.chunk_layout.as_ref().map(|c| c.len()).unwrap_or(0);
            let first_tier = resp.chunk_layout.as_ref().and_then(|c| c.first()).map(|c| c.tier).unwrap_or(protocol::StorageTier::Nvme);
            let tier_name = match first_tier {
                protocol::StorageTier::Ram => "RAM Shared Memory",
                protocol::StorageTier::Nvme => "NVMe Pool",
            };
            let dur_s = dur.as_secs_f64();
            let bw = if dur_s > 0.0 { (size as f64 / (1024.0 * 1024.0)) / dur_s } else { 0.0 };
            total_bytes_ingested += size;

            println!("  - Ingested and striped {} ({} chunks) across {} in {:.3} s ({:.2} MB/s).",
                format_bytes(size), chunks, tier_name, dur_s, bw);
            println!("  - Metadata registered in Moka RAM and synced to NVMe persistence manifest.");
            println!("Checkpoint '{}' is pre-warmed and ready for line-rate DMA loading.", target_uri);
        } else {
            let err = resp.error_message.unwrap_or_else(|| "Unknown error".to_string());
            println!("Failed to pre-load '{}': {}", target_uri, err);
        }
    }

    let all_dur = t_all_start.elapsed();
    println!("----------------------------------------------------------------------------------------------------");
    println!("Pre-Loading Complete: Ingested {} across {} object(s) in {:.2} s.",
        format_bytes(total_bytes_ingested), targets.len(), all_dur.as_secs_f64());
    println!("====================================================================================================");

    Ok(())
}

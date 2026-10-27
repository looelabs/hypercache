# ==============================================================================
# Stage 1: Rust Release Builder (hypercache & hypercache-admin)
# ==============================================================================
FROM rust:1.82-bookworm AS rust-builder

WORKDIR /src/control-plane

# Cache dependency layer first
COPY control-plane/Cargo.toml control-plane/Cargo.lock* ./
RUN mkdir -p src/bin && \
    echo "fn main() {}" > src/main.rs && \
    echo "fn main() {}" > src/bin/admin.rs && \
    echo "" > src/lib.rs && \
    cargo build --release && \
    rm -rf src

# Copy actual control-plane source and compile release binaries
COPY control-plane/src ./src
RUN touch src/lib.rs src/main.rs src/bin/admin.rs && \
    cargo build --release --bins && \
    strip target/release/hypercache target/release/hypercache-admin

# ==============================================================================
# Stage 2: Production Runtime Image (Worker Node Daemon, Controller & CLI)
# ==============================================================================
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    python3 \
    python3-pip \
    util-linux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /etc/hypercache

# Copy compiled Rust binaries
COPY --from=rust-builder /src/control-plane/target/release/hypercache /usr/local/bin/hypercache
COPY --from=rust-builder /src/control-plane/target/release/hypercache-admin /usr/local/bin/hypercache-admin

# Copy default configuration templates
COPY configs/hypercache-agent.toml /etc/hypercache/hypercache-agent.toml
COPY configs/hypercache-controller.toml /etc/hypercache/hypercache-controller.toml

# Install Python Data Plane & fsspec integration into container Python environment
COPY client-py /opt/hypercache/client-py
RUN pip3 install --no-cache-dir --break-system-packages -e /opt/hypercache/client-py

# Ensure runtime socket and metadata directories exist
RUN mkdir -p /run/hypercache /var/lib/hypercache /var/cache/hypercache

# Expose optional ports:
# - 9110: Prometheus metrics exporter (--metrics)
# - 9115: Worker Node Agent Fleet & P2P API (--agent)
# - 9120: Central Cluster Controller (hypercache-admin controller)
EXPOSE 9110 9115 9120

ENTRYPOINT ["/usr/local/bin/hypercache"]
CMD ["-c", "/etc/hypercache/hypercache-agent.toml", "start"]

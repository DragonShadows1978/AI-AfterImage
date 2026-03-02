#!/bin/bash
# Install AfterImage Embedding Daemon as a systemd service

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="afterimage-embedder"
SERVICE_FILE="${SCRIPT_DIR}/${SERVICE_NAME}.service"
SYSTEMD_DIR="/etc/systemd/system"

echo "AfterImage Embedding Daemon - Service Installer"
echo "================================================"

# Check for root/sudo
if [[ $EUID -ne 0 ]]; then
    echo "This script requires root privileges. Please run with sudo."
    exit 1
fi

# Check if service file exists
if [[ ! -f "$SERVICE_FILE" ]]; then
    echo "Error: Service file not found: $SERVICE_FILE"
    exit 1
fi

# Create environment file if it doesn't exist
ENV_FILE="/home/vader/.afterimage/embedder.env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Creating environment file: $ENV_FILE"
    mkdir -p "$(dirname "$ENV_FILE")"
    cat > "$ENV_FILE" << 'EOF'
# AfterImage Embedding Daemon Environment
# Uncomment and modify as needed

# PostgreSQL password (required for PostgreSQL backend)
AFTERIMAGE_PG_PASSWORD=afterimage

# Device selection: auto, cuda, cpu
# EMBEDDER_DEVICE=auto

# Batch size for embedding generation
# EMBEDDER_BATCH_SIZE=32

# Maximum entries per reindex cycle
# EMBEDDER_MAX_PER_CYCLE=500

# Reindex interval in seconds
# EMBEDDER_INTERVAL_SECONDS=300
EOF
    chown vader:vader "$ENV_FILE"
    chmod 600 "$ENV_FILE"
fi

# Copy service file
echo "Installing service file to $SYSTEMD_DIR"
cp "$SERVICE_FILE" "$SYSTEMD_DIR/"
chmod 644 "${SYSTEMD_DIR}/${SERVICE_NAME}.service"

# Reload systemd
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable service
echo "Enabling ${SERVICE_NAME} service..."
systemctl enable "${SERVICE_NAME}"

echo ""
echo "Installation complete!"
echo ""
echo "Commands:"
echo "  sudo systemctl start ${SERVICE_NAME}     # Start the service"
echo "  sudo systemctl stop ${SERVICE_NAME}      # Stop the service"
echo "  sudo systemctl status ${SERVICE_NAME}    # Check status"
echo "  sudo journalctl -u ${SERVICE_NAME} -f    # View logs"
echo ""
echo "Configuration:"
echo "  Edit $ENV_FILE to customize settings"
echo "  Then run: sudo systemctl restart ${SERVICE_NAME}"

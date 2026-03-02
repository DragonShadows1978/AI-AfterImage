#!/bin/bash
# Install AfterImage Embedding Daemon as a macOS launchd service

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="com.afterimage.embedder"
PLIST_TEMPLATE="${SCRIPT_DIR}/${SERVICE_NAME}.plist"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "AfterImage Embedding Daemon - macOS Service Installer"
echo "======================================================"

# Determine installation type
INSTALL_TYPE="${1:-user}"
if [[ "$INSTALL_TYPE" != "user" && "$INSTALL_TYPE" != "system" ]]; then
    echo "Usage: $0 [user|system]"
    echo "  user   - Install for current user (LaunchAgents, default)"
    echo "  system - Install system-wide (LaunchDaemons, requires sudo)"
    exit 1
fi

# Set directories based on install type
if [[ "$INSTALL_TYPE" == "system" ]]; then
    if [[ $EUID -ne 0 ]]; then
        echo "Error: System-wide installation requires root privileges."
        echo "Please run: sudo $0 system"
        exit 1
    fi
    LAUNCH_DIR="/Library/LaunchDaemons"
    RUN_USER="${SUDO_USER:-root}"
    HOME_DIR=$(eval echo "~$RUN_USER")
else
    LAUNCH_DIR="$HOME/Library/LaunchAgents"
    RUN_USER="$USER"
    HOME_DIR="$HOME"
fi

PLIST_FILE="${LAUNCH_DIR}/${SERVICE_NAME}.plist"

echo "Installation type: $INSTALL_TYPE"
echo "Launch directory: $LAUNCH_DIR"
echo "Running as user: $RUN_USER"
echo "Workspace: $WORKSPACE_DIR"
echo ""

# Check if plist template exists
if [[ ! -f "$PLIST_TEMPLATE" ]]; then
    echo "Error: Plist template not found: $PLIST_TEMPLATE"
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3."
    exit 1
fi

# Unload existing service if running
if launchctl list 2>/dev/null | grep -q "$SERVICE_NAME"; then
    echo "Stopping existing service..."
    if [[ "$INSTALL_TYPE" == "system" ]]; then
        launchctl bootout system/${SERVICE_NAME} 2>/dev/null || true
    else
        launchctl unload "$PLIST_FILE" 2>/dev/null || true
    fi
    sleep 1
fi

# Create required directories
echo "Creating directories..."
mkdir -p "${HOME_DIR}/.afterimage/logs"
mkdir -p "$LAUNCH_DIR"

# Create environment file if it doesn't exist
ENV_FILE="${HOME_DIR}/.afterimage/embedder.env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Creating environment file: $ENV_FILE"
    cat > "$ENV_FILE" << 'ENVEOF'
# AfterImage Embedding Daemon Environment (macOS)
# Source this file or set these variables before starting the daemon

# PostgreSQL password (required for PostgreSQL backend)
export AFTERIMAGE_PG_PASSWORD=afterimage

# Device selection: auto, cuda, cpu (macOS typically uses cpu or mps)
# export EMBEDDER_DEVICE=auto

# Batch size for embedding generation
# export EMBEDDER_BATCH_SIZE=32

# Maximum entries per reindex cycle
# export EMBEDDER_MAX_PER_CYCLE=500

# Reindex interval in seconds
# export EMBEDDER_INTERVAL_SECONDS=300

# Health check server
# export EMBEDDER_HEALTH_PORT=9090
ENVEOF
fi

# Generate plist from template
echo "Generating plist file..."
sed -e "s|__WORKSPACE_DIR__|${WORKSPACE_DIR}|g" \
    -e "s|__HOME_DIR__|${HOME_DIR}|g" \
    "$PLIST_TEMPLATE" > "$PLIST_FILE"

# Set permissions
if [[ "$INSTALL_TYPE" == "system" ]]; then
    chown root:wheel "$PLIST_FILE"
    chmod 644 "$PLIST_FILE"
    chown -R "$RUN_USER" "${HOME_DIR}/.afterimage"
else
    chmod 644 "$PLIST_FILE"
fi

# Validate plist
echo "Validating plist..."
if ! plutil -lint "$PLIST_FILE" > /dev/null 2>&1; then
    echo "Error: Invalid plist file generated"
    plutil -lint "$PLIST_FILE"
    exit 1
fi

# Load and start service
echo "Loading service..."
if [[ "$INSTALL_TYPE" == "system" ]]; then
    launchctl bootstrap system "$PLIST_FILE"
    launchctl enable "system/${SERVICE_NAME}"
else
    launchctl load "$PLIST_FILE"
fi

echo ""
echo "Installation complete!"
echo ""
echo "Commands:"
if [[ "$INSTALL_TYPE" == "system" ]]; then
    echo "  sudo launchctl kickstart system/${SERVICE_NAME}     # Start"
    echo "  sudo launchctl kill TERM system/${SERVICE_NAME}     # Stop"
    echo "  sudo launchctl print system/${SERVICE_NAME}         # Status"
else
    echo "  launchctl start ${SERVICE_NAME}                     # Start"
    echo "  launchctl stop ${SERVICE_NAME}                      # Stop"
    echo "  launchctl list | grep afterimage                    # Status"
fi
echo ""
echo "Logs:"
echo "  tail -f ${HOME_DIR}/.afterimage/logs/embedder.stdout.log"
echo "  tail -f ${HOME_DIR}/.afterimage/logs/embedder.stderr.log"
echo ""
echo "Configuration:"
echo "  Edit ${ENV_FILE}"
echo "  Then restart the service"

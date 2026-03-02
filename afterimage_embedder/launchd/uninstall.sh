#!/bin/bash
# Uninstall AfterImage Embedding Daemon macOS launchd service

set -e

SERVICE_NAME="com.afterimage.embedder"

echo "AfterImage Embedding Daemon - macOS Service Uninstaller"
echo "========================================================"

# Determine installation type
INSTALL_TYPE="${1:-user}"
if [[ "$INSTALL_TYPE" != "user" && "$INSTALL_TYPE" != "system" ]]; then
    echo "Usage: $0 [user|system]"
    echo "  user   - Uninstall user service (LaunchAgents, default)"
    echo "  system - Uninstall system service (LaunchDaemons, requires sudo)"
    exit 1
fi

# Set directories based on install type
if [[ "$INSTALL_TYPE" == "system" ]]; then
    if [[ $EUID -ne 0 ]]; then
        echo "Error: System-wide uninstallation requires root privileges."
        echo "Please run: sudo $0 system"
        exit 1
    fi
    LAUNCH_DIR="/Library/LaunchDaemons"
else
    LAUNCH_DIR="$HOME/Library/LaunchAgents"
fi

PLIST_FILE="${LAUNCH_DIR}/${SERVICE_NAME}.plist"

echo "Installation type: $INSTALL_TYPE"
echo "Plist file: $PLIST_FILE"
echo ""

# Check if service exists
if [[ ! -f "$PLIST_FILE" ]]; then
    echo "Service not installed at $PLIST_FILE"
    exit 0
fi

# Stop and unload service
echo "Stopping service..."
if [[ "$INSTALL_TYPE" == "system" ]]; then
    launchctl bootout system/${SERVICE_NAME} 2>/dev/null || true
else
    launchctl unload "$PLIST_FILE" 2>/dev/null || true
fi

# Remove plist file
echo "Removing plist file..."
rm -f "$PLIST_FILE"

echo ""
echo "Service uninstalled successfully."
echo ""
echo "Note: Configuration and logs are preserved at:"
echo "  ~/.afterimage/embedder.env"
echo "  ~/.afterimage/logs/"
echo ""
echo "To remove all data, run:"
echo "  rm -rf ~/.afterimage"
echo ""

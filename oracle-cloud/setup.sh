#!/bin/bash
# ==========================================================================
#   AgroModules - Oracle Cloud Free Tier Setup Script
#   Instance: Ubuntu 22.04+ (ARM or x86)
#   Purpose: Run Cloudflare Tunnel + API Gateway persistently
# ==========================================================================

set -e

echo "=========================================="
echo "  AgroModules Oracle Cloud Setup"
echo "=========================================="

# --- 1. System updates ---
echo "[1/6] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# --- 2. Install Python 3.11+ ---
echo "[2/6] Installing Python..."
sudo apt install -y python3 python3-pip python3-venv git curl wget

# --- 3. Install PostgreSQL ---
echo "[3/6] Installing PostgreSQL..."
sudo apt install -y postgresql postgresql-contrib
sudo systemctl enable postgresql
sudo systemctl start postgresql

# Setup database
sudo -u postgres psql -c "CREATE USER agro WITH PASSWORD 'agro';" 2>/dev/null || true
sudo -u postgres psql -c "CREATE DATABASE agrosensor OWNER agro;" 2>/dev/null || true
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE agrosensor TO agro;" 2>/dev/null || true

# --- 4. Install cloudflared ---
echo "[4/6] Installing cloudflared..."
ARCH=$(dpkg --print-architecture)
if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
    sudo dpkg -i cloudflared-linux-arm64.deb
    rm cloudflared-linux-arm64.deb
else
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
    sudo dpkg -i cloudflared-linux-amd64.deb
    rm cloudflared-linux-amd64.deb
fi
cloudflared --version

# --- 5. Clone/copy project ---
echo "[5/6] Setting up project directory..."
PROJECT_DIR="/opt/agromodules"
sudo mkdir -p $PROJECT_DIR
sudo chown $USER:$USER $PROJECT_DIR

echo ""
echo "  MANUAL STEP: Copy your project files to $PROJECT_DIR"
echo "  Options:"
echo "    a) git clone <your-repo-url> $PROJECT_DIR"
echo "    b) scp -r ./AgroModules/* user@oracle-ip:$PROJECT_DIR/"
echo ""

# --- 6. Setup Python venvs ---
echo "[6/6] Creating Python virtual environments..."
for SERVICE in ApiGateway AgroSensor Crop_Recommendation_Engine Plant_Disease_Detection Auth; do
    if [ -d "$PROJECT_DIR/$SERVICE" ] && [ -f "$PROJECT_DIR/$SERVICE/requirements.txt" ]; then
        echo "  Setting up venv for $SERVICE..."
        cd "$PROJECT_DIR/$SERVICE"
        python3 -m venv venv
        venv/bin/pip install --upgrade pip
        venv/bin/pip install -r requirements.txt
        echo "  [OK] $SERVICE venv ready"
    fi
done

# --- 7. Setup cloudflared credentials ---
echo ""
echo "=========================================="
echo "  CLOUDFLARED SETUP"
echo "=========================================="
echo ""
echo "  Copy your tunnel credentials to this instance:"
echo ""
echo "  1. Copy credentials file:"
echo "     scp ~/.cloudflared/63c51688-52a1-4d20-acb9-1af1b20f823e.json user@oracle-ip:~/.cloudflared/"
echo ""
echo "  2. Copy cert.pem:"
echo "     scp ~/.cloudflared/cert.pem user@oracle-ip:~/.cloudflared/"
echo ""
echo "  3. Update paths in cloudflare-tunnel-config.yaml:"
echo "     credentials-file: /home/$USER/.cloudflared/63c51688-52a1-4d20-acb9-1af1b20f823e.json"
echo "     origincert: /home/$USER/.cloudflared/cert.pem"
echo ""

# --- 8. Install systemd services ---
echo "Installing systemd services..."
sudo cp $PROJECT_DIR/oracle-cloud/tunnel-keeper.service /etc/systemd/system/
sudo cp $PROJECT_DIR/oracle-cloud/gateway-keeper.service /etc/systemd/system/
sudo systemctl daemon-reload

echo ""
echo "=========================================="
echo "  SETUP COMPLETE"
echo "=========================================="
echo ""
echo "  To start services:"
echo "    sudo systemctl enable --now tunnel-keeper"
echo "    sudo systemctl enable --now gateway-keeper"
echo ""
echo "  To check status:"
echo "    sudo systemctl status tunnel-keeper"
echo "    sudo systemctl status gateway-keeper"
echo ""
echo "  To view logs:"
echo "    journalctl -u tunnel-keeper -f"
echo "    journalctl -u gateway-keeper -f"
echo ""

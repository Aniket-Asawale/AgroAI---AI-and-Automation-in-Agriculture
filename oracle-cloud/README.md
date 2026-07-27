# AgroModules — Oracle Cloud Deployment

## Overview

Deploy all AgroModules services on an **Oracle Cloud Always-Free** instance so the tunnel stays alive 24/7 without your local machine.

## Prerequisites

- Oracle Cloud Free Tier account
- ARM instance (Ampere A1 — 4 OCPUs, 24GB RAM free forever) or AMD instance
- Ubuntu 22.04+ image
- SSH access configured

## Quick Start

### 1. SSH into your instance
```bash
ssh -i ~/.ssh/oracle_key ubuntu@<your-instance-ip>
```

### 2. Copy project files
```bash
# From your local machine:
scp -r ./AgroModules/* ubuntu@<oracle-ip>:/opt/agromodules/
```

### 3. Copy cloudflared credentials
```bash
# From your Windows machine:
scp ~/.cloudflared/63c51688-52a1-4d20-acb9-1af1b20f823e.json ubuntu@<oracle-ip>:~/.cloudflared/
scp ~/.cloudflared/cert.pem ubuntu@<oracle-ip>:~/.cloudflared/
```

### 4. Run setup script
```bash
cd /opt/agromodules
chmod +x oracle-cloud/setup.sh
./oracle-cloud/setup.sh
```

### 5. Use the Linux tunnel config
```bash
cp oracle-cloud/cloudflare-tunnel-config.yaml /opt/agromodules/cloudflare-tunnel-config.yaml
```

### 6. Enable all services
```bash
# Install service files
sudo cp oracle-cloud/*.service /etc/systemd/system/
sudo systemctl daemon-reload

# Enable and start ALL services
sudo systemctl enable --now tunnel-keeper
sudo systemctl enable --now gateway-keeper
sudo systemctl enable --now sensor-api
sudo systemctl enable --now crop-api
sudo systemctl enable --now disease-api
sudo systemctl enable --now auth-api
```

## Service Management

### Check status
```bash
# All services at once
for svc in tunnel-keeper gateway-keeper sensor-api crop-api disease-api auth-api; do
    echo "--- $svc ---"
    systemctl is-active $svc
done
```

### View logs
```bash
journalctl -u tunnel-keeper -f          # Tunnel logs
journalctl -u gateway-keeper -f         # Gateway logs
journalctl -u sensor-api -f             # Sensor API logs
```

### Restart a service
```bash
sudo systemctl restart gateway-keeper
```

### Stop everything
```bash
for svc in tunnel-keeper gateway-keeper sensor-api crop-api disease-api auth-api; do
    sudo systemctl stop $svc
done
```

## Architecture on Oracle Cloud

```
Internet → Cloudflare → Tunnel (cloudflared) → localhost:8080 (Gateway)
                                                    ├── :8000 (Sensor API)
                                                    ├── :8001 (Crop API)
                                                    ├── :8002 (Auth API)
                                                    ├── :8003 (Disease API)
                                                    ├── :8502 (Sensor Dashboard)
                                                    └── :7860 (Disease Dashboard)

Streamlit Cloud: croprecommendationengine.streamlit.app (separate, calls agroaiapp.me/api/crop)
```

## Switching from Local to Oracle Cloud

1. Stop the local tunnel and gateway keeper (`keep_tunnel_gateway_alive.bat`)
2. Start services on Oracle Cloud (`sudo systemctl start tunnel-keeper`)
3. The same tunnel token works — Cloudflare routes traffic to wherever cloudflared is running
4. **No DNS changes needed** — the tunnel ID stays the same

## Switching Back to Local

1. Stop services on Oracle Cloud (`sudo systemctl stop tunnel-keeper`)
2. Start your local `keep_tunnel_gateway_alive.bat`
3. Everything routes back to your local machine

## Oracle Cloud Firewall Note

> **IMPORTANT**: You do NOT need to open any inbound ports on the Oracle Cloud instance.
> Cloudflared creates an **outbound** connection to Cloudflare's edge servers.
> All traffic flows through this outbound tunnel — no inbound firewall rules needed.

## Sensor Dashboard (Static Files)

The sensor dashboard is a static site served by Python's HTTP server. To run it on Oracle Cloud:

```bash
# Manual start (or add another systemd service)
cd /opt/agromodules/AgroSensor
venv/bin/python -m http.server 8502 --directory dashboard &
```

Or create a systemd service for it (not included by default since it's just a static file server).

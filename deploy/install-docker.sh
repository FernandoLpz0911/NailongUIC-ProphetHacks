#!/bin/bash
set -eu
sudo apt-get update -q
sudo apt-get install -y -q ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
. /etc/os-release
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian ${VERSION_CODENAME} stable" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update -q
sudo apt-get install -y -q docker-ce docker-ce-cli containerd.io
sudo mkdir -p /opt/nailong/data
sudo chmod 777 /opt/nailong
sudo usermod -aG docker $USER
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet || true
echo "DOCKER_DONE"

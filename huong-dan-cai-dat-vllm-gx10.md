# Hướng dẫn cài đặt vLLM & 3 Model AI trên cụm 2 máy ASUS Ascent GX10

> **Sản phẩm:** ASUS Ascent GX10 · GPU NVIDIA Blackwell GB10 · ARM v9.2-A · 128GB LPDDR5x Unified Memory/node
> **Hệ điều hành:** NVIDIA DGX™ Base OS (Ubuntu Linux) — cài sẵn, OS duy nhất được hỗ trợ chính thức
> **Mạng nội bộ:** Node 1 `192.168.100.10/24` ↔ Node 2 `192.168.100.11/24` qua NVIDIA ConnectX-7 200G
> **Models:** MedGemma 27B-IT · MedGemma 1.5 4B · Llama 4 Scout (chạy song song)
> **Tài liệu tham khảo:** [ASUS Ascent GX10](https://www.asus.com/vn/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/) · [ASUS FAQ GX10](https://www.asus.com/support/faq/1056142/) · [NVIDIA DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)

---

## Thông số kỹ thuật ASUS Ascent GX10

> **Nguồn:** [ASUS Ascent GX10 Tech Specs](https://www.asus.com/vn/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/techspec/)

| Thành phần | Thông số |
|---|---|
| **Tên sản phẩm** | ASUS Ascent GX10 |
| **CPU** | ARM v9.2-A (GB10 Superchip) |
| **GPU** | NVIDIA Blackwell GPU (GB10, integrated) |
| **Memory** | 128 GB LPDDR5x — unified system memory (CPU+GPU dùng chung) |
| **Storage** | 1TB M.2 NVMe PCIe 4.0 SSD |
| **Mạng** | 1x NVIDIA ConnectX-7 SmartNIC (200G QSFP) · 1x 10G LAN · Wi-Fi 7 · Bluetooth 5.4 |
| **Cổng I/O** | 3x USB 3.2 Gen 2x2 Type-C · 1x USB-C PD 180W · 1x HDMI 2.1 · 1x Kensington Lock |
| **Nguồn** | 240W Adapter |
| **Kích thước** | 150 x 150 x 51 mm · 1.48 kg |
| **OS** | Ubuntu Linux (NVIDIA DGX™ Base OS) — OS duy nhất được kiểm tra và hỗ trợ chính thức |
| **Hiệu năng AI** | 1 petaFLOP (FP4) |
| **Driver** | Dòng 580/590-open — dành riêng cho Blackwell GB10 |
| **CUDA** | Phiên bản 13.x — GB10 (sm_121) yêu cầu CUDA 13 trở lên |
| **nvidia-smi** | Báo "Memory-Usage: Not Supported" — hành vi bình thường của unified memory |
| **vLLM** | Yêu cầu wheel CUDA 13 / aarch64 — không dùng pip install thông thường |
| **Swap** | Phải tắt trước khi vận hành — OOM trên unified memory có thể làm treo toàn bộ hệ thống |
| **Cập nhật hệ thống** | Thực hiện qua DGX Dashboard (`http://localhost:11000`) — không dùng `apt upgrade` |
| **Cluster** | Hỗ trợ kết nối tối đa 2 node qua QSFP 200G trực tiếp, hoặc nhiều hơn qua 200G switch |

---

## Mục lục

0. [Cấu hình mạng 200G — RoCE + MTU 9000](#0-cấu-hình-mạng-200g--roce--mtu-9000)
1. [Phân tích tài nguyên](#1-phân-tích-tài-nguyên)
2. [Kiến trúc triển khai](#2-kiến-trúc-triển-khai)
3. [Bước 1 – Kiểm tra hệ thống & cấu hình nền](#3-bước-1--kiểm-tra-hệ-thống--cấu-hình-nền)
4. [Bước 2 – Cập nhật Driver & CUDA](#4-bước-2--cập-nhật-driver--cuda)
5. [Bước 3 – Cài Docker & NVIDIA Container Toolkit](#5-bước-3--cài-docker--nvidia-container-toolkit)
6. [Bước 4 – Cài vLLM (CUDA 13 / aarch64)](#6-bước-4--cài-vllm-cuda-13--aarch64)
7. [Bước 5 – Khởi động Ray Cluster](#7-bước-5--khởi-động-ray-cluster)
8. [Bước 6 – Chạy 3 model song song](#8-bước-6--chạy-3-model-song-song)
9. [Bước 7 – Kiểm tra hoạt động](#9-bước-7--kiểm-tra-hoạt-động)
10. [Tối ưu hóa](#10-tối-ưu-hóa)
11. [Xử lý sự cố](#11-xử-lý-sự-cố)
12. [Recovery Guide](#12-recovery-guide--khôi-phục-máy-về-factory-state)

---

## 0. Cấu hình mạng 200G — RoCE + MTU 9000

> **Tài liệu tham khảo:** [NVIDIA DGX Spark — Spark Stacking](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html) · [NVIDIA dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)

Bước này phải hoàn thành trước tất cả các bước cài đặt phía sau. Nếu mạng 200G chưa được cấu hình đúng, NCCL và tensor-parallel sẽ không đạt đủ băng thông hoặc không hoạt động.

---

### 0.1 Kiến trúc mạng ConnectX-7

ConnectX-7 trên GX10 sử dụng kiến trúc **twin interface**: mỗi cổng QSFP vật lý chia ra 2 PCIe x4 link, tạo thành 4 interface logic:

```
1 cổng QSFP vật lý
├── PCIe x4 link 1 → enp1s0f1np1    (Ethernet)
│                  → rocep1s0f1      (RoCE/RDMA)
└── PCIe x4 link 2 → enP2p1s0f1np1  (Ethernet)
                   → roceP2p1s0f1    (RoCE/RDMA)
```

Mỗi twin cung cấp tối đa ~100G. Phải sử dụng cả 2 twin trong cấu hình NCCL để đạt full 200G.

---

### 0.2 Bước 1 — Xác định interface đang hoạt động (cả 2 máy)

```bash
ibdev2netdev
```

Kết quả mẫu:
```
rocep1s0f0   port 1 ==> enp1s0f0np0   (Down)
rocep1s0f1   port 1 ==> enp1s0f1np1   (Up)    ← cổng đang dùng
roceP2p1s0f0 port 1 ==> enP2p1s0f0np0 (Down)
roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)    ← twin của cổng trên
```

Ghi lại tên interface đang `Up` để dùng cho các bước tiếp theo. Nếu không có interface nào Up, kiểm tra lại cáp QSFP và reboot cả 2 máy.

---

### 0.3 Bước 2 — Cấu hình IP tĩnh + MTU 9000

#### Phương án A — Netplan (khuyến nghị, persistent qua reboot)

Thực hiện trên **cả 2 máy**:

```bash
# Tải netplan config chuẩn từ NVIDIA playbook
sudo wget -O /etc/netplan/40-cx7.yaml \
  https://github.com/NVIDIA/dgx-spark-playbooks/raw/main/nvidia/connect-two-sparks/assets/cx7-netplan.yaml

sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```

Gán IP tĩnh + MTU 9000:

```bash
# Node 1
sudo nmcli con modify enp1s0f1np1 \
  ipv4.addresses 192.168.100.10/24 \
  ipv4.method manual \
  802-3-ethernet.mtu 9000
sudo nmcli con down enp1s0f1np1 && sudo nmcli con up enp1s0f1np1

# Node 2
sudo nmcli con modify enp1s0f1np1 \
  ipv4.addresses 192.168.100.11/24 \
  ipv4.method manual \
  802-3-ethernet.mtu 9000
sudo nmcli con down enp1s0f1np1 && sudo nmcli con up enp1s0f1np1
```

Thay `enp1s0f1np1` bằng tên interface đang `Up` từ kết quả `ibdev2netdev` ở bước 0.2.

#### Phương án B — Gán IP thủ công (không persistent, mất sau reboot)

```bash
# Node 1
sudo ip addr add 192.168.100.10/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 up
sudo ip link set enp1s0f1np1 mtu 9000

# Node 2
sudo ip addr add 192.168.100.11/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 up
sudo ip link set enp1s0f1np1 mtu 9000
```

---

### 0.4 Bước 3 — Xác nhận kết nối và MTU

```bash
# Kiểm tra MTU đã đúng
ip link show enp1s0f1np1
# → mtu 9000

# Ping cơ bản
# Từ Node 1:
ping -c 3 192.168.100.11

# Từ Node 2:
ping -c 3 192.168.100.10

# Ping MTU test — xác nhận jumbo frame 9000 đi qua được
# 8972 = 9000 - 28 bytes (IP header + ICMP header)
# Từ Node 1:
ping -M do -s 8972 192.168.100.11

# Từ Node 2:
ping -M do -s 8972 192.168.100.10
```

---

### 0.5 Bước 4 — Kiểm tra RDMA / RoCE

```bash
# Xem danh sách RDMA devices
ibv_devices

# Kiểm tra trạng thái link
rdma link

# Kiểm tra kernel module
lsmod | grep mlx5
# mlx5_core và mlx5_ib phải có mặt
```

**Bandwidth test:**

```bash
# Node 1 — server mode (chạy trước)
ib_write_bw -d rocep1s0f1 -i 1 -p 12000 -F --report_gbits --run_infinitely

# Node 2 — client mode (chạy sau)
ib_write_bw -d rocep1s0f1 -i 1 -p 12000 -F --report_gbits 192.168.100.11

# Kỳ vọng: ~11.7 GB/s per twin
# Tổng 2 twin: ~23.4 GB/s (giới hạn thực tế của PCIe 5.0 x4 + link encoding)
```

---

### 0.6 Bước 5 — Cấu hình NCCL

Thêm vào `~/.bashrc` trên **cả 2 máy**:

```bash
cat >> ~/.bashrc << 'EOF'
# Khai báo cả 2 twin RoCE để đạt full 200G
export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1

# Interface điều khiển — dùng CX7, không dùng cổng 10G
export NCCL_SOCKET_IFNAME=enp1s0f1np1

# Bật RDMA
export NCCL_IB_DISABLE=0

# Bỏ comment dòng dưới nếu gặp lỗi GDR với unified memory
# export NCCL_NET_GDR_LEVEL=0
EOF

source ~/.bashrc
```

Nếu chỉ khai báo 1 interface trong `NCCL_IB_HCA`, NCCL chỉ sử dụng 1 PCIe x4 link, băng thông bị giới hạn ở ~100G.

---

### 0.7 Bước 6 — Thiết lập SSH passwordless (cho Ray)

NVIDIA cung cấp script tự động — chạy trên Node 1:

```bash
wget https://github.com/NVIDIA/dgx-spark-playbooks/raw/refs/heads/main/nvidia/connect-two-sparks/assets/discover-sparks
chmod +x discover-sparks
./discover-sparks
```

Hoặc cấu hình thủ công:

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
ssh-copy-id user@192.168.100.11
ssh user@192.168.100.11 "echo OK"
```

---

### 0.8 Quy hoạch mở rộng

Subnet `/24` cho phép gán thêm node mới chỉ bằng cách thêm IP mà không cần thay đổi cấu hình mạng hiện có:

| Node | IP |
|---|---|
| Node 1 (hiện tại) | 192.168.100.10 |
| Node 2 (hiện tại) | 192.168.100.11 |
| Node 3 (mở rộng) | 192.168.100.12 |
| Node 4+ | 192.168.100.13... |

Khi mở rộng lên 3 node trở lên, cần bổ sung L2 switch 200G (ví dụ MikroTik CRS812-DDQ hỗ trợ 8× 200G QSFP).

---

### 0.9 Checklist hoàn thành mạng

```
[ ] ibdev2netdev — thấy ít nhất 1 interface Up
[ ] ip link show  — mtu 9000
[ ] ping -c 3 192.168.100.11 từ Node 1 — 0% packet loss
[ ] ping -c 3 192.168.100.10 từ Node 2 — 0% packet loss
[ ] ping -M do -s 8972 — không có lỗi "Message too long"
[ ] lsmod | grep mlx5 — mlx5_core có mặt
[ ] rdma link — link active
[ ] ib_write_bw — ~11.7 GB/s per twin
[ ] NCCL_IB_HCA đã set trong .bashrc
[ ] SSH passwordless giữa 2 node hoạt động
```

---

## 1. Phân tích tài nguyên

### Ước tính bộ nhớ (FP8) — Unified Memory Architecture

| Model | Params | Bộ nhớ (FP8) | Node chạy |
|---|---|---|---|
| MedGemma 27B-IT | 27B | ~27 GB | Node 1 + Node 2 (tensor split) |
| Llama 4 Scout | 17B active / 109B MoE | ~50 GB (sau FP8) / ~202 GB (checkpoint gốc) | Node 1 + Node 2 (tp=2, Ray) |
| MedGemma 1.5 4B | 4B | ~4 GB | Node 1 |
| **Tổng** | | **~71 GB** | Tổng 256 GB — đủ dư |

128GB trên mỗi máy là bộ nhớ dùng chung giữa CPU và GPU. Khi hệ thống hết bộ nhớ, toàn bộ máy có thể bị treo thay vì chỉ crash một process. Blackwell GB10 có Tensor Core thế hệ 5 với hardware accelerator riêng cho FP8 và FP4, giúp tăng throughput ~2x so với FP16.

---

## 2. Kiến trúc triển khai

```
┌─────────────────────────────────────────────────────────────────────┐
│  Node 1 – Head  (192.168.100.10)                                     │
│                                                                      │
│  ● Ray Head Node  (port 6379)                                        │
│  ● MedGemma 27B-IT   (port 8000)  ──── tensor-parallel ────┐           │
│  ● MedGemma 1.5 4B    (port 8002)                            │           │
└─────────────────────────────────────────────────────────┼───────────┘
                   200G QSFP ConnectX-7 / RoCE            │
┌─────────────────────────────────────────────────────────┼───────────┐
│  Node 2 – Worker  (192.168.100.11)                       │           │
│                                                          │           │
│  ● Ray Worker Node                                       │           │
│  ● MedGemma 27B-IT   (nhận tensor từ Node 1) ───────────────┘           │
│  ● Llama 4 Scout  (port 8001)                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Bước 1 – Kiểm tra hệ thống & cấu hình nền

> Thực hiện trên **cả 2 máy**

### 3.1 Kiểm tra GPU và ConnectX-7

```bash
# Kiểm tra GPU GB10
lspci | grep -i nvidia
# Kỳ vọng: 000f:01:00.0 VGA compatible controller: NVIDIA Corporation Device 2e12

# Kiểm tra ConnectX-7 NIC
lspci | grep -i mellanox
# Kỳ vọng: ít nhất 2 dòng Mellanox MT2910 Family [ConnectX-7]

# Kiểm tra driver
nvidia-smi
# "Memory-Usage: Not Supported" là hành vi bình thường của unified memory
```

### 3.2 Kiểm tra kết nối mạng 200G

```bash
ip addr show

# Từ Node 1:
ping 192.168.100.11

# Từ Node 2:
ping 192.168.100.10
```

### 3.3 Cố định tên interface mạng

Tên interface có thể thay đổi sau reboot nếu không được cố định:

```bash
ip link show   # xem tên và MAC của ConnectX-7

# Tạo file cố định tên — thay <MAC_CX7> bằng địa chỉ MAC thực tế
sudo tee /etc/systemd/network/10-cx7.link << 'EOF'
[Match]
MACAddress=<MAC_CX7>

[Link]
Name=cx7
EOF

sudo systemctl restart systemd-networkd
```

### 3.4 Tắt swap

```bash
sudo swapoff -a

# Xác nhận đã tắt (không có output = đã tắt)
swapon --show

# Tắt vĩnh viễn
sudo sed -i '/swap/s/^/#/' /etc/fstab
```

### 3.5 Cài công cụ cần thiết

```bash
# Chỉ update danh sách package
sudo apt update

sudo apt install -y build-essential curl wget git
```

---

## 4. Bước 2 – Cập nhật Driver & CUDA

GX10 đã có DGX OS với driver cài sẵn. Thực hiện bước này khi cần cập nhật lên phiên bản mới hơn hoặc sau khi cài lại OS.

### 4.1 Cập nhật qua DGX Dashboard (khuyến nghị)

```bash
# Truy cập trực tiếp trên máy GX10
http://localhost:11000

# Hoặc SSH tunnel từ máy khác
ssh -L 11000:localhost:11000 user@192.168.100.10
# Mở: http://localhost:11000
```

Vào **System Updates** → cập nhật toàn bộ.

### 4.2 Cập nhật thủ công qua CLI

```bash
nvidia-smi | grep "Driver Version"

sudo apt update

sudo apt install -y \
  nvidia-driver-580-open \
  libnvidia-nscq \
  nvidia-modprobe \
  datacenter-gpu-manager-4-cuda13 \
  nv-persistence-mode

sudo reboot
```

### 4.3 Nâng lên driver 590 + CUDA 13.1

```bash
wget https://developer.download.nvidia.com/compute/cuda/13.1.1/local_installers/cuda_13.1.1_590.48.01_linux_sbsa.run

sudo sh cuda_13.1.1_590.48.01_linux_sbsa.run \
  --extract=/tmp/cuda_extract

sudo sh /tmp/cuda_extract/NVIDIA-Linux-aarch64-590.48.01.run \
  --silent -m kernel-open
```

### 4.4 Cấu hình biến môi trường CUDA

```bash
cat >> ~/.bashrc << 'EOF'
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
EOF

source ~/.bashrc
```

### 4.5 Kiểm tra

```bash
nvidia-smi | grep "Driver Version"
# → 580.x hoặc 590.x

nvcc --version
# → CUDA 13.x

python3 -c "
import subprocess
result = subprocess.run(
  ['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'],
  capture_output=True, text=True)
print(result.stdout)
"
# → NVIDIA GB10, 12.1
```

---

## 5. Bước 3 – Cài Docker & NVIDIA Container Toolkit

> Thực hiện trên **cả 2 máy** — GX10 với DGX OS thường đã có Docker, kiểm tra trước khi cài.

```bash
docker --version && nvidia-ctk --version

# Nếu chưa có Docker:
curl -fsSL https://get.docker.com | sudo bash
sudo usermod -aG docker $USER
newgrp docker

# Nếu chưa có NVIDIA Container Toolkit:
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Xác nhận
docker run --rm --runtime nvidia --gpus all ubuntu nvidia-smi
```

---

## 6. Bước 4 – Cài vLLM (CUDA 13 / aarch64)

GX10 dùng CUDA 13.x — wheel vLLM thông thường từ PyPI biên dịch cho CUDA 12.x sẽ báo lỗi `libcudart.so.12: cannot open shared object file`. Phải dùng một trong 2 phương pháp dưới đây.

### Phương pháp A — Docker Image (khuyến nghị cho multi-node với Ray)

> Thực hiện trên **cả 2 máy**

```bash
docker pull vllm/vllm-openai:latest

docker images | grep vllm
```

Nếu image báo lỗi CUDA khi chạy, chỉ định tag cụ thể hỗ trợ aarch64/cu130:

```bash
docker pull vllm/vllm-openai:v0.8.0
```

### Phương pháp B — Python virtualenv với wheel cu130

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

uv venv ~/.venv-vllm --python 3.12
source ~/.venv-vllm/bin/activate

# PyTorch cu130 — có wheel aarch64 chính thức
uv pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130

# vLLM cu130 cho aarch64
uv pip install -U vllm \
  --extra-index-url https://wheels.vllm.ai/nightly/cu130

# flash-attn không cần cài — vLLM đã bundle FlashInfer

python3 -c "import vllm; print('vLLM version:', vllm.__version__)"
```

---

## 7. Bước 5 – Khởi động Ray Cluster

### 7.1 Mở port cần thiết (cả 2 máy)

```bash
sudo ufw allow 6379
sudo ufw allow 8265
sudo ufw allow 8000
sudo ufw allow 8001
sudo ufw allow 8002
sudo ufw allow 10000:20000/tcp
```

### 7.2 Node 1 – Head Node

```bash
docker run -d \
  --runtime nvidia \
  --gpus all \
  --network host \
  --name ray-head \
  --restart unless-stopped \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  ray start --head \
    --node-ip-address=192.168.100.10 \
    --port=6379 \
    --block
```

### 7.3 Node 2 – Worker Node

```bash
docker run -d \
  --runtime nvidia \
  --gpus all \
  --network host \
  --name ray-worker \
  --restart unless-stopped \
  vllm/vllm-openai:latest \
  ray start \
    --address=192.168.100.10:6379 \
    --block
```

### 7.4 Xác nhận cluster

```bash
docker exec ray-head ray status
# Kỳ vọng: 2 nodes active
```

---

## 8. Bước 6 – Chạy 3 model song song

Cần HuggingFace Token và đã accept license (xem mục 10). Thứ tự load: 4B → 27B → Scout.

### 8.1 MedGemma 1.5 4B – Node 1, port 8002

```bash
docker run -d \
  --runtime nvidia --gpus all --network host \
  --name medgemma-1.5-4b --restart unless-stopped \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
  vllm/vllm-openai:latest \
  google/medgemma-1.5-4b-it \
    --tensor-parallel-size 1 \
    --dtype auto \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-model-len 4096 \
    --port 8002
```

Kiểm tra inference hoạt động trước khi load model tiếp theo.

> **Lưu ý image entrypoint:** `vllm/vllm-openai` đã có sẵn entrypoint là `vllm serve` — chỉ truyền thẳng tên model và các tham số, không lặp lại `vllm serve` trong lệnh docker run.

### 8.2 MedGemma 27B-IT – Span 2 node, port 8000

```bash
docker run -d \
  --runtime nvidia --gpus all --network host \
  --name medgemma-27b --restart unless-stopped \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
  vllm/vllm-openai:latest \
  google/medgemma-27b-it \
    --tensor-parallel-size 2 \
    --dtype auto \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-model-len 8192 \
    --port 8000
```

`--tensor-parallel-size 2` — Ray tự phân phối tensor sang Node 2 qua kết nối 200G.

### 8.3 Llama 4 Scout – Span 2 node, port 8001

Llama 4 Scout có checkpoint ~202GB — vượt quá 128GB của 1 node. Bắt buộc phải span 2 node qua Ray cluster. Phải khởi động Ray Head và Ray Worker (Bước 5) trước khi chạy lệnh này.

**Bước 1 — Xác nhận Ray cluster đã có 2 node:**
```bash
docker exec ray-head ray status
# → phải thấy 2 nodes active
```

**Bước 2 — Chạy Llama 4 Scout trên Node 1:**
```bash
docker run -d \
  --runtime nvidia --gpus all --network host \
  --name llama4-scout --restart unless-stopped \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN \
  vllm/vllm-openai:latest \
  meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --tensor-parallel-size 2 \
    --distributed-executor-backend ray \
    --dtype auto \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-model-len 8192 \
    --port 8001
```

> `--tensor-parallel-size 2` + `--distributed-executor-backend ray` → Ray tự phân phối model sang Node 2 qua kết nối 200G.

> **FP8 Cache:** Lần đầu chạy, vLLM sẽ convert weights sang FP8 (~10-15 phút). Các lần chạy sau sẽ load từ cache, không cần convert lại.

**Theo dõi tiến trình:**
```bash
docker logs -f llama4-scout
# Chờ đến khi thấy:
# INFO: Uvicorn running on http://0.0.0.0:8001
```

---

## 9. Bước 7 – Kiểm tra hoạt động

### 9.1 Kiểm tra models đã load

```bash
curl http://192.168.100.10:8000/v1/models   # MedGemma 27B-IT
curl http://192.168.100.10:8001/v1/models   # Llama 4 Scout
curl http://192.168.100.10:8002/v1/models   # MedGemma 1.5 4B
```

### 9.2 Test inference

```bash
curl http://192.168.100.10:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/medgemma-1.5-4b-it",
    "messages": [{"role": "user", "content": "Triệu chứng của viêm phổi là gì?"}],
    "max_tokens": 200
  }'
```

### 9.3 Theo dõi tài nguyên

```bash
nvidia-smi

free -h

# DGX Dashboard — giao diện đồ họa đầy đủ
# http://localhost:11000

docker logs -f medgemma-1.5-4b
docker logs -f medgemma-27b
docker logs -f llama4-scout
```

---

## 10. Tối ưu hóa

### HuggingFace Token

Cả 3 model là gated model, phải accept license trước khi download:

- MedGemma 27B-IT → https://huggingface.co/google/medgemma-27b-it
- MedGemma 1.5 4B → https://huggingface.co/google/medgemma-1.5-4b-it
- Llama 4 Scout → https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
- Tạo token → https://huggingface.co/settings/tokens

Khai báo token trên **cả 2 máy**:

```bash
echo 'export HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxx' >> ~/.bashrc
source ~/.bashrc
```

Kiểm tra đã nhận chưa:

```bash
echo $HUGGING_FACE_HUB_TOKEN
# → phải hiện ra token, không phải dòng trống
```

Khi chạy `docker run`, biến `$HUGGING_FACE_HUB_TOKEN` được truyền vào container qua flag `-e HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN`.

### Thứ tự khởi động

```
1. Xác nhận driver (nvidia-smi) — 580.x hoặc 590.x
2. Xác nhận CUDA (nvcc --version) — 13.x
3. Tắt swap (swapoff -a)
4. Xác nhận mạng 200G (ping + ib_write_bw)
5. Khởi động Ray Head  (Node 1)
6. Khởi động Ray Worker (Node 2)
7. Load MedGemma 1.5 4B   → test curl
8. Load MedGemma 27B-IT  → test curl
9. Load Llama 4 Scout → test curl
```

### FP8 Quantization

vLLM dùng 2 flag riêng biệt để bật FP8, không phải `--dtype fp8`:

| Flag | Tác dụng | Giá trị |
|---|---|---|
| `--dtype` | Data type tính toán | `auto` — vLLM tự chọn tối ưu |
| `--quantization` | Quantize weights model | `fp8` |
| `--kv-cache-dtype` | Quantize KV cache | `fp8` hoặc `fp8_e4m3` |

```bash
# Ví dụ kết hợp đầy đủ
vllm serve <model> \
  --dtype auto \
  --quantization fp8 \
  --kv-cache-dtype fp8
```

### Chạy vLLM không qua Docker

Khi Docker image không tương thích CUDA 13:

```bash
source ~/.venv-vllm/bin/activate

vllm serve google/medgemma-1.5-4b-it \
  --dtype auto \
  --quantization fp8 \
  --kv-cache-dtype fp8 \
  --max-model-len 4096 \
  --port 8002
```

---

## 11. Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `nvidia-smi` báo "Memory-Usage: Not Supported" | Hành vi bình thường của unified memory | Dùng `free -h` để xem bộ nhớ thực tế |
| Lỗi `libcudart.so.12` khi import vLLM | Wheel CUDA 12 không tương thích với GX10 | Cài lại vLLM từ wheel cu130 (Bước 6) |
| Máy treo hoàn toàn khi hết bộ nhớ | Swap chưa tắt | `sudo swapoff -a` và cập nhật `/etc/fstab` |
| Ray worker không kết nối được vào cluster | Firewall chặn port | `sudo ufw allow 6379` và các port liên quan |
| `ImportError` khi import vLLM | CUDA version mismatch | Kiểm tra `nvcc --version` phải ra 13.x |
| Download model chậm | Lần đầu tải về | Model được cache tại `~/.cache/huggingface` |
| Lỗi `401 Unauthorized` từ HuggingFace | Token sai hoặc chưa accept license | Vào web accept license, kiểm tra lại token |
| Tên interface 200G thay đổi sau reboot | Chưa cố định tên interface | Tạo file `/etc/systemd/network/10-cx7.link` (Bước 3.3) |
| Container tự restart liên tục | Lỗi bộ nhớ hoặc CUDA | `docker logs --tail 100 <tên_container>` |
| Máy không boot được sau cài đặt driver sai | Driver không tương thích với kernel DGX OS | Boot recovery mode, gỡ driver sai, cài lại `nvidia-driver-580-open` |
| NCCL chỉ đạt ~100G thay vì 200G | Chỉ khai báo 1 twin trong NCCL_IB_HCA | Set `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1` trong .bashrc |

---

## 12. Recovery Guide — Khôi phục máy về factory state

> **Nguồn chính thức:** [ASUS FAQ — How to Update ASUS Ascent GX10 OS](https://www.asus.com/us/support/faq/1056213/)

Recovery sẽ xóa toàn bộ dữ liệu trên SSD và cài lại DGX OS về trạng thái factory. Có 3 phương án tùy tình trạng máy.

---

### 12.1 Kiểm tra phiên bản OS hiện tại (qua BIOS)

```
1. Giữ [Del] khi khởi động → vào BIOS
2. Vào Advanced → Firmware Version Information
3. Xem OS version hiện tại
```

---

### 12.2 Phương án 1 — Cập nhật qua DGX Dashboard (máy còn vào được)

```
1. Đăng nhập hệ thống GX10
2. Mở Ubuntu Start Menu
3. Tìm và mở NVIDIA DGX Dashboard
4. Đăng nhập bằng tài khoản user hệ thống
5. Vào Settings → Update → Click "Update Now"
6. Chờ cập nhật hoàn tất
```

Hoặc truy cập trực tiếp qua trình duyệt:
```
http://localhost:11000
```

---

### 12.3 Phương án 2 — Cập nhật firmware qua Terminal (offline)

Tải gói firmware từ [trang support ASUS](https://www.asus.com/vn/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/helpdesk_download/), giải nén rồi chạy:

```bash
# SOC Firmware
sudo su
./capsule_update.sh bsp_v11_socfw302_bios_0101_ec_2.66.3.3_signed.cap

# EC Firmware
./capsule_update.sh ec_2.66.3.3_signed.cap

# PD Firmware (cần reboot và chạy lại 2 lần)
./capsule_update.sh usbpd_50.cap
# Reboot thủ công từ Ubuntu
./capsule_update.sh usbpd_50.cap

# TPM Firmware
./capsule_update.sh tpm_7.2.4.1.cap
```

---

### 12.4 Phương án 3 — Cài lại OS qua USB (máy không vào được OS)

#### Chuẩn bị

| Thiết bị | Yêu cầu |
|---|---|
| USB flash drive | 16GB trở lên — dữ liệu trên USB sẽ bị xóa |
| Bàn phím | Cắm trực tiếp qua cổng USB |
| Màn hình | Cắm qua HDMI vào GX10 |
| Máy tính phụ | Để tải ISO và tạo USB |

#### Bước 1 — Tải ISO từ ASUS

Vào trang support chính thức của ASUS, chọn Linux OS, tải:
- **USB Formatting Utility** (công cụ ghi ISO ra USB)
- **ISO file** (file cài đặt DGX OS)

```
https://www.asus.com/vn/networking-iot-servers/desktop-ai-supercomputer/
ultra-small-ai-supercomputers/asus-ascent-gx10/helpdesk_download/
```

#### Bước 2 — Ghi ISO ra USB

Dùng USB Formatting Utility vừa tải để ghi ISO vào USB drive.

#### Bước 3 — Vào BIOS

```
1. Cắm USB vào GX10
2. Bật máy, giữ [Del] ngay từ đầu để vào BIOS
3. Vào Boot → Boot Option #1 → UEFI: USB USB Hard Drive, Partition 1
4. Nhấn [F4] để Save and Exit
```

#### Bước 4 — Cài đặt OS

```
[Menu] Chọn "DGX Spark Installation Options"
       → Chọn "Install DGX OS 7.x for DGX Spark"
       → Chờ cài đặt hoàn tất
```

#### Bước 5 — First Boot OOBE

```
1.  Click "Get Started"
2.  Chọn Language và Timezone → Continue
3.  Chọn Keyboard Layout → Continue
4.  Accept license → Continue
5.  Tạo Username và Password → Continue
6.  Chọn tùy chọn NVIDIA improvement → Continue
7.  Kết nối Wi-Fi hoặc cắm LAN 10G
8.  Chờ cài đặt hoàn tất (~10-15 phút)
```

---

### 12.5 Checklist sau Recovery

```bash
# Tắt swap
sudo swapoff -a
sudo sed -i '/swap/s/^/#/' /etc/fstab

# Xem interface ConnectX-7 để cấu hình lại mạng 200G
ip link show

# Kiểm tra driver
nvidia-smi | grep "Driver Version"
# → 580.x hoặc 590.x

# Lưu HuggingFace token
echo 'export HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxx' >> ~/.bashrc
source ~/.bashrc
```

Sau đó thực hiện lại từ Mục 0 trong tài liệu này.

---

### 12.6 Liên hệ hỗ trợ

| | Link |
|---|---|
| ASUS Support GX10 | https://www.asus.com/vn/support/ |
| ASUS FAQ GX10 | https://www.asus.com/support/faq/1056142/ |
| ASUS GX10 GitHub Discussions | https://github.com/orgs/asus-ascent-gx10/discussions |
| NVIDIA Developer Forums (GB10) | https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/719 |

---

*Phiên bản: 05/2026 · Tài liệu kỹ thuật nội bộ · Dựa trên ASUS Ascent GX10 Official Documentation, NVIDIA DGX Spark User Guide và NVIDIA DGX OS 7 User Guide*

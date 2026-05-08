# Hướng dẫn cài đặt vLLM & 3 Model AI trên cụm 2 máy ASUS Ascent GX10

> **Sản phẩm:** ASUS Ascent GX10 · GPU NVIDIA Blackwell GB10 · ARM v9.2-A · 128GB LPDDR5x Unified Memory/node
> **Hệ điều hành:** NVIDIA DGX™ Base OS (Ubuntu Linux) — cài sẵn, OS duy nhất được hỗ trợ chính thức
> **Mạng nội bộ:** Node 1 `192.168.100.10/24` ↔ Node 2 `192.168.100.11/24` qua NVIDIA ConnectX-7 200G
> **Models:** MedGemma 27B-IT · MedGemma 1.5 4B IT · Llama 4 Scout 17B (chạy song song)
> **Tài liệu tham khảo:** [ASUS Ascent GX10](https://www.asus.com/vn/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/) · [ASUS FAQ GX10](https://www.asus.com/support/faq/1056142/) · [NVIDIA DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)

---

## Thông số kỹ thuật ASUS Ascent GX10

> **Nguồn:** [ASUS Ascent GX10 Tech Specs](https://www.asus.com/vn/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/techspec/)

| Thành phần            | Thông số                                                                               |
| --------------------- | -------------------------------------------------------------------------------------- |
| **Tên sản phẩm**      | ASUS Ascent GX10                                                                       |
| **CPU**               | ARM v9.2-A (GB10 Superchip)                                                            |
| **GPU**               | NVIDIA Blackwell GPU (GB10, integrated)                                                |
| **Memory**            | 128 GB LPDDR5x — unified system memory (CPU+GPU dùng chung)                            |
| **Storage**           | 1TB M.2 NVMe PCIe 4.0 SSD                                                              |
| **Mạng**              | 1x NVIDIA ConnectX-7 SmartNIC (200G QSFP) · 1x 10G LAN · Wi-Fi 7 · Bluetooth 5.4       |
| **Cổng I/O**          | 3x USB 3.2 Gen 2x2 Type-C · 1x USB-C PD 180W · 1x HDMI 2.1 · 1x Kensington Lock        |
| **Nguồn**             | 240W Adapter                                                                           |
| **Kích thước**        | 150 x 150 x 51 mm · 1.48 kg                                                            |
| **OS**                | Ubuntu Linux (NVIDIA DGX™ Base OS) — OS duy nhất được kiểm tra và hỗ trợ chính thức    |
| **Hiệu năng AI**      | 1 petaFLOP (FP4)                                                                       |
| **Driver**            | Dòng 580/590-open — dành riêng cho Blackwell GB10                                      |
| **CUDA**              | Phiên bản 13.x — GB10 (sm_121) yêu cầu CUDA 13 trở lên                                 |
| **nvidia-smi**        | Báo "Memory-Usage: Not Supported" — hành vi bình thường của unified memory             |
| **vLLM**              | Yêu cầu wheel CUDA 13 / aarch64 — không dùng pip install thông thường                  |
| **Swap**              | Phải tắt trước khi vận hành — OOM trên unified memory có thể làm treo toàn bộ hệ thống |
| **Cập nhật hệ thống** | Thực hiện qua DGX Dashboard (`http://localhost:11000`) — không dùng `apt upgrade`      |
| **Cluster**           | Hỗ trợ kết nối tối đa 2 node qua QSFP 200G trực tiếp, hoặc nhiều hơn qua 200G switch   |

---

## Mục lục

0. [Cấu hình mạng 200G — RoCE + MTU](#0-cấu-hình-mạng-200g--roce--mtu)
1. [Phân tích tài nguyên](#1-phân-tích-tài-nguyên)

---

## I. Phân tích tài nguyên

### 1. Ước tính bộ nhớ (FP8) — Unified Memory Architecture

| Model              | Params | Bộ nhớ (FP8) | Node chạy                      |
| ------------------ | ------ | ------------ | ------------------------------ |
| MedGemma 27B-IT    | 27B    | ~27 GB       | Node 1 + Node 2 (tensor split) |
| MedGemma 1.5 4B IT | 4B     | ~4 GB        | Node 1                         |
| Llama 4 Scout 17B  | 17B    | ~50 GB       | Node 2                         |
| **Tổng**           |        | **~75 GB**   | Tổng 128 GB — đủ dư            |

128GB trên mỗi máy là bộ nhớ dùng chung giữa CPU và GPU. Khi hệ thống hết bộ nhớ, toàn bộ máy có thể bị treo thay vì chỉ crash một process. Blackwell GB10 có Tensor Core thế hệ 5 với hardware accelerator riêng cho FP8 và FP4, giúp tăng throughput ~2x so với FP16.

---

### 2. Kiến trúc triển khai

```
┌─────────────────────────────────────────────────────────────────────┐
│  Node 1 – Head  (192.168.100.10)                                    │
│                                                                     │
│  ● Ray Head Node  (port 6379)                                       │
│  ● MedGemma 27B-IT   (port 8000)  ──── tensor-parallel ────┐        │
│  ● MedGemma 1.5 4B-IT    (port 8002)                       │        │
└────────────────────────────────────────────────────────────┼────────┘
                   200G QSFP ConnectX-7 / RoCE               │
┌────────────────────────────────────────────────────────────┼────────┐
│  Node 2 – Worker  (192.168.100.11)                         │        │
│                                                            │        │
│  ● Ray Worker Node                                         │        │
│  ● MedGemma 27B-IT   (nhận tensor từ Node 1) ──────────────┘        │
│  ● Llama 4 Scout 17B  (port 8001)                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3. Kiểm tra hệ thống & cấu hình nền

> Thực hiện trên **cả 2 máy**

#### 3.1 Kiểm tra GPU và ConnectX-7

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

#### 3.2 Tắt swap

```bash
sudo swapoff -a

# Xác nhận đã tắt (không có output = đã tắt)
swapon --show

# Tắt vĩnh viễn
sudo sed -i '/swap/s/^/#/' /etc/fstab
```

#### 3.3 Cài công cụ cần thiết

```bash
# Chỉ update danh sách package
sudo apt update

sudo apt install -y build-essential curl wget git
```

---

## II. Cấu hình mạng 200G — RoCE + MTU

> **Tài liệu tham khảo:** [NVIDIA DGX Spark — Spark Stacking](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html) · [NVIDIA dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)

Bước này phải hoàn thành trước tất cả các bước cài đặt phía sau. Nếu mạng 200G chưa được cấu hình đúng, NCCL và tensor-parallel sẽ không đạt đủ băng thông hoặc không hoạt động.

---

### 1 Kiến trúc mạng ConnectX-7

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

#### 2. Thiết lập khuyến nghị

Thực hiện trên **cả 2 máy**:

```bash
sudo wget -O /etc/netplan/40-cx7.yaml https://github.com/NVIDIA/dgx-spark-playbooks/raw/main/nvidia/connect-two-sparks/assets/cx7-netplan.yaml
sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```

Tạo thủ công khi nội dung tệp [cx7-netplan.yaml](https://www.asus.com/vn/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/techspec/) khi không thể kết nối bằng `wget`.

```xml
network:
  version: 2
  ethernets:
    enp1s0f0np0:
      link-local: [ ipv4 ]
    enp1s0f1np1:
      link-local: [ ipv4 ]
```

---

### 3 — Kiểm tra RDMA / RoCE

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

### 4 — Thiết lập SSH passwordless (cho Ray)

NVIDIA cung cấp script tự động — chạy trên Node 1:

```bash
wget https://github.com/NVIDIA/dgx-spark-playbooks/raw/refs/heads/main/nvidia/connect-two-sparks/assets/discover-sparks
chmod +x discover-sparks
./discover-sparks
```

---

### 5 — Quy hoạch mở rộng

Subnet `/24` cho phép gán thêm node mới chỉ bằng cách thêm IP mà không cần thay đổi cấu hình mạng hiện có:

| Node              | IP                |
| ----------------- | ----------------- |
| Node 1 (hiện tại) | 192.168.100.10    |
| Node 2 (hiện tại) | 192.168.100.11    |
| Node 3 (mở rộng)  | 192.168.100.12    |
| Node 4+           | 192.168.100.13... |

Khi mở rộng lên 3 node trở lên, cần bổ sung L2 switch 200G (ví dụ MikroTik CRS812-DDQ hỗ trợ 8× 200G QSFP).

---

## III. Driver & CUDA

GX10 đã có DGX OS với driver cài sẵn. Thực hiện bước này khi cần cập nhật lên phiên bản mới hơn hoặc sau khi cài lại OS.

### Cập nhật qua DGX Dashboard (khuyến nghị)

```bash
# Truy cập trực tiếp trên máy GX10
http://localhost:11000

# Hoặc SSH tunnel từ máy khác
ssh -L 11000:localhost:11000 user@192.168.100.10
# Mở: http://localhost:11000
```

Vào **System Updates** → cập nhật toàn bộ.

---

### Cài Docker & NVIDIA Container Toolkit

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

## IV. Cài vLLM

GX10 dùng CUDA 13.x — wheel vLLM thông thường từ PyPI biên dịch cho CUDA 12.x sẽ báo lỗi `libcudart.so.12: cannot open shared object file`. Phải dùng một trong 2 phương pháp dưới đây.

### 1. Cài đặt vLLM

Github [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker/).

```
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
```

### 2. HuggingFace Token

Cả 3 model là gated model, phải accept license trước khi download:

- MedGemma 27B-IT → https://huggingface.co/google/medgemma-27b-it
- MedGemma 1.5 4B → https://huggingface.co/google/medgemma-1.5-4b-it
- Llama 4 Scout → https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
- Tạo token → https://huggingface.co/settings/tokens

Khai báo token trên **cả 2 máy**:

```bash
hf auth login

echo 'export HF_TOKEN=hf_xxxxxxxxxxxxxxxx' >> ~/.bashrc
source ~/.bashrc
```

Kiểm tra đã nhận chưa:

```bash
hf auth whoami
echo $HF_TOKEN
# → phải hiện ra token, không phải dòng trống
```

### 3. Tạo các tệp khai báo cấu hình cho vLLM trong thư mục `recipes`

**`MedGemma-4b-1.5-it-bf16.yaml`**:

```xml
recipe_version: "1"
name: MedGemma-1.5-4B-IT
description: vLLM serving MedGemma-1.5-4B-IT (bfloat16)

model: google/medgemma-1.5-4b-it

cluster_only: false
solo_only: false

container: vllm-node-tf5-medgemma4b

build_args:
    - --tf5

mods: []

defaults:
    port: 8002
    host: 0.0.0.0
    gpu_memory_utilization: 0.1
    max_model_len: 8192
    max_num_batched_tokens: 8192
    max_num_seqs: 8
env: {}

command: |
    vllm serve google/medgemma-1.5-4b-it \
      --dtype bfloat16 \
      --quantization fp8 \
      --kv-cache-dtype fp8 \
      --gpu-memory-utilization {gpu_memory_utilization}
      --max-model-len {max_model_len} \
      --max-num-batched-tokens {max_num_batched_tokens} \
      --max-num-seqs {max_num_seqs} \
      --port {port} \
      --host {host}
```

---

**`MedGemma-27b-it-bf16.yaml`**:

```xml
recipe_version: "1"
name: MedGemma-27B-IT
description: vLLM serving MedGemma-27B-IT (bfloat16)

model: google/medgemma-27b-it

cluster_only: false
solo_only: false

container: vllm-node-tf5-medgemma27b

build_args:
    - --tf5

mods: []

defaults:
    port: 8001
    host: 0.0.0.0
    gpu_memory_utilization: 0.7
    max_model_len: 65536
    max_num_batched_tokens: 98304
    max_num_seqs: 8
    tensor_parallel: 2

env: {}

command: |
    vllm serve google/medgemma-27b-it \
      --dtype bfloat16 \
      --quantization fp8 \
      --kv-cache-dtype fp8 \
      --gpu-memory-utilization {gpu_memory_utilization} \
      --max-model-len {max_model_len} \
      --max-num-batched-tokens {max_num_batched_tokens} \
      --max-num-seqs {max_num_seqs} \
      --port {port} \
      --host {host} \
      --load-format fastsafetensors \
      --enable-prefix-caching \
      --enable-chunked-prefill \
      --enable-auto-tool-choice \
      --tool-call-parser pythonic \
      -tp {tensor_parallel} --distributed-executor-backend ray
```

---

**`Llama-4-Scout-17B-16E-Instruct-fb8.yaml`**:

```xml
recipe_version: "1"
name: Llama-4-Scout-17B-16E-Instruct
description: vLLM serving Llama-4-Scout-17B-16E-Instruct (bfloat16)

model: nvidia/Llama-4-Scout-17B-16E-Instruct-FP8

cluster_only: false
solo_only: false

container: vllm-node-tf5-llama4scout

build_args:
    - --tf5

mods: []

defaults:
    port: 8000
    host: 0.0.0.0
    gpu_memory_utilization: 0.9
    max_model_len: 8192
    max_num_batched_tokens: 8192
    max_num_seqs: 8
    tensor_parallel: 2

env: {}

command: |
    vllm serve nvidia/Llama-4-Scout-17B-16E-Instruct-FP8 \
      --dtype auto \
      --quantization fp8 \
      --kv-cache-dtype fp8 \
      --gpu-memory-utilization {gpu_memory_utilization} \
      --max-model-len {max_model_len} \
      --max-num-batched-tokens {max_num_batched_tokens} \
      --max-num-seqs {max_num_seqs} \
      --port {port} \
      --host {host} \
      --load-format fastsafetensors \
      --enable-prefix-caching \
      -tp {tensor_parallel} --distributed-executor-backend ray
```

---

### Chạy vLLM theo thứ tự và kiểm tra trước khi chạy model tiếp theo.

#### Node 1 (chạy trên node 1 tự động sync sang node 2 qua Ray cluster)

```bash
cd ~/spark-vllm-docker
./run-recipe.sh MedGemma-27b-it-bf16 --setup --name=vllm_node_27b
```

#### Node 2 (only node 2)

```bash
cd ~/spark-vllm-docker
./run-recipe.sh Llama-4-Scout-17B-16E-Instruct-fb8 --setup --name=vllm_node_llama4scout
```

#### Node 1 (only node 1)

```bash
cd ~/spark-vllm-docker
./run-recipe.sh MedGemma-4b-1.5-it-bf16 --setup --name=vllm_node_4b
```

#### Kiểm tra models đã load

```bash
curl http://192.168.100.10:8000/v1/models   # MedGemma 27B-IT
curl http://192.168.100.10:8002/v1/models   # MedGemma 1.5 4B-IT
curl http://192.168.100.11:8001/v1/models   # Llama 4 Scout 17B
```

#### Test inference

```bash
curl http://192.168.100.10:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/medgemma-1.5-4b-it",
    "messages": [{"role": "user", "content": "Triệu chứng của viêm phổi là gì?"}],
    "max_tokens": 200
  }'
```

#### Theo dõi tài nguyên

```bash
nvidia-smi

free -h

# DGX Dashboard — giao diện đồ họa đầy đủ
# http://localhost:11000

docker logs -f vllm_node_4b
docker logs -f vllm_node_27b
docker logs -f vllm_node_llama4scout
```

---

### 12.6 Liên hệ hỗ trợ

|                                | Link                                                                           |
| ------------------------------ | ------------------------------------------------------------------------------ |
| ASUS Support GX10              | https://www.asus.com/vn/support/                                               |
| ASUS FAQ GX10                  | https://www.asus.com/support/faq/1056142/                                      |
| ASUS GX10 GitHub Discussions   | https://github.com/orgs/asus-ascent-gx10/discussions                           |
| NVIDIA Developer Forums (GB10) | https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/719 |

---

_Phiên bản: 05/2026 · Tài liệu kỹ thuật nội bộ · Dựa trên ASUS Ascent GX10 Official Documentation, NVIDIA DGX Spark User Guide và NVIDIA DGX OS 7 User Guide_

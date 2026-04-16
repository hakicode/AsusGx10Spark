Dưới đây là phiên bản tài liệu được chuẩn hoá lại theo hướng **rõ ràng – có thứ tự – dễ triển khai thực tế** cho cụm cluster sử dụng **ASUS Ascent GX10**.
Thông tin thiết bị xem tại: [ASUS-ASCENT-GX10](https://www.asus.com/vn/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10)

---

# HƯỚNG DẪN THIẾT LẬP CLUSTER ASUS GX10 (MULTI-NODE vLLM)

## 1. Chuẩn bị hệ thống

### 1.1 Yêu cầu phần cứng

- Tối thiểu: 2 node **GX10**
- Kết nối tốc độ cao:
    - **NVIDIA ConnectX-7 (InfiniBand / Ethernet)**

- Dây kết nối tương thích (DAC / Fiber)

---

## 2. Cài đặt hệ điều hành

### 2.1 OS thống nhất

- Sử dụng: **Ubuntu 24.04 LTS**
- Cài đặt giống nhau trên tất cả node

### 2.2 Chuẩn hoá user

- Tạo cùng 1 user logic trên tất cả node:

```bash
sudo adduser lvai
sudo usermod -aG sudo lvai
```

- Đảm bảo:
    - UID/GID giống nhau giữa các node (quan trọng khi mount / share)

```bash
id lvai
```

---

## 3. Thiết lập kết nối mạng cluster

### 3.1 Kết nối vật lý

- Cắm cáp **ConnectX-7**:
    - slot 1 ↔ slot 1 (khuyến nghị)
    - hoặc slot 2 ↔ slot 2

> Tránh cắm lệch slot → dễ gây mismatch topology

---

### 3.2 Kiểm tra interface

```bash
ip a
```

Tìm interface dạng:

- `enp*`
- `ib*` (nếu dùng InfiniBand)

---

### 3.3 Gán IP tĩnh (ví dụ)

Node 1:

```bash
192.168.100.1/24
```

Node 2:

```bash
192.168.100.2/24
```

Config netplan:

```yaml
network:
    version: 2
    ethernets:
        enp1s0f0:
            addresses:
                - 192.168.100.1/24
```

Apply:

```bash
sudo netplan apply
```

---

### 3.4 Kiểm tra kết nối

```bash
ping 192.168.100.2
```

---

### 3.5 SSH không mật khẩu

```bash
ssh-keygen -t rsa
ssh-copy-id lvai@192.168.100.2
```

---

### 3.6 Tham chiếu tài liệu NVIDIA

- NVIDIA DGX Spark clustering guideline:
    - [https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)

---

## 4. Cài đặt môi trường Docker

```bash
sudo apt update
sudo apt install -y docker.io
```

Fix permission:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

Kiểm tra:

```bash
docker run hello-world
```

### 4.1 Cài đặt Python và Hugging Face

```bash
sudo apt install python3-venv -y
python3 -m venv ~/venv
source ~/venv/bin/activate

pip install huggingface_hub

hf auth login
hf auth whoami
```

Ghi chú:

- Kích hoạt môi trường `~/venv` trước khi chạy `pip` và `hf`.
- Nếu recipe vẫn yêu cầu token môi trường, vẫn có thể dùng `export HF_TOKEN="xxx"` ở phần dưới.

---

## 5. Cài đặt Spark + vLLM (Multi-node)

### 5.1 Clone repo

```bash
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
```

---

### 5.2 Build cluster

```bash
./build-and-copy.sh -c
```

Ý nghĩa:

- `-c` = cluster mode (multi-node)
- Script sẽ:
    - Build image
    - Copy config sang node khác
    - Setup distributed runtime

---

### 5.3 Kiểm tra container

```bash
docker ps
```

---

## 6. Cài đặt mô hình LLM

### 6.0 Chuẩn bị Hugging Face token

Một số model cần quyền truy cập từ Hugging Face. Có thể đăng nhập bằng CLI trước, hoặc khai báo token trước khi chạy recipe:

```bash
hf auth login
hf auth whoami
```

Hoặc dùng biến môi trường:

```bash
export HF_TOKEN="xxx"
```

Nếu muốn lưu lâu dài, thêm biến này vào `~/.bashrc` hoặc `~/.zshrc`.

### 6.1 Thư mục recipes

```bash
cd recipes
```

---

### 6.2 Chọn model

Ví dụ:

- LLaMA
- Mistral
- Qwen

---

### 6.3 Deploy model

Theo từng recipe:

```bash
./run-<model>.sh
```

### 6.4 Ví dụ cấu hình LLM cho cluster 2 node

Khi vận hành trên cụm gồm 2 node Spark/GX10, cấu hình mẫu nên đặt:

- `tp=2`
- `pp` là tham số khác của hệ thống, không dùng để thay thế cho Tensor Parallel

Ví dụ cấu hình dạng YAML:

```yaml
model:
    tp: 2
```

Ghi chú:

- Với cluster 2 node, `tp=2` là giá trị mẫu phù hợp cho Tensor Parallel.
- Nếu recipe hoặc launcher của từng model có thêm tham số riêng, giữ theo đúng schema của hệ thống đó.

---

## 7. Kiến trúc phân bổ tài nguyên (QUAN TRỌNG)

### 7.1 Nguyên tắc

Cluster vLLM **KHÔNG merge VRAM vật lý**, mà:

- mỗi node giữ KV cache riêng
- request được shard qua nhiều node

---

### 7.2 Tính toán cần lưu ý

#### a. Sequence Length

- tính **trên mỗi GPU/node**
- không phải tổng cluster

#### b. Batch Size

- tổng throughput = sum(batch mỗi node)

#### c. Concurrent Users

- scale theo số node

#### d. KV Cache

- nằm **local từng node**
- không shared

---

## 8. Kiểm tra hoạt động cluster

### 8.1 Test inference

```bash
curl http://<node-ip>:8000/v1/completions
```

---

### 8.2 Theo dõi GPU

```bash
nvidia-smi
```

---

### 8.3 Theo dõi network (RDMA)

```bash
ibstat
```

---

## 9. Các lỗi thường gặp

### 9.1 Interface không có IP

```
Interface enp* is Up but has no IP
```

→ fix:

- cấu hình netplan
- hoặc DHCP fail

---

### 9.2 Docker permission denied

→ fix:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

---

### 9.3 Node không thấy nhau

- kiểm tra:
    - ping
    - firewall

```bash
sudo ufw disable
```

### 9.4 Permission denied (cache)

Nếu gặp lỗi quyền với cache của Hugging Face, fix bằng:

```bash
sudo chown -R $USER:$USER ~/.cache/huggingface
```

---

## 10. Khuyến nghị vận hành

- Dùng network riêng cho cluster (không đi chung LAN)
- Ưu tiên:
    - RDMA / InfiniBand

- Đồng bộ:
    - timezone
    - OS version
    - driver NVIDIA

---

## 11. Checklist nhanh

- [ ] OS giống nhau (Ubuntu 24.04)
- [ ] User giống nhau (UID/GID)
- [ ] ConnectX-7 nối đúng slot
- [ ] IP tĩnh hoạt động
- [ ] SSH key login OK
- [ ] Docker chạy OK
- [ ] build-and-copy.sh -c thành công
- [ ] model chạy được

---

Nếu cần, tôi có thể viết thêm phần:

- tuning vLLM (KV cache, tensor parallel, pipeline parallel)
- benchmark cluster GX10 (tokens/sec)
- hoặc thiết kế topology 2 → 8 node chuẩn production

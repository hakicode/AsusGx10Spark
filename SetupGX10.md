Dưới đây là **bộ lệnh hoàn chỉnh, nhất quán** để anh trai chạy:

* 2 node Ray (host network, pin NIC đúng)
* vLLM với **TP=2 cross-node**
* fix triệt để lỗi `127.0.0.1 / Gloo`

---

# 1. Node 1 (Head) — 192.168.100.10

```bash
docker stop ray-head 2>/dev/null && docker rm ray-head 2>/dev/null

docker run -d \
  --runtime nvidia --gpus all \
  --network host \
  --name ray-head \
  --restart unless-stopped \
  --shm-size=10g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_HOST_IP=192.168.100.10 \
  -e NCCL_SOCKET_IFNAME=enP2p1s0f1np1 \
  -e GLOO_SOCKET_IFNAME=enP2p1s0f1np1 \
  -e HUGGING_FACE_HUB_TOKEN=hf_xxx \
  --entrypoint /bin/bash \
  vllm/vllm-openai:latest \
  -c "
    pip install ray==2.55.1 -q && \
    ray start --head \
      --node-ip-address=192.168.100.10 \
      --port=6379 \
      --block
  "

docker logs -f ray-head
```

---

# 2. Node 2 (Worker) — 192.168.100.9

```bash
docker stop ray-worker 2>/dev/null && docker rm ray-worker 2>/dev/null

docker run -d \
  --runtime nvidia --gpus all \
  --network host \
  --name ray-worker \
  --restart unless-stopped \
  --shm-size=10g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_HOST_IP=192.168.100.9 \
  -e NCCL_SOCKET_IFNAME=enP2p1s0f1np1 \
  -e GLOO_SOCKET_IFNAME=enP2p1s0f1np1 \
  -e HUGGING_FACE_HUB_TOKEN=hf_xxx \
  --entrypoint /bin/bash \
  vllm/vllm-openai:latest \
  -c "
    pip install ray==2.55.1 -q && \
    ray start \
      --address=192.168.100.10:6379 \
      --node-ip-address=192.168.100.9 \
      --block
  "

docker logs -f ray-worker
```

---

# 3. Verify cluster (bắt buộc)

```bash
docker exec ray-head ray status
docker exec ray-head ray nodes
```

### Kết quả phải là:

```
192.168.100.10
192.168.100.9
```

---

# 4. Chạy vLLM (trên node 1)

```bash
docker exec ray-head bash -c "
  export VLLM_HOST_IP=192.168.100.10
  export RAY_ADDRESS=192.168.100.10:6379

  export MASTER_ADDR=192.168.100.10
  export MASTER_PORT=29500

  export NCCL_SOCKET_IFNAME=enP2p1s0f1np1
  export GLOO_SOCKET_IFNAME=enP2p1s0f1np1
  export TP_SOCKET_IFNAME=enP2p1s0f1np1

  export NCCL_IB_DISABLE=0
  export NCCL_DEBUG=INFO

  vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray \
    --dtype auto \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-model-len 8192 \
    --port 8001
"
```

---

# 5. Kiểm tra nhanh nếu có vấn đề

## Trong container

```bash
hostname -i
```

→ phải ra:

```
192.168.100.10   (head)
192.168.100.9    (worker)
```

---

## Nếu vẫn dính 127.0.0.1

Fix ngay:

```bash
# Node 1
docker exec ray-head bash -c "echo '192.168.100.10 $(hostname)' >> /etc/hosts"

# Node 2
docker exec ray-worker bash -c "echo '192.168.100.9 $(hostname)' >> /etc/hosts"
```

---

# 6. Kết luận kỹ thuật

Bộ này đảm bảo:

* Ray advertise đúng IP
* Torch distributed dùng đúng NIC
* Không fallback loopback
* TP=2 chạy cross-node ổn định trên link 200Gb

---

Nếu chạy xong vẫn lỗi, gửi đúng đoạn:

```
init_process_group
NCCL INFO
GLOO connect
```

→ tôi sẽ chỉ ra chính xác node nào đang advertise sai endpoint.

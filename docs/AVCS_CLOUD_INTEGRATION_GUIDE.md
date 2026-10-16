# AVCS Cloud Inference & Integration Guide
### *Zero-Leakage Proprietary Visual Cortex Deployment for External Projects*

---

## 1. Executive Summary & Architecture

The **Artificial Visual Cortex System (AVCS)** is a biological System 1 (Eyes + Hands) neural perception engine designed for autonomous desktop and screen interaction. 

To protect proprietary neural weights (`avcs_cortex_weights.pt` and `human_eye_weights.pt`) and prevent model extraction or reverse-engineering, AVCS uses a **Zero-Leakage Cloud Architecture**:

```
+-----------------------------------------------------------------------------------------+
|                                    AVCS CLOUD CLUSTER                                    |
|  (Hosted on AWS / GCP / Azure / DigitalOcean or Private Dedicated CPU VPS Server)        |
|                                                                                         |
|   +---------------------------------------------------------------------------------+   |
|   |                          FastAPI / Uvicorn API Gateway                          |   |
|   |                      [X-API-Key Authentication & Rate Limiter]                  |   |
|   +---------------------------------------------------------------------------------+   |
|                                           |                                             |
|        +----------------------------------+----------------------------------+          |
|        |                                                                     |          |
|   +---------------------------------------+             +---------------------------+   |
|   |         HumanEye Cortical Backbone    |             |    Foveal Acuity Engine   |   |
|   |    - Magnocellular Motion Flow        |             |    - 20x Zoom Inspection  |   |
|   |    - Parvocellular Affordance Heatmap |             |    - Boundary Detection   |   |
|   |    - 2D Spatial NMS (3x3)             |             |    - Contrast Gating      |   |
|   +---------------------------------------+             +---------------------------+   |
|        |                                                                                |
|   +---------------------------------------+             +---------------------------+   |
|   |         Visual Focus Detector         |             |   Servoing Kinematics     |   |
|   |    - 0.8-2.0 Hz Caret Blink Detector  |             |   - 5th-Order Min Jerk    |   |
|   |    - Fail-Closed Security Gating      |             |   - Fitts' Law Duration   |   |
|   +---------------------------------------+             +---------------------------+   |
+-----------------------------------------------------------------------------------------+
                                            ^
                                            |  HTTPS (Zero Weights Downloaded)
                                            |  Encrypted Base64 JPEGs (<50 KB)
                                            |  Normalized (x, y) Actions & Kinematics
                                            v
+-----------------------------------------------------------------------------------------+
|                                 CLIENT: "PROJECT X"                                     |
|                      (External Tech App / User Laptop / VM)                             |
|                                                                                         |
|   +---------------------------------------------------------------------------------+   |
|   |                             AVCSCloudClient SDK                                 |   |
|   |    - NO PyTorch dependency.                                                     |   |
|   |    - ZERO weight files downloaded or stored locally.                            |   |
|   |    - High-speed local screen capture (mss / GDI DIBSection, <10ms).             |   |
|   |    - Local motor execution (Win32 SendInput / Virtual Driver).                  |   |
|   +---------------------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------------------+
```

### Key Guarantees:
1. **Zero Weight Leakage**: Client installations never download `.pt` files. The neural networks execute exclusively in memory inside the cloud container.
2. **Ultra-Lightweight Client**: Client only requires `requests`, `numpy`, `pillow`, and `mss`. It does **not** need PyTorch or a GPU.
3. **High-Speed Network Protocol**: Frames are compressed to high-quality base64 JPEGs (~40-60 KB). Payload transit latency is $<15$ ms over standard broadband.
4. **Single-Thread CPU Inference**: The cloud model runs on standard CPU instances (2-4 vCPUs) with $\approx 10$ ms forward pass latency.
5. **Fail-Closed Safety**: Typing actions are verified for visual caret blink before keystrokes are dispatched, preventing erratic input leaks.

---

## 2. Server Deployment (Cloud Setup)

### Option A: 1-Click Docker Deployment (Recommended)

1. **Clone the repository on your Cloud VPS / Server**:
   ```bash
   git clone https://github.com/your-org/forge.git
   cd forge
   ```

2. **Configure your private service credential**:
   ```bash
   export AVCS_INFERENCE_API_KEY="$(openssl rand -hex 32)"
   ```
   Keep this value server-side only; the AVCS platform's Next.js server must use
   the same value in `AVCS_INFERENCE_API_KEY`.

3. **Build and launch the container**:
   ```bash
   docker compose -f avcs/server/docker-compose.yml up -d --build
   ```

4. **Verify Health**:
   ```bash
   curl -H "X-API-Key: ${AVCS_INFERENCE_API_KEY}" http://127.0.0.1:8000/v1/health
   ```
   *Expected Response:*
   ```json
   {
     "status": "healthy",
     "service": "AVCS Cloud Neural Inference",
     "weights_loaded": true,
     "fovea_weights_loaded": true,
     "device": "cpu"
   }
   ```

---

### Option B: Direct Python / Systemd on Ubuntu / Debian VPS

1. **Install Python dependencies on the server**:
   ```bash
   sudo apt-get update && sudo apt-get install -y python3-pip
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install fastapi uvicorn pydantic numpy pillow requests
   ```

2. **Start the AVCS Cloud Service**:
   ```bash
   export AVCS_INFERENCE_API_KEY="$(openssl rand -hex 32)"
   export OMP_NUM_THREADS=1
   python3 -m uvicorn avcs.server.api:app --host 127.0.0.1 --port 8000 --workers 2
   ```

3. **Nginx Reverse Proxy with HTTPS (Production)**:
   ```nginx
   server {
       server_name avcs-api.yourdomain.com;

       client_max_body_size 10M;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```
   Obtain free SSL using Certbot:
   ```bash
   sudo certbot --nginx -d avcs-api.yourdomain.com
   ```

---

## 3. Client Integration in "Project X"

### Step 1: Install Client Dependencies
In your external project's virtual environment, run:
```bash
pip install requests pillow numpy mss
```
*(Notice: No PyTorch or CUDA required!)*

### Step 2: Include the Client SDK
Copy `avcs/server/client.py` into your project directory (e.g. `project_x/avcs_client.py`).

### Step 3: Quickstart Integration Example

```python
"""example_project_x.py - Using AVCS Cloud from an external app."""
import time
import os
from avcs.server.client import AVCSCloudClient

# 1. Initialize the Cloud Client
client = AVCSCloudClient(
    endpoint_url="https://avcs-api.yourdomain.com",  # Or http://127.0.0.1:8000 for local dev
    api_key=os.environ["AVCS_INFERENCE_API_KEY"],
    force_virtual_motor=False,                       # Set True to simulate without real mouse clicks
)

# 2. Ping Cloud Server
status = client.ping()
print(f"[+] AVCS Cloud Connected: {status['status']} (Weights Loaded: {status['weights_loaded']})")

# 3. Perceive the Current Desktop Screen
state = client.see()
print(f"[+] Primary Scene: {state['category']} (Confidence: {state['category_confidence']:.2f})")
print(f"[+] Stability State: {state['stability']}, Surprise: {state['surprise_score']}")
print(f"[+] Cloud Processing Latency: {state['latency_ms']} ms")

# 4. Extract Clickable UI Elements (Affordances)
clickables = client.get_affordances(action_type="clickable")
print(f"[+] Discovered {len(clickables)} interactive buttons/targets.")

for i, (x, y, kind, score) in enumerate(clickables):
    print(f"    - Target #{i+1}: kind={kind}, pos=({x:.3f}, {y:.3f}), confidence={score:.2f}")

# 5. High-Acuity Foveation (Zoomed Inspection)
if clickables:
    target_pos = clickables[0][:2]
    fovea = client.foveate(target=target_pos)
    print(f"[+] Foveal Acuity on target: element={fovea['element_kind']}, bbox={fovea['bbox']}")

    # 6. Smooth Minimum-Jerk Mouse Movement & Click
    print(f"[+] Moving cursor to target with human-like kinematics...")
    client.click(target=target_pos, button="left")

# 7. Safe Fail-Closed Typing
# AVCS will verify caret blinking before dispatching keystrokes.
typeables = client.get_affordances(action_type="typeable")
if typeables:
    input_box = typeables[0][:2]
    print(f"[+] Attempting to type into input box at {input_box}...")
    try:
        client.type_into(
            target=input_box,
            text="Hello from Project X powered by AVCS Cloud!",
            enforce_focus=True,  # Fail-closed security guard
        )
        print("[+] Keystrokes safely delivered!")
    except RuntimeError as err:
        print(f"[-] Blocked by fail-closed focus gate: {err}")

# Clean up
client.close()
```

---

## 4. REST API Specification

### Endpoint 1: Health & Diagnostics
- **Method**: `GET /v1/health`
- **Headers**: None required.
- **Response**:
  ```json
  {
    "status": "healthy",
    "service": "AVCS Cloud Neural Inference",
    "weights_loaded": true,
    "fovea_weights_loaded": true,
    "device": "cpu"
  }
  ```

---

### Endpoint 2: Full Screen Perception
- **Method**: `POST /v1/perceive`
- **Headers**: `X-API-Key: <your_key>`
- **Request Body**:
  ```json
  {
    "image_b64": "<base64_encoded_jpeg>",
    "action_type": "all",
    "top_k": 5
  }
  ```
- **Response**:
  ```json
  {
    "category": "desktop",
    "category_confidence": 0.88,
    "stability": "stable",
    "surprise_score": 0.042,
    "latency_ms": 11.45,
    "affordances": [
      [0.452, 0.312, "clickable", 0.94],
      [0.610, 0.725, "typeable", 0.88]
    ]
  }
  ```

---

### Endpoint 3: Foveal Acuity Refinement
- **Method**: `POST /v1/foveate`
- **Headers**: `X-API-Key: <your_key>`
- **Request Body**:
  ```json
  {
    "image_b64": "<base64_encoded_jpeg>",
    "target": [0.452, 0.312]
  }
  ```
- **Response**:
  ```json
  {
    "element_kind": "button",
    "confidence": 0.91,
    "contrast_score": 0.082,
    "bbox": [0.42, 0.30, 0.48, 0.33],
    "latency_ms": 2.15
  }
  ```

---

### Endpoint 4: Fail-Closed Focus Verification
- **Method**: `POST /v1/verify_focus`
- **Headers**: `X-API-Key: <your_key>`
- **Request Body**:
  ```json
  {
    "current_image_b64": "<base64_jpeg_t1>",
    "reference_image_b64": "<base64_jpeg_t0>",
    "target": [0.610, 0.725]
  }
  ```
- **Response**:
  ```json
  {
    "is_focused": true,
    "confidence": 0.84,
    "caret_detected": true,
    "caret_blinking": true
  }
  ```

---

### Endpoint 5: Kinematic Trajectory Planning
- **Method**: `POST /v1/plan_trajectory`
- **Headers**: `X-API-Key: <your_key>`
- **Request Body**:
  ```json
  {
    "start": [0.1, 0.1],
    "target": [0.8, 0.8],
    "duration_s": 0.45
  }
  ```
- **Response**:
  ```json
  {
    "duration_s": 0.45,
    "waypoints_count": 55,
    "waypoints": [
      {"timestamp": 0.0, "x": 0.1, "y": 0.1, "vx": 0.0, "vy": 0.0},
      {"timestamp": 0.008, "x": 0.102, "y": 0.102, "vx": 0.04, "vy": 0.04}
    ]
  }
  ```

---

## 5. Security & Intellectual Property Safeguards

| Threat Vector | Cloud-Only Defense Mechanism |
|---|---|
| **Weight Scraping / Exfiltration** | Weights exist exclusively on the server in memory. Client never receives `.pt` or binary blobs. |
| **Reverse Engineering** | Client only receives processed coordinates `(x, y)` and bounding boxes, not activation tensors. |
| **Unauthorized Usage** | Mandatory `X-API-Key` validation on all inference endpoints. |
| **Accidental Keystroke Spills** | `/v1/verify_focus` requires visual caret blink (0.8-2.0 Hz) before `type_into()` proceeds. |
| **Runaway Bot Movements** | OS cursor breakout failsafe halts execution if the user bumps the mouse to screen corner `(0, 0)`. |

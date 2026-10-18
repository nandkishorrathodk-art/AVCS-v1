# AVCS Cloud Inference & Integration Guide
### *Remote inference integration for external projects*

---

## System 1 → System 2 privacy boundary

For the intended architecture, run capture, perception, grounding, and motor
actions locally. Send only a minimized, allowlisted text observation and the
user's text task to System 2 with `TextOnlySystem2Client`:

```python
from avcs.server import TextOnlySystem2Client

brain = TextOnlySystem2Client(
    endpoint_url="https://your-system2.example",
    api_key="your_system2_key",
)
turn = controller.consult_system2(
    task,
    brain,
    consent_confirmed=user_has_consented,
)
goal = turn.goal
```

The client posts to `/v1/system2/goal` and accepts a text `goal` response. It
does not capture, encode, or attach images or embeddings. The remote System 2
service must implement that endpoint. Text can still contain sensitive data;
the client masks some common secret patterns, but this is not comprehensive
redaction. Minimize observations locally and obtain explicit user consent
before sharing. `consult_system2()` returns the goal for review and never
dispatches mouse or keyboard input. Free-form goal text is not an executable
action protocol.

For live desktop acceptance, construct `AVCSController(require_real_capture=True)`
and use `start_streaming()` / `stop_streaming()`. This prevents synthetic
fallback, but does not guarantee a particular frame rate. Record
`controller.capture_backend` and `controller.stream_metrics` from the target
machine; do not label the configured target rate as measured performance.

`AVCSCloudClient` and the visual-inference endpoints documented below are a
separate legacy mode that sends screen frames to a remote visual inference
service. Do not use that mode for the strict zero-pixel System 1/System 2
architecture.

---

## 1. Executive Summary & Architecture

AVCS provides a remote inference API and a client that can capture frames locally and send them to a configured server. This is a cloud deployment option, not a zero-leakage or local-only architecture: screenshots leave the client device and are processed by the server.

The following diagram describes the client/server data path:

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
                                            |  HTTPS for remote endpoints
                                            |  Screen frames sent to the server
                                            |  Normalized (x, y) Actions & Kinematics
                                            v
+-----------------------------------------------------------------------------------------+
|                                 CLIENT: "PROJECT X"                                     |
|                      (External Tech App / User Laptop / VM)                             |
|                                                                                         |
|   +---------------------------------------------------------------------------------+   |
|   |                             AVCSCloudClient SDK                                 |   |
|   |    - Does not download model weights during inference.                          |   |
|   |    - Sends captured screen frames to the configured inference server.           |   |
|   |    - Uses the project's declared runtime dependencies.                          |   |
|   |    - Local motor execution (Win32 SendInput / Virtual Driver).                  |   |
|   +---------------------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------------------+
```

### Deployment facts and limits:
1. The client sends captured screen frames to the configured server. HTTPS protects the network transport; it does not prevent the server from receiving or processing those frames.
2. Non-loopback HTTP endpoints are rejected by default. `allow_insecure_http=True` is an explicit opt-in for trusted test networks only; do not use it for production credentials or screen data.
3. Model weights are not downloaded by the client as part of inference. This alone does not establish that the client has no model/runtime dependencies; use the package metadata for installation requirements.
4. Frame size, network latency, inference time and total action latency depend on the image, hardware, network and server load. Measure them on the deployment environment; this guide makes no fixed latency or FPS guarantee.
5. `verify_focus()` requires a boolean response. The client blocks typing when focus is false or malformed, but this visual check is not a compliance certification or proof of a completed business action.
6. `/v1/ground` returns a candidate point, not authorization to click or proof that the named target exists. Validate the application outcome independently and keep high-impact actions behind human approval.

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
Install AVCS from a repository checkout so the client and its internal capture and
motor modules are present:
```bash
pip install ".[client,desktop]"
```

Do not copy only `avcs/server/client.py`; it imports other AVCS modules. The base
package currently declares PyTorch, NumPy and Pillow. The `client` extra adds
Requests; `desktop` adds optional screen capture support.

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

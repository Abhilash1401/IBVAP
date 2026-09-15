# IBVAP V1 — Intelligent Border Video Analytics Platform

**AI-Based Intelligent Video Analytics Platform for Border Surveillance Using Existing CCTV Infrastructure**

IBVAP V1 is a local/offline video analytics platform designed around a lightweight two-stage surveillance pipeline. Instead of running object detection continuously on every video frame, the system first uses a lightweight motion-detection gate and invokes the more expensive local YOLO detector only when meaningful motion is detected.

The project is intentionally modular: video ingestion, motion gating, orchestration, object detection, event persistence, alerts, API delivery, and the operator dashboard are separated so that each component can evolve without coupling the entire system.

> **V1 scope:** motion detection + human/vehicle detection + event persistence + event clips + REST/SSE + operator dashboard. Advanced capabilities such as face recognition, ANPR/OCR, weapon detection, multi-camera tracking, adaptive exclusion zones, and cloud AI are outside V1.

---

## 1. System Architecture

```text
             Local Video / USB Camera / RTSP CCTV
                           |
                           v
                 +----------------------+
                 | Video Integration    |
                 | Source + Reader      |
                 +----------+-----------+
                            |
                       FramePacket
                            |
                            v
                 +----------------------+
                 | Engine 1              |
                 | Lightweight Motion    |
                 | Detection / Gate      |
                 +----------+-----------+
                            |
                    Motion detected?
                      /            \
                    NO              YES
                    |                |
                 Suppress      MotionEvent /
                               ClassificationRequest
                                     |
                                     v
                           +----------------+
                           | Core            |
                           | Orchestration   |
                           +-------+--------+
                                   |
                                   v
                           +----------------+
                           | Engine 2       |
                           | Local YOLO     |
                           | Human/Vehicle  |
                           +-------+--------+
                                   |
                         ClassificationResult
                                   |
                    +--------------+--------------+
                    |                             |
                    v                             v
              Event Store                    Clip Writer
                SQLite                    pre + event + post
                    |                             |
                    +--------------+--------------+
                                   |
                                   v
                           +----------------+
                           | FastAPI        |
                           | REST + SSE     |
                           +-------+--------+
                                   |
                                   v
                           React / TypeScript
                              Operator UI
```

### Architectural rule

Core owns orchestration and routing. Components should communicate through the defined contracts and orchestration layer rather than creating direct Engine 1 → Engine 2 coupling.

The main V1 flow is therefore:

```text
Video source
  -> FramePacket
  -> Engine 1 motion gate
  -> Core
  -> Engine 2 YOLO
  -> ClassificationResult
  -> Event/Alert/Clip persistence
  -> FastAPI REST/SSE
  -> Frontend
```

---

## 2. Why Two Processing Stages?

Continuous object detection on every frame is unnecessary for a perimeter scene that may remain static for long periods.

IBVAP first asks:

> **Is there meaningful visual motion in this frame?**

If not, the frame is suppressed from expensive object detection. If motion is detected, the relevant frame/request is routed through Core to Engine 2.

```text
Every incoming frame
        |
        v
  Lightweight Engine 1
        |
   +----+----+
   |         |
No motion  Motion
   |         |
Suppress     v
         Local YOLO
             |
       Human / Vehicle
```

This is the principal V1 technical concept. Any workload-reduction or latency metric shown by the UI should be calculated from real runtime data; the project must not present fabricated benchmark numbers.

---

## 3. V1 Components

### Video Integration

Responsible for:

- Local video files (`.mp4`, `.avi`, `.mov`, etc.)
- Local-network RTSP/IP camera sources
- USB/laptop webcam support in the pipeline runner
- Frame acquisition
- Source metadata and status
- End-of-stream/error handling
- Resource cleanup
- Standardized `FramePacket` creation
- Rolling frame buffer support

Processing frames remain in memory. Frames are not permanently stored individually as the normal operating model.

### Engine 1 — Motion Detection

Engine 1 is a lightweight motion gate. It determines whether a scene contains meaningful motion and exposes motion-related information such as score and threshold/region metadata.

It does **not** perform human/vehicle recognition.

### Core — Orchestration

Core manages plugin/component lifecycle and routing. The intended lifecycle is:

```text
DISCOVER -> LOAD -> INITIALIZE -> READY -> PROCESSING -> STOP -> SHUTDOWN
```

Common plugin operations are:

```text
initialize(config)
process(input)
get_status()
shutdown()
```

### Engine 2 — Local YOLO

Engine 2 performs local object detection using the project model weights (`yolov8n.pt`). Its V1 semantic targets are:

- `HUMAN`
- `VEHICLE`
- `NON_TARGET`

The detector returns bounding boxes, confidence, and classification metadata.

**Important:** standard COCO YOLO does not provide a `tree` class. The project must not claim unsupported classes simply for presentation purposes.

### Event Store

Events and classification information are persisted in SQLite. The API exposes event history and event details to the dashboard.

### Clip Writer

The rolling buffer provides pre-event frames. When an event occurs, the system combines pre-event context with event/post-event frames to create a short local event clip.

```text
Pre-event buffer + Trigger/event + Post-event frames
                         |
                         v
                    Event clip
```

The backend creates clips; the frontend plays them.

### Alert Manager

Provides alert/notification handling and cooldown/deduplication behavior where configured.

### FastAPI

The API exposes current system state, detections, events, alerts, notifications, snapshots, clips, and an SSE event stream.

### Frontend

The target UI architecture is React + TypeScript + Vite. An earlier static dashboard prototype may exist under `frontend/dist`; it should be treated as a visual/functional reference rather than the final maintainable frontend architecture.

---

## 4. Repository Structure

```text
ibvap/
├── backend/
│   ├── alerts/
│   │   └── manager.py
│   ├── api/
│   │   └── server.py
│   ├── clips/
│   │   └── writer.py
│   ├── contracts/
│   │   ├── classification.py
│   │   ├── classification_result.py
│   │   ├── frame.py
│   │   └── motion.py
│   ├── core/
│   │   ├── orchestrator.py
│   │   └── plugin.py
│   ├── engine1/
│   │   └── motion_detector.py
│   ├── engine2/
│   │   ├── detector.py
│   │   └── MODEL_CARD.md
│   ├── events/
│   │   └── store.py
│   └── video/
│       ├── buffer.py
│       ├── exceptions.py
│       ├── file_source.py
│       ├── reader.py
│       ├── rtsp_source.py
│       └── source.py
├── examples/
│   └── run_demo.py
├── frontend/
├── storage/
│   └── event_clips/
├── tests/
├── pyproject.toml
├── requirements.txt
├── run.py
├── run_pipeline.py
├── serve.py
├── yolov8n.pt
└── README.md
```

---

## 5. Data Contracts

The backend uses explicit contracts to keep components decoupled.

### `FramePacket`

Represents an acquired video frame and its source metadata. The contract includes fields such as:

- `frame_id`
- `source_id`
- `timestamp`
- `frame` (`numpy.ndarray`)
- frame width/height
- sequence number

### `MotionEvent`

Represents the output of the motion-gating stage, including the source/frame relationship and motion information.

### `ClassificationRequest`

Carries the relevant frame/request information from the orchestration flow into Engine 2.

### `ClassificationResult`

Carries the Engine 2 result, including detected objects, confidence, bounding boxes, and associated metadata.

The exact Python models in `backend/contracts/` are the source of truth for field definitions.

---

## 6. API Surface

The current FastAPI application defines endpoints including:

```text
GET  /health
GET  /stats
GET  /status
GET  /detections/latest
GET  /detections/recent
GET  /alerts
GET  /alerts/stats
GET  /events
GET  /events/{event_id}
GET  /events/{event_id}/clip
GET  /notifications
POST /notifications/{alert_id}/ack
GET  /snapshot.jpg
GET  /events/stream
```

The exact request/response models and runtime behavior are defined by `backend/api/server.py` and the associated backend components. Do not create duplicate frontend-side versions of these contracts when the backend already provides the required data.

### SSE

`/events/stream` provides the live event channel used by the operator interface where applicable.

---

## 7. Running IBVAP

Create/activate a Python virtual environment and install dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Webcam / local camera

```powershell
python run_pipeline.py --camera --engine2 --display --api --api-port 8000
```

Then open:

```text
http://localhost:8000
```

or:

```text
http://127.0.0.1:8000
```

### Local video

```powershell
python run_pipeline.py --file path/to/video.mp4 --engine2 --display --api --api-port 8000
```

### RTSP CCTV/IP camera

```powershell
python run_pipeline.py --rtsp "rtsp://<camera-address>/<stream>" --engine2 --display --api --api-port 8000
```

### Synthetic demo

```powershell
python run_pipeline.py --demo --engine2 --display --api --api-port 8000
```

### Useful Engine 2 options

```text
--engine2-model <path>
--engine2-conf <threshold>
--engine2-imgsz <size>
```

The current runner also supports configuration for alert cooldown, event database, clip directory, pre-event frames, post-event frames, and live snapshot frequency. Use:

```powershell
python run_pipeline.py --help
```

to inspect the current CLI contract before deployment.

---

## 8. Testing

The reviewed repository test suite has previously been executed successfully:

```text
51 passed
```

Run the suite with:

```powershell
python -m pytest -q
```

The tests cover areas including:

- contracts
- video buffer
- file source
- RTSP source
- video reader
- Engine 1 motion detection
- Engine 2 detection
- Core orchestration
- alerts/API

Tests should be rerun after backend changes.

---

## 9. Frontend / Operator Experience

The final operator application is planned around these routes:

```text
/dashboard   Dashboard
/live        Live Monitor
/events      Events
/cameras     Cameras
/analytics   Analytics
/system      System Status
/settings    Settings
```

### Dashboard

The Dashboard is the high-level operational overview. It can show:

- cameras online
- motion events
- human detections
- vehicle detections
- total events
- live preview
- recent events
- system/plugin status

The Dashboard must distinguish real backend telemetry from static presentation values.

### Live Monitor

The Live Monitor has a deliberate two-state interaction model.

**State 1 — Camera Grid**

Opening `/live` shows multiple camera cards first. A camera is not automatically expanded.

**State 2 — Selected Camera Detail**

Clicking a camera card selects that camera and opens the detailed monitoring interface. The detail view can contain:

- large live preview
- detection overlays
- camera information
- controls
- live events
- detection statistics
- stream health

A required:

```text
<- Back to Camera Grid
```

control returns to the multiple-camera overview.

The frontend selection state should conceptually behave as:

```text
selectedCamera = null
        -> CameraGrid

selectedCamera = CAM_01
        -> CameraDetail(CAM_01)

Back to Camera Grid
        -> selectedCamera = null
```

The Back action must not stop the backend inference pipeline.

**Pause**, when implemented, pauses only the frontend display/preview. It must not terminate backend ingestion or inference.

---

## 10. Locked UI Design System

IBVAP uses a **Dark Bento + Neumorphic hybrid** visual language intended to look like a professional border-surveillance command console.

### Color tokens

| Token | Value | Purpose |
|---|---|---|
| App background | `#061525` | Main background |
| Deep surface | `#081C2E` | Sidebar/deep surface |
| Primary card | `#0B2540` | Main cards |
| Secondary card | `#0E2D4A` | Secondary surfaces |
| Primary cyan | `#00B8FF` | Primary action/accent |
| Aqua | `#16D6E8` | Highlight |
| Blue action | `#1683FF` | Action state |
| Healthy | `#16E09A` | Online/healthy |
| Critical | `#FF3B55` | Alerts/errors |
| Motion | `#FF9D2E` | Motion/warning |
| Vehicle | `#A76BFF` | Vehicle category |
| Primary text | `#F4FAFF` | Main text |
| Secondary text | `#B7D1E8` | Secondary text |
| Muted text | `#7897B5` | Supporting text |
| Border | `#163B59` | Panel borders |

Preferred shadows:

```text
rgba(0,0,0,0.28)
rgba(0,184,255,0.08)
```

Avoid unrelated palettes, glassmorphism, excessive gradients, and decorative neon effects that reduce operational readability.

---

## 11. V1 Scope Boundaries

The following are explicitly **out of V1**:

- Face recognition
- ANPR/OCR
- Weapon detection
- Drone detection
- Re-identification (Re-ID)
- Multi-camera tracking
- Advanced behavior recognition
- Cloud AI
- Kafka/MQTT/Redis/RabbitMQ
- Unnecessary microservices
- Adaptive exclusion zones
- Engine 2 → Engine 1 feedback learning
- Tree/vegetation learning

### V2 adaptive exclusion concept

A future architecture may introduce:

```text
Engine 2
   -> FeedbackEvent
   -> Core / Feedback Router
   -> Engine 1
   -> update exclusion zone
```

This must not be silently implemented as part of V1.

---

## 12. Engineering Principles

### Backend is the source of truth

Frontend values should come from actual API/SSE state whenever that information exists.

### No fabricated telemetry

Do not present hard-coded values as measured performance. Metrics such as latency, FPS, bitrate, frame drops, workload reduction, or uptime must be measured or supplied by the backend before being represented as live telemetry.

### No duplicate pipelines

The frontend must never run a second inference pipeline. React displays and controls the existing backend system.

### Small compatible changes

When modifying the project:

1. Inspect existing implementation.
2. Identify the exact missing behavior.
3. Reuse existing contracts/APIs.
4. Make the smallest compatible change.
5. Run tests.
6. Verify the actual runtime behavior.

### Preserve working components

The existing backend is a foundation, not a disposable prototype. Avoid unnecessary rewrites of Video Integration, Engine 1, Engine 2, Core, events, clips, or API components.

---

## 13. Known Development Considerations

These are items to verify during continued development rather than assumptions that they are already solved:

1. The pipeline runner currently uses localhost binding for its API server. LAN deployment should use a configurable API host rather than permanently exposing the service.
2. CORS should eventually be configurable/restricted for deployment rather than remaining universally permissive.
3. The current live-preview mechanism may use periodically refreshed JPEG snapshots rather than a browser-native continuous RTSP stream. The UI must describe the actual implementation accurately.
4. The event store should contain a single unambiguous implementation of each method; remove duplicate definitions if encountered.
5. The original static dashboard prototype should be treated as reference material while the long-term frontend becomes a proper React/TypeScript/Vite source tree.
6. A multi-camera visual grid does not automatically imply simultaneous multi-camera inference. Do not introduce an unplanned distributed multi-camera processing architecture merely to populate UI cards.

---

## 14. GitHub Review Guidance

Before pushing the project, verify that the repository contains:

- source code
- tests
- model documentation
- README
- dependency configuration
- appropriate `.gitignore`
- no secrets/API keys/passwords
- no unnecessary virtual environments
- no generated Python cache files
- no accidental local databases or temporary event clips unless intentionally included

The GitHub repository should communicate the actual state of the project. Do not describe planned features as completed features.

A reviewer should be able to understand from this README:

```text
What IBVAP is
      ↓
Why the two-stage pipeline exists
      ↓
How video enters the system
      ↓
How Engine 1 gates expensive inference
      ↓
How Core routes processing
      ↓
How Engine 2 produces detections
      ↓
How events/clips are persisted
      ↓
How FastAPI exposes the system
      ↓
How the frontend consumes the system
      ↓
What is V1 and what is not V1
```

---

## 15. Project Status

The backend foundation is substantially implemented and currently testable. The latest local verification of the reviewed project produced:

```text
51 tests passed
```

The principal remaining development area is the production-quality React/TypeScript/Vite operator frontend and its final end-to-end validation against the existing backend contracts.

This status is an implementation assessment, not a claimed benchmark or completion percentage.

---

## 16. Recommended Development Order

```text
1. Preserve and validate backend foundation
          ↓
2. Establish proper React + TypeScript + Vite frontend
          ↓
3. Dashboard
          ↓
4. Live Monitor
      Camera Grid
          ↓ click
      Camera Detail
          ↓ Back
      Camera Grid
          ↓
5. Events
          ↓
6. Cameras
          ↓
7. Analytics
          ↓
8. System Status
          ↓
9. Settings
          ↓
10. End-to-end validation
          ↓
11. GitHub review / cleanup
```

---

## License

No license has been specified in the current project metadata. Add an appropriate license before publishing the repository publicly if the project will be distributed outside the team.

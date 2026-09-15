# Engine 2 Model Card — IBVAP V1

## Model Specification

| Field                   | Value                                                        |
|-------------------------|--------------------------------------------------------------|
| **Model Name**          | YOLOv8n (Nano)                                               |
| **Model Version**       | Ultralytics YOLOv8                                           |
| **Weights File**        | `yolov8n.pt`                                                 |
| **Training Dataset**    | COCO 2017 (80 classes)                                       |
| **Input Resolution**    | 640×640 (configurable)                                       |
| **Confidence Threshold**| 0.40 (configurable via `Engine2Config.confidence_threshold`)  |
| **Hardware Assumptions** | CPU (Intel/AMD x86_64); GPU (CUDA) used automatically if available |

## IBVAP V1 Class Mapping

Engine 2 maps COCO model classes to three IBVAP V1 categories:

| IBVAP Class   | COCO Class IDs | COCO Class Names                             |
|---------------|----------------|----------------------------------------------|
| **HUMAN**     | 0              | person                                       |
| **VEHICLE**   | 1, 2, 3, 5, 7 | bicycle, car, motorcycle, bus, truck         |
| **NON_TARGET**| all others     | Not used for V1 alerting                     |

> **Note**: Standard COCO-trained YOLO models do not provide specialized classes
> such as TREE, DRONE, or WEAPON. V1 does not require these.

## Performance Notes

- Actual inference times are recorded per-frame in `ClassificationResult.processing_time_ms`.
- FPS and CPU/GPU utilization are logged where measured; no synthetic benchmarks are claimed.
- YOLOv8n is optimized for speed over accuracy. For higher accuracy at the cost of
  latency, consider `yolov8s.pt` (small) or `yolov8m.pt` (medium).

## V2 Boundary

V2 may introduce:
- Nuisance-object classification (animals, debris)
- Feedback loop to Engine 1
- Adaptive exclusion zones
- Specialized detectors (face, ANPR)

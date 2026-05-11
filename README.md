# GroundTruth — Mastering Delays: Fusing Predictive Analytics with Real-Time Vision

**Team:** Hackstreet Boys | **Partner:** Swissport

A closed-loop flight delay management system that connects pre-arrival intelligence with real-time ground control. Two pipelines work in tandem: Pipeline 1 flags high-risk flights before they land; Pipeline 2 confirms and tracks delays as they unfold on the apron.

---

## Pipeline 1 — Delay Prediction Based on Historic Data

Historical flight data is ingested into a **Neo4j Knowledge Graph** to map structural dependencies and identify specific delay risks. Graph features are then fed into a **Gradient Boosting** model that predicts arrival delays in exact minutes before the aircraft lands, producing a precise **Flight On-Time Classification**.

**Strategic Relevance:** Shifts ground handling from reactive to proactive. Early predictions allow dispatchers to optimize resources for the upcoming turnaround and ensure critical processes are prioritized before the aircraft lands.

---

## Pipeline 2 — Delay Prediction with Live Feed

Live camera feeds are processed using the **YOLO Computer Vision** algorithm to detect specific ground handling events. Detected timestamps are cross-referenced in real-time against target durations defined in the **BPMN model** (Camunda). The deviation between the *Target* (BPMN) and *Actual* (Video) states continuously updates the delay classification while the aircraft is still on the ground.

**Strategic Relevance:** Bridges the gap between planning and reality. Monitoring live video against the BPMN standard reveals bottlenecks the moment they occur, enabling immediate corrective action and preventing small operational slips from cascading into major departure delays.

---

## Strategic Impact

| | |
|---|---|
| **Value** | Creates a closed-loop system connecting pre-arrival intelligence with real-time ground control. Pipeline 1 acts as the strategic "Early Warning System"; Pipeline 2 actively monitors the critical path defined in the BPMN model. |
| **Results** | Transforms estimation into certainty. Pipeline 1 predicts risks from history; Pipeline 2 confirms them via live video. Cross-referencing both inputs pinpoints exactly when and why delays occur — turning probabilities into facts. |

---

## Tech Stack

| Tool | Role |
|---|---|
| R | Data Preprocessing |
| Neo4j | Knowledge Graph |
| Gradient Boosting | Delay Prediction |
| LangChain | LLM Framework |
| Camunda | BPMN Modeling |
| YOLO | Computer Vision |
| PyCharm | Python IDE |
| Google Vertex AI | AI Platform |

---

## Repository Structure

```
Object_detection/
├── turnaround clip.mp4   # Source apron camera footage
├── turnaround.py         # Frame extraction & YOLO dataset preparation
├── train_model.py        # YOLO11 model training (Pipeline 2)
├── yolo11n.pt            # YOLO11 nano pretrained weights
└── runs/detect/train/
    └── weights/
        ├── best.pt       # Best trained checkpoint
        └── last.pt       # Latest checkpoint
```

## Detected Classes (Pipeline 2)

| Class | Description |
|---|---|
| `bridge_connected` | Jet bridge connected to aircraft |
| `cleaning_crew_vehicle` | Cleaning crew vehicle on apron |
| `luggage_vehicle` | Baggage/luggage handling vehicle |

## Quickstart

```bash
# Step 1 — Extract frames from apron footage and build the YOLO dataset
cd Object_detection
python turnaround.py

# Annotate frames in dataset/images/ with a tool like Label Studio,
# then export YOLO-format labels to dataset/labels/

# Step 2 — Train the YOLO model (GPU recommended)
python train_model.py
```

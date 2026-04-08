# YOLO Object Detection Accelerator (SystemC)

## Overview
This project implements a pipelined YOLO-inspired object detection accelerator using SystemC.  
The design models a processing pipeline with real-time streaming input, 2D convolution, bounding box prediction, and performance evaluation.

---

## Features
 - Pipelined architecture (Convolution → ReLU → Detection)
 - Real 2D convolution using a sharpening kernel with zero-padding
 - YOLO grid-based detection — image divided into 3×3 grid of cells
 - Cell-relative bounding box — each detection corresponds to an entire grid cell, not an exact pixel location 
 - YOLO dual output — objectness flag (object present?) + confidence score (how sure?)
 - Hardware-friendly fixed-point confidence score (no floating point)
 - Gaussian-like object blob — more realistic than a single bright pixel
 - Randomised object position and background noise — different every run
 - Reset-aware pipeline — all modules hold outputs at 0 during reset
 - Performance metrics: latency, throughput, tracking error

---

## System Architecture

```
InputGen → Conv (2D) → ReLU → Detect
              ↑                   ↑
          Sharpening          3-cycle pipeline
           kernel             alignment registers
```

---

## Modules

### InputGen
- Generates a 6×6 image frame every clock cycle
- Object is a Gaussian-like blob: peak = `20`, inner ring = `14`, outer ring = `8`, background = `0–4`
- Object centre randomised each frame via `rand() % IMG_SIZE`
- Holds all outputs at zero during reset

### Conv
- Applies a real 2D convolution using a 3×3 sharpening kernel with zero-padding at borders
- Amplifies the high-value object blob relative to background noise
  
### ReLU
- Element-wise activation: clamps all negative convolution outputs to zero

### Detect

- Divides the 6×6 ReLU output into a **3×3 grid of cells** (2px each)
- Computes the **maximum activation per cell** — the cell with the highest max is the detection
- This is the core YOLO detection mechanism: each grid cell predicts whether an object is present
- **Objectness output:** binary flag — is the best cell activation above `OBJECTNESS_THRESHOLD = 10`?
- **Confidence output:** how dominant is the best cell vs the runner-up?
- **Bounding box:** pixel coordinates of the detected grid cell — `(cell_x × CELL_SIZE, cell_y × CELL_SIZE, CELL_SIZE, CELL_SIZE)`
- 3-stage pipeline shift register aligns ground-truth position with detection output

---

## Detection Output

For each frame (after 3-cycle pipeline warm-up):
```
Actual Object (aligned) : (x, y)
Detected Grid Cell      : (gi, gj)  [cell size = 2px]
Bounding Box (pixels)   : (x, y, 2, 2)
Objectness              : YES — object detected
Confidence              : XX.YY%
Tracking Error          : N px
```

---

## YOLO Concepts Demonstrated
 
| YOLO Concept | Implementation |
|---|---|
| Grid-based detection | 3×3 grid over 6×6 image; each cell predicts independently |
| Cell responsibility | Object assigned to the cell containing its centre pixel |
| Bounding box prediction | bbox = cell origin + cell size in pixel coordinates |
| Objectness score | Binary threshold on best cell activation |
| Confidence score | Fixed-point ratio of best cell to runner-up cell |
| Feature extraction | Conv (sharpening kernel) + ReLU activation |
| Pipelined inference | 3-stage hardware pipeline with alignment registers |
 
---

## Confidence Score
 
Computed using integer fixed-point arithmetic — no floats, fully synthesisable:
```
conf_fp = (best_cell_val - second_cell_val) * 10000 / best_cell_val
```
---

## Performance Metrics

| Metric | Value |
|---|---|
| Clock period | 10 ns |
| Pipeline depth | 3 stages |
| Latency | 3 cycles (30 ns) |
| Throughput | 100 MHz (1 frame/cycle after fill) |

---

## Tools and Technologies
- SystemC 2.3.3
- G++ with SystemC libraries
- Hardware modelling concepts (pipelining, fixed-point arithmetic, reset logic)

---

## Key Concepts Demonstrated
- YOLO grid-based object detection
- Hardware pipelining and pipeline alignment
- Real 2D convolution in hardware
- Fixed-point arithmetic for synthesisable scoring
- Objectness and confidence as separate outputs
- Reset-aware module design
- CNN-style feature extraction (convolution + ReLU + detection)
- Performance evaluation in hardware systems
 
---

## Notes
- **Warm-up:** The first 3 frames show `Confidence: 0.00%` and `Objectness: NO` — expected while the pipeline fills from reset.
- **Tracking error:** It is bounded by the grid cell size (2px), since detection is cell-level rather than pixel-level.
- **Gaussian blob:** The object is modelled as a blob with decreasing intensity (20 → 14 → 8) so the convolution produces a strong localised response, making grid-cell detection robust against noise.
- **Objectness threshold:** Set to `10`. Cells with max activation ≤ 10 are treated as background — directly analogous to the confidence threshold used during real YOLO inference.

---

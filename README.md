# YOLO Object Detection Accelerator (SystemC)

## Overview
This project implements a pipelined YOLO-inspired object detection accelerator using SystemC.  
The design models a processing pipeline with real-time streaming input, 2D convolution, bounding box prediction, and performance evaluation.

---

## Features
- Pipelined architecture (Convolution → ReLU → Detection)
- Real 2D convolution using a sharpening kernel with zero-padding
- Hardware-friendly fixed-point confidence score (no floating point)
- Randomised object position and background noise — different every run
- Bounding box output: (x, y, width, height, confidence)
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
- Generates a 4×4 image frame every clock cycle
- Object pixel = `20`; background = random noise `0–4`, re-seeded each run
- Object position randomised each frame via `rand() % IMG_SIZE`
- Holds all outputs at zero during reset

### Conv
- Applies a real 2D convolution using a 3×3 sharpening kernel with zero-padding at borders

### ReLU
- Element-wise activation: clamps all negative convolution outputs to zero

### Detect
- Argmax search across ReLU output to find the brightest pixel (detected object)
- 3-stage pipeline shift register aligns ground-truth position with detection output, compensating for the 3-cycle delay
- Pipeline registers are module member variables — reset cleanly on `rst`
- Computes fixed-point confidence score and Manhattan distance tracking error

---

## Detection Output

For each frame (after 3-cycle pipeline warm-up):
```
Actual Object (aligned): (x, y)
Detected BBox: (x, y, 1, 1)
Confidence: XX.YY%
Tracking Error: N
```

---

## Confidence Score
 
Computed using integer fixed-point arithmetic — no floats, fully synthesisable:
```
conf_fp = (max_val - second_max) * 10000 / max_val
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
- Hardware pipelining and pipeline alignment
- Real 2D convolution in hardware
- Fixed-point arithmetic for synthesisable confidence scoring
- Reset-aware module design
- Parallel processing across a pixel grid
- Real-time data streaming simulation
- CNN-style feature extraction (convolution + ReLU + detection)
- Performance evaluation in hardware systems

---

## Notes
- **Warm-up:** The first 3 frames show `Confidence: 0.00%` — expected while the pipeline fills from reset.
- **Tracking error:** Is `0` in steady state — argmax reliably finds the object and the pipeline alignment register compensates for the 3-cycle delay.
- **Noise robustness:** The sharpening kernel amplifies the high-value object pixel relative to low-value noise, keeping detection accurate across different random backgrounds.

---

# YOLO Object Detection Accelerator (SystemC)

## Overview
This project implements a pipelined YOLO object detection accelerator using SystemC.  
The design models a processing pipeline with real-time streaming input, bounding box prediction, and performance evaluation.

---

## Features
- Pipelined architecture (Convolution → ReLU → Detection)
- Real-time input generation (frame-by-frame simulation)
- Moving object tracking
- Bounding box output: (x, y, width, height, confidence)
- Performance metrics:
  - Latency
  - Throughput
  - Tracking error

---

## System Architecture
Input Generator → Convolution → ReLU → Detection

---

## Input Model
- A moving object is simulated across frames
- Background contains random noise
- Each clock cycle represents a new frame
- Mimics real-time video stream input

---

## Detection Output
For each frame:
(x, y, width, height, confidence)

---

## Performance Metrics
- **Latency:** 3 cycles (30 ns)
- **Throughput:** ~1 output per cycle (after pipeline fill)
- **Tracking Error:** Measures difference between actual and detected position

---

## Tools and Technologies used
- SystemC
- Hardware modeling concepts

---

## Key Concepts Demonstrated
- Hardware pipelining
- Parallel processing
- Real-time data streaming
- CNN-style feature extraction
- Performance evaluation in hardware systems

---

## Notes
- Detection may not exactly match object position due to convolution spreading effects
- Tracking error reflects this behavior and validates detection performance
  
---

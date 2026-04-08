#include "yolo.h"
#include <cstdlib>   // abs(), rand()
#include <iostream>
using namespace std;

// Sharpening kernel — amplifies centre pixel relative to neighbours.
// This makes the object pixel stand out clearly after convolution,
// which is what allows the grid-based argmax to detect it reliably.
int kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    { 0, -1,  0},
    {-1,  5, -1},
    { 0, -1,  0}
};

// Objectness threshold — minimum cell activation to declare an object present.
// Models the confidence threshold used in real YOLO inference.
#define OBJECTNESS_THRESHOLD 10

//----------------------------------
// INPUT GENERATOR
//----------------------------------
void InputGen::generate()
{
    if (rst.read())
    {
        obj_x_out.write(0);
        obj_y_out.write(0);
        for (int i = 0; i < IMG_SIZE; i++)
            for (int j = 0; j < IMG_SIZE; j++)
                image[i][j].write(0);
        return;
    }

    static int frame = 0;

    // Object centre — random pixel position each frame
    int obj_x = rand() % IMG_SIZE;
    int obj_y = rand() % IMG_SIZE;

    cout << "\n[Input Frame " << frame << "]";
    cout << "  Object centre: (" << obj_x << "," << obj_y << ")\n";

    obj_x_out.write(obj_x);
    obj_y_out.write(obj_y);

    // Generate 16x16 image:
    // Object is a Gaussian-like blob — peak at (obj_x, obj_y), falling off
    // with Manhattan distance. Background is random noise 0-4.
    // This is more realistic than a single bright pixel and better
    // represents how a real object looks in a feature map.
    for (int i = 0; i < IMG_SIZE; i++)
    {
        for (int j = 0; j < IMG_SIZE; j++)
        {
            int dist = abs(i - obj_x) + abs(j - obj_y);
            int val;
            if      (dist == 0) val = 20;          // peak
            else if (dist == 1) val = 14;          // inner ring
            else if (dist == 2) val = 8;           // outer ring
            else                val = rand() % 5;  // background noise

            image[i][j].write(val);
            cout << val << "\t";
        }
        cout << "\n";
    }

    frame++;
}

//----------------------------------
// CONVOLUTION
// Applies 3x3 sharpening kernel over the full 16x16 image.
// Zero-padding used at borders so output size equals input size.
//----------------------------------
void Conv::process()
{
    if (rst.read())
    {
        for (int i = 0; i < IMG_SIZE; i++)
            for (int j = 0; j < IMG_SIZE; j++)
                conv_out[i][j].write(0);
        return;
    }

    int half = KERNEL_SIZE / 2;

    for (int i = 0; i < IMG_SIZE; i++)
    {
        for (int j = 0; j < IMG_SIZE; j++)
        {
            int sum = 0;

            for (int ki = 0; ki < KERNEL_SIZE; ki++)
            {
                for (int kj = 0; kj < KERNEL_SIZE; kj++)
                {
                    int si = i + ki - half;
                    int sj = j + kj - half;

                    int pixel = (si >= 0 && si < IMG_SIZE &&
                                 sj >= 0 && sj < IMG_SIZE)
                                ? image[si][sj].read() : 0;

                    sum += pixel * kernel[ki][kj];
                }
            }

            conv_out[i][j].write(sum);
        }
    }
}

//----------------------------------
// RELU
// Clamps negative values to zero.
// Removes negative convolution responses — only positive activations pass.
//----------------------------------
void ReLU::process()
{
    if (rst.read())
    {
        for (int i = 0; i < IMG_SIZE; i++)
            for (int j = 0; j < IMG_SIZE; j++)
                relu_out[i][j].write(0);
        return;
    }

    for (int i = 0; i < IMG_SIZE; i++)
    {
        for (int j = 0; j < IMG_SIZE; j++)
        {
            int val = conv_in[i][j].read();
            if (val < 0) val = 0;
            relu_out[i][j].write(val);
        }
    }
}

//----------------------------------
// DETECTION
//
// YOLO grid-based detection:
// 1. Divide the 6x6 feature map into a 2x2 grid of cells (3px each).
// 2. For each grid cell, compute the maximum activation within that cell.
// 3. The cell with the highest max activation is the predicted detection.
// 4. The bounding box is the pixel-coordinate extent of that grid cell.
// 5. Objectness flag signals whether any cell exceeds the threshold.
// 6. Confidence score measures how dominant the best cell is over the runner-up.
//----------------------------------
void Detect::process()
{
    if (rst.read())
    {
        bbox_x.write(0);
        bbox_y.write(0);
        bbox_w.write(0);
        bbox_h.write(0);
        confidence.write(0);
        objectness.write(false);

        for (int i = 0; i < 3; i++) { x_pipe[i] = 0; y_pipe[i] = 0; }
        return;
    }

    // -------- 3-CYCLE PIPELINE ALIGNMENT --------
    // Ground-truth position takes 3 cycles to propagate through
    // InputGen → Conv → ReLU before reaching Detect.
    // Shift register delays the ground-truth by the same 3 cycles
    // so comparison is valid.
    x_pipe[0] = x_pipe[1];
    x_pipe[1] = x_pipe[2];
    x_pipe[2] = obj_x_in.read();

    y_pipe[0] = y_pipe[1];
    y_pipe[1] = y_pipe[2];
    y_pipe[2] = obj_y_in.read();

    int actual_x = x_pipe[0];
    int actual_y = y_pipe[0];

    // -------- GRID-BASED ARGMAX (core YOLO concept) --------
    // For each of the GRID_SIZE x GRID_SIZE cells, find the max pixel
    // activation within that cell. The cell with the highest max
    // is declared the detection — this is exactly what YOLO does.
    int best_cell_x   = 0;
    int best_cell_y   = 0;
    int best_cell_val = 0;
    int second_cell_val = 0;

    for (int gi = 0; gi < GRID_SIZE; gi++)
    {
        for (int gj = 0; gj < GRID_SIZE; gj++)
        {
            // Compute max activation inside this grid cell
            int cell_max = 0;
            for (int pi = 0; pi < CELL_SIZE; pi++)
            {
                for (int pj = 0; pj < CELL_SIZE; pj++)
                {
                    int val = relu_in[gi * CELL_SIZE + pi]
                                     [gj * CELL_SIZE + pj].read();
                    if (val > cell_max)
                        cell_max = val;
                }
            }

            // Track best and second-best cell for confidence calculation
            if (cell_max > best_cell_val)
            {
                second_cell_val = best_cell_val;
                best_cell_val   = cell_max;
                best_cell_x     = gi;
                best_cell_y     = gj;
            }
            else if (cell_max > second_cell_val)
            {
                second_cell_val = cell_max;
            }
        }
    }

    // -------- OBJECTNESS (YOLO dual output — part 1) --------
    // Binary flag: is the best cell activation above the threshold?
    // Models the objectness score in real YOLO — separates "object present"
    // from "how confident are we about the location".
    bool obj_present = (best_cell_val > OBJECTNESS_THRESHOLD);
    objectness.write(obj_present);

    // -------- CONFIDENCE (YOLO dual output — part 2) --------
    // Fixed-point integer: how dominant is the best cell vs the runner-up?
    // Scaled by 10000 — value 8350 means 83.50%. No floats, fully synthesisable.
    int conf_fp = (best_cell_val > 0)
                  ? ((best_cell_val - second_cell_val) * 10000) / best_cell_val
                  : 0;

    // -------- BOUNDING BOX (cell-relative, YOLO-style) --------
    // bbox top-left = grid cell origin in pixel coordinates.
    // bbox width and height = CELL_SIZE (each prediction covers one full cell).
    // In real YOLO, (x,y) is predicted as an offset within the cell;
    int out_bbox_x = best_cell_x * CELL_SIZE;
    int out_bbox_y = best_cell_y * CELL_SIZE;

    // -------- TRACKING ERROR --------
    // Manhattan distance between actual object pixel and detected cell origin.
    int error = abs(out_bbox_x - actual_x) + abs(out_bbox_y - actual_y);

    // -------- OUTPUT --------
    int conf_int  = conf_fp / 100;
    int conf_frac = conf_fp % 100;

    cout << "Actual Object (aligned) : (" << actual_x << "," << actual_y << ")\n";
    cout << "Detected Grid Cell      : (" << best_cell_x << "," << best_cell_y << ")"
         << "  [cell size = " << CELL_SIZE << "px]\n";
    cout << "Bounding Box (pixels)   : ("
         << out_bbox_x << "," << out_bbox_y << ","
         << CELL_SIZE  << "," << CELL_SIZE  << ")\n";
    cout << "Objectness              : " << (obj_present ? "YES — object detected" : "NO  — below threshold") << "\n";
    cout << "Confidence              : " << conf_int << "."
         << (conf_frac < 10 ? "0" : "") << conf_frac << "%\n";
    cout << "Tracking Error          : " << error << " px\n";

    bbox_x.write(out_bbox_x);
    bbox_y.write(out_bbox_y);
    bbox_w.write(CELL_SIZE);
    bbox_h.write(CELL_SIZE);
    confidence.write(conf_fp);
}
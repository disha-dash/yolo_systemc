#include "yolo.h"
#include <cstdlib>   // abs(), rand()
#include <iostream>
using namespace std;

// Edge-detection kernel (replaces unused identity kernel)
// Uses a simple sharpening kernel so convolution has a visible effect
int kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    { 0, -1,  0},
    {-1,  5, -1},
    { 0, -1,  0}
};

//----------------------------------
// INPUT GENERATOR
//----------------------------------
void InputGen::generate()
{
    // FIX #5: respect reset — hold outputs at 0 during reset
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

    int obj_x = rand() % IMG_SIZE;
    int obj_y = rand() % IMG_SIZE;

    cout << "\n[Input Frame " << frame << "]\n";

    obj_x_out.write(obj_x);
    obj_y_out.write(obj_y);

    for (int i = 0; i < IMG_SIZE; i++)
    {
        for (int j = 0; j < IMG_SIZE; j++)
        {
            int val = (i == obj_x && j == obj_y) ? 20 : rand() % 5;
            image[i][j].write(val);
            cout << val << " ";
        }
        cout << endl;
    }

    frame++;
}

//----------------------------------
// CONVOLUTION  (FIX #1: actual 2-D convolution using kernel)
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

                    // Zero-padding at borders
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

        // FIX #2: reset member pipeline registers on rst
        for (int i = 0; i < 3; i++) { x_pipe[i] = 0; y_pipe[i] = 0; }
        return;
    }

    // -------- 3-CYCLE PIPELINE ALIGNMENT (member variables) --------
    // FIX #2: use member x_pipe/y_pipe instead of static locals
    x_pipe[0] = x_pipe[1];
    x_pipe[1] = x_pipe[2];
    x_pipe[2] = obj_x_in.read();

    y_pipe[0] = y_pipe[1];
    y_pipe[1] = y_pipe[2];
    y_pipe[2] = obj_y_in.read();

    int actual_x = x_pipe[0];
    int actual_y = y_pipe[0];

    // -------- ARGMAX DETECTION --------
    int max_val  = 0;
    int cx = 0, cy = 0;

    for (int i = 0; i < IMG_SIZE; i++)
    {
        for (int j = 0; j < IMG_SIZE; j++)
        {
            int val = relu_in[i][j].read();
            if (val > max_val)
            {
                max_val = val;
                cx = i;
                cy = j;
            }
        }
    }

    // -------- SECOND MAX (FOR REALISTIC CONFIDENCE) --------
    int second_max = 0;

    for (int i = 0; i < IMG_SIZE; i++)
    {
        for (int j = 0; j < IMG_SIZE; j++)
        {
            int val = relu_in[i][j].read();
            if (val > second_max && val < max_val)
                second_max = val;
        }
    }

    // -------- FIXED-POINT CONFIDENCE (hardware-friendly, no float) --------
    // Scaled by 10000: value 8350 means 83.50%
    // Formula: conf_fp = (max_val - second_max) * 10000 / max_val
    // All operations are pure integer — synthesisable as-is
    int conf_fp = (max_val > 0)
                  ? ((max_val - second_max) * 10000) / max_val
                  : 0;

    // -------- TRACKING ERROR --------
    int error = abs(cx - actual_x) + abs(cy - actual_y);

    // -------- OUTPUT --------
    // Display as XX.YY% by splitting integer and fractional parts
    int conf_int  = conf_fp / 100;          // e.g. 83
    int conf_frac = conf_fp % 100;          // e.g. 50

    cout << "Actual Object (aligned): ("
         << actual_x << "," << actual_y << ")\n";
    cout << "Detected BBox: ("
         << cx << "," << cy << ",1,1)\n";
    cout << "Confidence: " << conf_int << "."
         << (conf_frac < 10 ? "0" : "") << conf_frac << "%\n";
    cout << "Tracking Error: " << error << "\n";

    bbox_x.write(cx);
    bbox_y.write(cy);
    bbox_w.write(1);
    bbox_h.write(1);
    confidence.write(conf_fp);   // hardware signal carries fixed-point value
}
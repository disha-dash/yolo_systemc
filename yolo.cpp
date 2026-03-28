// ================= yolo.cpp (IMPROVED) =================
#include "yolo.h"
#include <cstdlib>
#include <iostream>
#include <cmath>
using namespace std;

// Kernel (sharper)
int kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {0,1,0},
    {1,4,1},
    {0,1,0}
};

//----------------------------------
// INPUT GENERATOR
//----------------------------------
void InputGen::generate()
{
    static int frame = 0;

    int obj_x = frame % IMG_SIZE;
    int obj_y = frame % IMG_SIZE;

    cout << "\n[Input Frame " << frame << "]\n";
    cout << "Actual Object: (" << obj_x << "," << obj_y << ")\n";

    obj_x_out.write(obj_x);
    obj_y_out.write(obj_y);

    for(int i=0;i<IMG_SIZE;i++)
    {
        for(int j=0;j<IMG_SIZE;j++)
        {
            int val = (i==obj_x && j==obj_y) ? 20 : rand()%5;
            image[i][j].write(val);
            cout << val << " ";
        }
        cout << endl;
    }

    frame++;
}

//----------------------------------
// CONVOLUTION
//----------------------------------
void Conv::process()
{
    if (rst.read())
    {
        for(int i=0;i<IMG_SIZE;i++)
            for(int j=0;j<IMG_SIZE;j++)
                conv_out[i][j].write(0);
        return;
    }

    for(int i=1;i<IMG_SIZE-1;i++)
    {
        for(int j=1;j<IMG_SIZE-1;j++)
        {
            int sum = 0;

            for(int ki=0;ki<KERNEL_SIZE;ki++)
            {
                for(int kj=0;kj<KERNEL_SIZE;kj++)
                {
                    sum += image[i+ki-1][j+kj-1].read() * kernel[ki][kj];
                }
            }

            sum /= 8; // normalize
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
        for(int i=0;i<IMG_SIZE;i++)
            for(int j=0;j<IMG_SIZE;j++)
                relu_out[i][j].write(0);
        return;
    }

    for(int i=0;i<IMG_SIZE;i++)\    {
        for(int j=0;j<IMG_SIZE;j++)
        {
            int val = conv_in[i][j].read();
            if(val < 0) val = 0;
            relu_out[i][j].write(val);
        }
    }
}

//----------------------------------
// DETECTION (REGRESSION BASED)
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
        return;
    }

    float sum_val = 0;
    float weighted_x = 0;
    float weighted_y = 0;

    for(int i=0;i<IMG_SIZE;i++)
    {
        for(int j=0;j<IMG_SIZE;j++)
        {
            float v = relu_in[i][j].read();
            sum_val += v;
            weighted_x += i * v;
            weighted_y += j * v;
        }
    }

    float cx = (sum_val > 0) ? (weighted_x / sum_val) : 0;
    float cy = (sum_val > 0) ? (weighted_y / sum_val) : 0;

    // Variance for width & height
    float var_x = 0, var_y = 0;

    for(int i=0;i<IMG_SIZE;i++)
    {
        for(int j=0;j<IMG_SIZE;j++)
        {
            float v = relu_in[i][j].read();
            var_x += (i - cx)*(i - cx) * v;
            var_y += (j - cy)*(j - cy) * v;
        }
    }

    float w = (sum_val > 0) ? sqrt(var_x / sum_val) : 1;
    float h = (sum_val > 0) ? sqrt(var_y / sum_val) : 1;

    // Confidence normalized to 0-100
    float max_possible = IMG_SIZE * IMG_SIZE * 20.0;
    float conf = (sum_val / max_possible) * 100.0;

    int actual_x = obj_x_in.read();
    int actual_y = obj_y_in.read();

    int error = abs((int)cx - actual_x) + abs((int)cy - actual_y);

    cout << "Detected BBox: (" << cx << "," << cy
         << "," << w << "," << h << ")\n";
    cout << "Confidence: " << conf << "\n";
    cout << "Tracking Error: " << error << "\n";

    bbox_x.write((int)cx);
    bbox_y.write((int)cy);
    bbox_w.write((int)w);
    bbox_h.write((int)h);
    confidence.write((int)conf);
}

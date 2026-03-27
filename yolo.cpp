#include "yolo.h"
#include <cstdlib>
#include <iostream>
using namespace std;

// Performance tracking
static int current_cycle = 0;
static int first_output_cycle = -1;
static int total_error = 0;

// Kernel
int kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {1,1,1},
    {1,1,1},
    {1,1,1}
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

    // WRITE ONCE PER FRAME
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

    for(int i=0;i<IMG_SIZE;i++)
    {
        for(int j=0;j<IMG_SIZE;j++)
        {
            int val = conv_in[i][j].read();
            if(val < 0) val = 0;

            relu_out[i][j].write(val);
        }
    }
}

//----------------------------------
// DETECTION + METRICS
//----------------------------------
void Detect::process()
{
    current_cycle++;

    if (rst.read())
    {
        bbox_x.write(0);
        bbox_y.write(0);
        bbox_w.write(0);
        bbox_h.write(0);
        confidence.write(0);
        return;
    }

    int max_val = 0;
    int cx = 0, cy = 0;

    for(int i=0;i<IMG_SIZE;i++)
    {
        for(int j=0;j<IMG_SIZE;j++)
        {
            int val = relu_in[i][j].read();

            if(val > max_val)
            {
                max_val = val;
                cx = i;
                cy = j;
            }
        }
    }

    if(first_output_cycle == -1 && max_val > 0)
    {
        first_output_cycle = current_cycle;
    }

    int w = 1 + (max_val % IMG_SIZE);
    int h = 1 + ((max_val / IMG_SIZE) % IMG_SIZE);

    int actual_x = obj_x_in.read();
    int actual_y = obj_y_in.read();

    int error = abs(cx - actual_x) + abs(cy - actual_y);
    total_error += error;

    cout << "Detected BBox: (" << cx << "," << cy 
         << "," << w << "," << h << ")\n";
    cout << "Confidence: " << max_val << "\n";
    cout << "Tracking Error: " << error << "\n";

    bbox_x.write(cx);
    bbox_y.write(cy);
    bbox_w.write(w);
    bbox_h.write(h);
    confidence.write(max_val);
}
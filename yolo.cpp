#include "yolo.h"
#include <cstdlib>
#include <iostream>
using namespace std;

// Identity kernel (no blur)
int kernel[KERNEL_SIZE][KERNEL_SIZE] = {
    {0,0,0},
    {0,1,0},
    {0,0,0}
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
// CONVOLUTION (PASS-THROUGH)
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

    for(int i=0;i<IMG_SIZE;i++)
    {
        for(int j=0;j<IMG_SIZE;j++)
        {
            conv_out[i][j].write(image[i][j].read());
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
// DETECTION (FINAL VERSION)
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

    // -------- 3-CYCLE PIPELINE ALIGNMENT --------
    static int x_pipe[3] = {0,0,0};
    static int y_pipe[3] = {0,0,0};

    x_pipe[0] = x_pipe[1];
    x_pipe[1] = x_pipe[2];
    x_pipe[2] = obj_x_in.read();

    y_pipe[0] = y_pipe[1];
    y_pipe[1] = y_pipe[2];
    y_pipe[2] = obj_y_in.read();

    int actual_x = x_pipe[0];
    int actual_y = y_pipe[0];

    // -------- ARGMAX DETECTION --------
    int max_val = 0;
    int cx = 0, cy = 0;

    for(int i=0;i<IMG_SIZE;i++){
        for(int j=0;j<IMG_SIZE;j++){
            int val = relu_in[i][j].read();
            if(val > max_val){
                max_val = val;
                cx = i;
                cy = j;
            }
        }
    }

    // -------- SECOND MAX (FOR REALISTIC CONFIDENCE) --------
    int second_max = 0;

    for(int i=0;i<IMG_SIZE;i++){
        for(int j=0;j<IMG_SIZE;j++){
            int val = relu_in[i][j].read();
            if(val > second_max && val < max_val){
                second_max = val;
            }
        }
    }

    float conf = ((max_val - second_max) / 20.0) * 100;

    // -------- ERROR --------
    int error = abs(cx - actual_x) + abs(cy - actual_y);

    // -------- CLEAN PRINTING (ALIGNED) --------
    cout << "Actual Object (aligned): (" 
         << actual_x << "," << actual_y << ")\n";

    cout << "Detected BBox: (" 
         << cx << "," << cy << ",1,1)\n";

    cout << "Confidence: " << conf << "\n";
    cout << "Tracking Error: " << error << "\n";

    bbox_x.write(cx);
    bbox_y.write(cy);
    bbox_w.write(1);
    bbox_h.write(1);
    confidence.write((int)conf);
}
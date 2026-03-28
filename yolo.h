#ifndef YOLO_H
#define YOLO_H

#include <systemc.h>

#define IMG_SIZE 8
#define KERNEL_SIZE 3

//----------------------------------
// Input Generator
//----------------------------------
SC_MODULE(InputGen)
{
    sc_in<bool> clk;
    sc_out<int> image[IMG_SIZE][IMG_SIZE];

    sc_out<int> obj_x_out, obj_y_out; // NEW

    void generate();

    SC_CTOR(InputGen)
    {
        SC_METHOD(generate);
        sensitive << clk.pos();
    }
};

//----------------------------------
// Convolution
//----------------------------------
SC_MODULE(Conv)
{
    sc_in<bool> clk;
    sc_in<bool> rst;

    sc_in<int> image[IMG_SIZE][IMG_SIZE];
    sc_out<int> conv_out[IMG_SIZE][IMG_SIZE];

    void process();

    SC_CTOR(Conv)
    {
        SC_METHOD(process);
        sensitive << clk.pos();
    }
};

//----------------------------------
// ReLU
//----------------------------------
SC_MODULE(ReLU)
{
    sc_in<bool> clk;
    sc_in<bool> rst;

    sc_in<int> conv_in[IMG_SIZE][IMG_SIZE];
    sc_out<int> relu_out[IMG_SIZE][IMG_SIZE];

    void process();

    SC_CTOR(ReLU)
    {
        SC_METHOD(process);
        sensitive << clk.pos();
    }
};

//----------------------------------
// Detection
//----------------------------------
SC_MODULE(Detect)
{
    sc_in<bool> clk;
    sc_in<bool> rst;

    sc_in<int> relu_in[IMG_SIZE][IMG_SIZE];

    sc_in<int> obj_x_in, obj_y_in; // NEW

    sc_out<int> bbox_x, bbox_y;
    sc_out<int> bbox_w, bbox_h;
    sc_out<int> confidence;

    void process();

    SC_CTOR(Detect)
    {
        SC_METHOD(process);
        sensitive << clk.pos();
    }
};

#endif
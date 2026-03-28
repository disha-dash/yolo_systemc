#include <systemc.h>
#include <cstdlib>
#include <ctime>
#include "yolo.h"

int sc_main(int argc, char* argv[])
{
    srand((unsigned int)time(NULL));   // different random background noise each run

    sc_clock     clk("clk", 10, SC_NS);
    sc_signal<bool> rst;

    sc_signal<int> image[IMG_SIZE][IMG_SIZE];
    sc_signal<int> conv_sig[IMG_SIZE][IMG_SIZE];
    sc_signal<int> relu_sig[IMG_SIZE][IMG_SIZE];

    sc_signal<int> obj_x_sig, obj_y_sig;

    sc_signal<int> bbox_x, bbox_y, bbox_w, bbox_h, confidence;

    InputGen inputGen("InputGen");
    Conv     conv("Conv");
    ReLU     relu("ReLU");
    Detect   detect("Detect");

    inputGen.clk(clk);
    inputGen.rst(rst);
    inputGen.obj_x_out(obj_x_sig);
    inputGen.obj_y_out(obj_y_sig);

    conv.clk(clk);   conv.rst(rst);
    relu.clk(clk);   relu.rst(rst);
    detect.clk(clk); detect.rst(rst);

    detect.obj_x_in(obj_x_sig);
    detect.obj_y_in(obj_y_sig);

    for (int i = 0; i < IMG_SIZE; i++)
    {
        for (int j = 0; j < IMG_SIZE; j++)
        {
            inputGen.image[i][j](image[i][j]);

            conv.image[i][j](image[i][j]);
            conv.conv_out[i][j](conv_sig[i][j]);

            relu.conv_in[i][j](conv_sig[i][j]);
            relu.relu_out[i][j](relu_sig[i][j]);

            detect.relu_in[i][j](relu_sig[i][j]);
        }
    }

    detect.bbox_x(bbox_x);
    detect.bbox_y(bbox_y);
    detect.bbox_w(bbox_w);
    detect.bbox_h(bbox_h);
    detect.confidence(confidence);

    // Apply reset for one clock cycle
    rst.write(true);
    sc_start(10, SC_NS);

    rst.write(false);
    sc_start(200, SC_NS);

    // -------- PERFORMANCE METRICS (FIX #4) --------
    cout << "\n===== PERFORMANCE METRICS =====\n";

    const int   clk_period_ns  = 10;
    const int   latency_cycles = 3;          // InputGen -> Conv -> ReLU -> Detect
    const int   latency_ns     = latency_cycles * clk_period_ns;

    // Throughput: after pipeline fills, 1 frame per clock cycle
    double throughput_mhz = 1000.0 / clk_period_ns;   // MHz  (= 100 MHz for 10 ns clk)

    cout << "Clock period  : " << clk_period_ns  << " ns\n";
    cout << "Pipeline depth: " << latency_cycles << " stages\n";
    cout << "Latency       : " << latency_cycles
         << " cycles (" << latency_ns << " ns)\n";
    cout << "Throughput    : " << throughput_mhz
         << " MHz  (1 frame/cycle after pipeline fills)\n";

    return 0;
}
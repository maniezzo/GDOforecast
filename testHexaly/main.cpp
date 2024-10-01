#include "c:/hexaly_13_0/include/optimizer/hexalyoptimizer.h"
#include <fstream>
#include <iostream>
#include <vector>

using namespace hexaly;
using namespace std;

class OptimalBucket {
public:
    // Hexaly Optimizer
    HexalyOptimizer optimizer;

    // Hexaly Program variables
    HxExpression R;
    HxExpression r;
    HxExpression h;

    HxExpression surface;
    HxExpression volume;

    void solve(int limit) {
        hxdouble PI = 3.14159265359;

        // Declare the optimization model
        HxModel model = optimizer.getModel();

        // Numerical decisions
        R = model.floatVar(0.0, 1.0);
        r = model.floatVar(0.0, 1.0);
        h = model.floatVar(0.0, 1.0);

        // Surface must not exceed the surface of the plain disc
        surface = PI * model.pow(r, 2) + PI * (R + r) * model.sqrt(model.pow(R - r, 2) + model.pow(h, 2));
        model.constraint(model.leq(surface, PI));

        // Maximize the volume
        volume = PI * h / 3 * (model.pow(R, 2) + R * r + model.pow(r, 2));
        model.maximize(volume);

        model.close();

        // Parametrize the optimizer
        optimizer.getParam().setTimeLimit(limit);

        optimizer.solve();
    }

    /* Write the solution in a file with the following format:
     *  - surface and volume of the bucket
     *  - values of R, r and h */
    void writeSolution(const string& fileName) {
        ofstream outfile;
        outfile.exceptions(ofstream::failbit | ofstream::badbit);
        outfile.open(fileName.c_str());

        outfile << surface.getDoubleValue() << " " << volume.getDoubleValue() << endl;
        outfile << R.getDoubleValue() << " " << r.getDoubleValue() << " " << h.getDoubleValue() << endl;
    }
};

int main(int argc, char** argv) {
    const char* solFile = argc > 1 ? argv[1] : NULL;
    const char* strTimeLimit = argc > 2 ? argv[2] : "2";

    try {
        OptimalBucket model;
        model.solve(atoi(strTimeLimit));
        if (solFile != NULL)
            model.writeSolution(solFile);
        return 0;
    } catch (const exception& e) {
        cerr << "An error occurred:" << e.what() << endl;
        return 1;
    }
}
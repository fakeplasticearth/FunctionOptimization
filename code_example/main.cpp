/* \file main.cpp
* 
*/
#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES

#include <iostream>
#include <math.h>
#include <cstdio>
#include "../StohasticOptimization.h"
#include "../GradientDescent.h"

//https://en.wikipedia.org/wiki/Rosenbrock_function
// Minimum point: {1, 1}
double rosenbrock_func2(std::vector<double> arg) {
    return pow(1 - arg[0], 2) + 100 * pow(arg[1] - pow(arg[0], 2), 2);
}

std::vector<double> rosenbrock_func2_grad(std::vector<double> arg) {
    return std::vector<double>({ 2 * (200 * pow(arg[0], 3) - 200 * arg[0] * arg[1] + arg[0] - 1), 200 * (arg[1] - pow(arg[0], 2)) });
}

/*! \example main.cpp
* Example of optimization.
*/
int main() {
    /*
    * Lets perform a stohastic optimization of Rosenbrock function.
    */
    dimensional_limits x_limits(-10, 10), y_limits(-10, 10); // Dimensional limits
    BoxArea box_area(2, { x_limits, y_limits }); // Area of search
    std::vector<double> first_point = {-4.5, 3.4}; // Initial point of search
    Function func(rosenbrock_func2, 2); // Function that we will optimize
    /*
    * Lets set stop criterion. Let the maximum number of iterations be 5000 and maximum number of iterations after last improvement be 200.
    */
    StopCriterion* stop_criterion = new IterAfterImpSSC(5000, 200);
    double delta = 1; // Delta parameter
    double p = 0.5; // Probability parameter
    /*
    * Lets set optimization method with the parameters.
    */
    OptimizationMethod* optimization_method = new StohasticOptimization(&func, &box_area, stop_criterion, first_point, delta, p);

    /*
    * Perform optimization.
    */
    OptResult opt_result = optimization_method->optimize();
    std::cout << "Optimization result: \n";
    std::cout << "\t Iteration num: " << opt_result.get_n_iter();
    std::cout << "\n\t Optimal point: ";
    std::vector<double> opt_point = opt_result.get_min_point();
    // Print coordinates of min point.
    for (int i = 0; i < opt_point.size(); ++i) {
        std::cout << opt_point[i] << ' ';
    }

    std::cout << "\n\t Function value: " << opt_result.get_min_value() << "\n";

    delete optimization_method;
    delete stop_criterion;
    /*
    * Now lets perform a gradient descent
    */
    dimensional_limits x_limits2(-5, 5), y_limits2(-10, 15); // Dimensional limits
    BoxArea box_area2(2, { x_limits2, y_limits2 }); // Area of search
    std::vector<double> first_point2 = { -3, 3 }; // Initial point of search
    Function func2(rosenbrock_func2, 2); // Function that we will optimize
    MDFunction grad(rosenbrock_func2_grad, 2); // Gradient of function
    std::string norm_name = "l2"; // Lets use l2 norm
    unsigned int steps = 100; // Lets do 100 steps on each iteration
    /*
    * Lets set stop criterion. Let the maximum number of iterations be 5000 and min grad norm threshold be 1e-6.
    */
    StopCriterion* stop_criterion2 = new MinGradNormGDSC(5000, 1e-6);

    /*
    * Lets set optimization method with the parameters
    */
    OptimizationMethod* optimization_method2 = new GradientDescent(&func2, &grad, &box_area2, stop_criterion2, first_point2, steps, norm_name);

    /*
    * Perform optimization.
    */
    opt_result = optimization_method2->optimize();
    std::cout << "Optimization result: \n";
    std::cout << "\t Iteration num: " << opt_result.get_n_iter();
    std::cout << "\n\t Optimal point: ";
    opt_point = opt_result.get_min_point();
    // Print coordinates of min point.
    for (int i = 0; i < opt_point.size(); ++i) {
        std::cout << opt_point[i] << ' ';
    }

    std::cout << "\n\t Function value: " << opt_result.get_min_value() << "\n";

    delete optimization_method2;
    delete stop_criterion2;
    return 0;
}
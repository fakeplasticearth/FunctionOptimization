#define _CRT_SECURE_NO_WARNINGS
#define _USE_MATH_DEFINES

#include <iostream>
#include <math.h>
#include <cstdio>
#include "StohasticOptimization.h"
#include "GradientDescent.h"

//Functions

// Minimum point: {0, 0}
double spheric_func2(std::vector<double> arg) {
    return pow(arg[0], 2) + pow(arg[1], 2);
}

// Minimum point: {0, 0, 0}
double spheric_func3(std::vector<double> arg) {
    return pow(arg[0], 2) + pow(arg[1], 2) + pow(arg[2], 2);
}

//https://en.wikipedia.org/wiki/Rosenbrock_function
// Minimum point: {1, 1}
double rosenbrock_func2(std::vector<double> arg) {
    return pow(1 - arg[0], 2) + 100 * pow(arg[1] - pow(arg[0], 2), 2);
}

// Minimum point: {1, 1, 1}
double rosenbrock_func3(std::vector<double> arg) {
    return pow(1 - arg[0], 2) + 100 * pow(arg[1] - pow(arg[0], 2), 2) + pow(1 - arg[1], 2) + 100 * pow(arg[2] - pow(arg[1], 2), 2);
}

//https://en.wikipedia.org/wiki/Rastrigin_function
// Minimum point: {0, 0}
double rastrigin_func2(std::vector<double> arg) {
    return 10 * 2 + pow(arg[0], 2) - 10 * cos(2 * M_PI * arg[0]) + pow(arg[1], 2) - 10 * cos(2 * M_PI * arg[1]);
}

// Minimum point: {0, 0, 0}
double rastrigin_func3(std::vector<double> arg) {
    return 10 * 2 + pow(arg[0], 2) - 10 * cos(2 * M_PI * arg[0]) + pow(arg[1], 2) - 10 * cos(2 * M_PI * arg[1]) +
        pow(arg[2], 2) - 10 * cos(2 * M_PI * arg[2]);
}

// Three hump camel function https://en.wikipedia.org/wiki/Test_functions_for_optimization
// Minimum point: {0, 0}
double camel_func(std::vector<double> arg) {
    return 2 * pow(arg[0], 2) - 1.05 * pow(arg[0], 4) + pow(arg[0], 6) / 6 + arg[0] * arg[1] + pow(arg[1], 2);
}

//Gradients

std::vector<double> spheric_func2_grad(std::vector<double> arg) {
    return std::vector<double>({ 2 * arg[0], 2 * arg[1] });
}

std::vector<double> spheric_func3_grad(std::vector<double> arg) {
    return std::vector<double>({ 2 * arg[0], 2 * arg[1], 2 * arg[2] });
}

std::vector<double> rosenbrock_func2_grad(std::vector<double> arg) {
    return std::vector<double>({2 * (200 * pow(arg[0], 3) - 200 * arg[0] * arg[1] + arg[0] - 1), 200 * (arg[1] - pow(arg[0], 2))});
}

std::vector<double> rosenbrock_func3_grad(std::vector<double> arg) {
    return std::vector<double>({ 2 * (200 * pow(arg[0], 3) - 200 * arg[0] * arg[1] + arg[0] - 1), 
        200 * (arg[1] - pow(arg[0], 2)) + 2 * (200 * pow(arg[1], 3) - 200 * arg[1] * arg[2] + arg[1] - 1),
        200 * (arg[2] - pow(arg[1], 2))});
}

std::vector<double> rastrigin_func2_grad(std::vector<double> arg) {
    return std::vector<double>({2 * arg[0] + 20 * M_PI * sin(2 * M_PI * arg[0]),
        2 * arg[1] + 20 * M_PI * sin(2 * M_PI * arg[1]) });
}

std::vector<double> rastrigin_func3_grad(std::vector<double> arg) {
    return std::vector<double>({ 2 * arg[0] + 20 * M_PI * sin(2 * M_PI * arg[0]),
        2 * arg[1] + 20 * M_PI * sin(2 * M_PI * arg[1]),
        2 * arg[2] + 20 * M_PI * sin(2 * M_PI * arg[2]) });
}

std::vector<double> camel_func_grad(std::vector<double> arg) {
    return std::vector<double>({4 * arg[0] - 4.2 * pow(arg[0], 3) + pow(arg[0], 5) + arg[1],
        arg[0] + 2 * arg[1]});
}


int main() {
    Function func;
    MDFunction grad;
    BoxArea box_area;
    OptimizationMethod* optimization_method = nullptr;
    StopCriterion* stop_criterion = nullptr;
    std::vector<dimensional_limits> limits;
    std::vector<double> first_point, opt_point;
    bool f, incorrect_input, method_parameters_type, stop_criterion_parameters_type, first_point_type, exit_programm = 0, norm_type;
    double delta, p, lower, upper, min_last_imp_norm, lr, min_grad_norm, min_last_step_norm, min_rel_imp_norm;
    OptResult opt_result;
    std::string norm_name;

    unsigned int iter_max_num, max_iter_after_imp, dims, method_index, stop_criterion_index, func_index;
    std::cout << "Stohastic optimization and gradient descent \n";

    try {
        while (!exit_programm) {

            std::cout << "Do you want to exit programm? \n\t 0: Set parameters \n\t 1: Exit programm\n>";
            if (!scanf("%hhiu", &exit_programm)) 
                throw std::invalid_argument("Incorrect action index input.");
            if (!exit_programm) {
                func_index = 0;
                std::cout << "Enter function index.\n\t 1: Spheric R^2 \n\t 2: Spheric R^3 \n\t 3: Rosenbrock R^2 \n\t ";
                std::cout << "4: Rosenbrock R^3 \n\t 5: Rastrigin R^2 \n\t 6: Rastrigin R^3 \n\t 7: Camel \n>";
                if (!scanf("%iu", &func_index)) 
                    throw std::invalid_argument("Incorrect function index input.");
                switch (func_index) {
                case 1:
                    func = Function(spheric_func2, 2);
                    grad = MDFunction(spheric_func2_grad, 2);
                    break;
                case 2:
                    func = Function(spheric_func3, 3);
                    grad = MDFunction(spheric_func3_grad, 3);
                    break;
                case 3:
                    func = Function(rosenbrock_func2, 2);
                    grad = MDFunction(rosenbrock_func2_grad, 2);
                    break;
                case 4:
                    func = Function(rosenbrock_func3, 3);
                    grad = MDFunction(rosenbrock_func3_grad, 3);
                    break;
                case 5:
                    func = Function(rastrigin_func2, 2);
                    grad = MDFunction(rastrigin_func2_grad, 2);
                    break;
                case 6:
                    func = Function(rastrigin_func3, 3);
                    grad = MDFunction(rastrigin_func3_grad, 3);
                    break;
                case 7:
                    func = Function(camel_func, 2);
                    grad = MDFunction(camel_func_grad, 2);
                    break;
                default:
                    throw std::invalid_argument("Function index must be a decimal number from 1 to 7.");
                    break;
                }
                dims = func.get_n_dim();
                limits.resize(dims);

                std::cout << "Enter borders of search box: \n";
                for (int i = 0; i < dims; ++i) {
                    incorrect_input = 1;
                    while (incorrect_input) {
                        incorrect_input = 0;
                        std::cout << "Enter dimension " << i + 1 << " limits: ";
                        if (!scanf("%lf", &lower)) 
                            throw std::invalid_argument("Incorrect lower dimensional limit input.");
                        if (!scanf("%lf", &upper)) 
                            throw std::invalid_argument("Incorrect upper dimensional limit input.");
                        if (lower > upper)
                            incorrect_input = 1;
                    }
                    limits[i] = dimensional_limits(lower, upper);
                }
                box_area = BoxArea(dims, limits);

                std::cout << "Enter optimization method index.\n";
                std::cout << "\t 1: Stohastic \n\t 2: Gradient descent \n>";
                method_index = 0;
                incorrect_input = 1;
                while (incorrect_input) {
                    incorrect_input = 0;
                    if (!scanf("%iu", &method_index)) 
                        throw std::invalid_argument("Incorrect method index input.");
                    if (!(method_index == 1 || method_index == 2))
                        incorrect_input = 1;
                }

                std::cout << "Enter parameters of optimization method.\n";
                std::cout << "\t 0: Default parameters \n\t 1: Otherwise \n";
                if (!scanf("%hhiu", &method_parameters_type)) 
                    throw std::invalid_argument("Incorrect method parameters type input.");

                switch (method_index) {
                case 1:
                    switch (method_parameters_type) {
                    case 0:
                        delta = 1;
                        p = 0.5;
                        break;
                    case 1:
                        incorrect_input = 1;
                        while (incorrect_input) {
                            std::cout << "Enter delta parameter.\n>";
                            incorrect_input = 0;
                            if (!scanf("%lf", &delta)) 
                                throw std::invalid_argument("Incorrect delta parameter input.");
                        }

                        incorrect_input = 1;
                        while (incorrect_input) {
                            std::cout << "Enter probability.\n>";
                            incorrect_input = 0;
                            if (!scanf("%lf", &p)) 
                                throw std::invalid_argument("Incorrect probability parameter input.");
                            if (p < 0 || p > 1)
                                incorrect_input = 1;
                        }

                        
                    }
                    
                    std::cout << "Enter stop criterion index.\n";
                    incorrect_input = 1;
                    while (incorrect_input) {
                        incorrect_input = 0;
                        std::cout << "\t 1: Iteration num limit \n\t 2: Iteration num after last improvment limit \n\t 3: Min norm of last improvment\n";
                        if (!scanf("%iu", &stop_criterion_index)) 
                            throw std::invalid_argument("Incorrect stop criterion index input.");
                        if (stop_criterion_index == 0 || stop_criterion_index > 3)
                            incorrect_input = 1;
                    }

                    std::cout << "Stop criterion parameters. \n";
                    std::cout << "\t 0: Default \n\t 1: Otherwise\n";
                    if (!scanf("%hhiu", &stop_criterion_parameters_type)) 
                        throw std::invalid_argument("Incorrect stop criterion parameter type input.");

                    switch (stop_criterion_parameters_type) {
                    case 0:
                        iter_max_num = 5000;
                        max_iter_after_imp = 1000;
                        min_last_imp_norm = 1e-4;
                        break;
                    case 1:
                        incorrect_input = 1;
                        while (incorrect_input) {
                            incorrect_input = 0;
                            std::cout << "Enter max num of iterations.\n>";
                            if (!scanf("%iu", &iter_max_num)) 
                                throw std::invalid_argument("Incorrect max iteration num parameter input.");
                            if (iter_max_num == 0)
                                incorrect_input = 1;
                        }

                        switch (stop_criterion_index) {
                        case 2:
                            incorrect_input = 1;
                            while (incorrect_input) {
                                incorrect_input = 0;
                                std::cout << "Enter iterations after improvment max num.\n>";
                                if (!scanf("%iu", &max_iter_after_imp)) 
                                    throw std::invalid_argument("Incorrect iteration after improvment num parameter input.");
                                if (max_iter_after_imp == 0)
                                    incorrect_input = 1;
                            }
                            break;
                        case 3:
                            incorrect_input = 1;
                            while (incorrect_input) {
                                incorrect_input = 0;
                                std::cout << "Enter min norm of the last improvment.\n>";
                                if (!scanf("%lf", &min_last_imp_norm)) 
                                    throw std::invalid_argument("Incorrect min norm of the last improvment parameter input.");
                            }
                            break;
                        }
                    }

                    first_point = std::vector<double>(dims);
                    std::cout << "Enter first point.\n";
                    std::cout << "\t 0: Default \n\t 1: Otherwise\n";
                    if (!scanf("%hhiu", &first_point_type)) 
                        throw std::invalid_argument("Incorrect first point type parameter input.");
                    switch (first_point_type) {
                    case 0:
                        for (int i = 0; i < dims; ++i) {
                            first_point[i] = limits[i].lower + (limits[i].upper - limits[i].lower) / 2;
                        }
                        break;
                    case 1:
                        incorrect_input = 1;
                        while (incorrect_input) {
                            std::cout << "Enter first point coordinates: ";
                            incorrect_input = 0;
                            for (int i = 0; i < dims; ++i) {
                                if (!scanf("%lf", &first_point[i])) 
                                    throw std::invalid_argument("Incorrect point coordinate input.");
                            }
                            if (!box_area.is_in(first_point)) {
                                incorrect_input = 1;
                                std::cout << "Point is not in the area.\n";
                            }
                        }
                        break;
                    }

                    switch (stop_criterion_index) {
                    case 1:
                        stop_criterion = new IterOnlySSC(iter_max_num);
                        break;
                    case 2:
                        stop_criterion = new IterAfterImpSSC(iter_max_num, max_iter_after_imp);
                        break;
                    case 3:
                        stop_criterion = new NormDiffSSC(iter_max_num, min_last_imp_norm);
                        break;
                    default:
                        stop_criterion = nullptr;
                        break;
                    }
                    optimization_method = new StohasticOptimization(&func, &box_area, stop_criterion, first_point, delta, p);
                    break;

                 case 2:
                     switch (method_parameters_type) {
                     case 0:
                         lr = 3e-4;
                         break;
                     case 1:
                         incorrect_input = 1;
                         while (incorrect_input) {
                             std::cout << "Enter step rate parameter.\n>";
                             incorrect_input = 0;
                             if (!scanf("%lf", &lr))
                                 throw std::invalid_argument("Incorrect step rate parameter input.");
                             if (lr <= 0.)
                                 incorrect_input = 1;
                         }
                     }

                     std::cout << "Enter stop criterion index.\n";
                     incorrect_input = 1;
                     while (incorrect_input) {
                         incorrect_input = 0;
                         std::cout << "\t 1: Min grad norm limit \n\t 2: Min last step norm limit \n\t 3: Min relative improvment norm limit\n>";
                         if (!scanf("%iu", &stop_criterion_index))
                             throw std::invalid_argument("Incorrect stop criterion index input.");
                         if (stop_criterion_index == 0 || stop_criterion_index > 3)
                             incorrect_input = 1;
                     }

                     std::cout << "Enter stop criterion parameters type. \n";
                     std::cout << "\t 0: Default \n\t 1: Otherwise\n>";
                     if (!scanf("%hhiu", &stop_criterion_parameters_type))
                         throw std::invalid_argument("Incorrect stop criterion parameter type input.");

                     switch (stop_criterion_parameters_type) {
                     case 0:
                         iter_max_num = 5000;
                         min_grad_norm = 1e-6;
                         min_last_step_norm = 1e-6;
                         min_rel_imp_norm = 1e-4;
                         break;
                     case 1:
                         incorrect_input = 1;
                         while (incorrect_input) {
                             incorrect_input = 0;
                             std::cout << "Enter max num of iterations.\n>";
                             if (!scanf("%iu", &iter_max_num))
                                 throw std::invalid_argument("Incorrect max iteration num parameter input.");
                             if (iter_max_num == 0)
                                 incorrect_input = 1;
                         }

                         switch (stop_criterion_index) {
                         case 1:
                             incorrect_input = 1;
                             while (incorrect_input) {
                                 incorrect_input = 0;
                                 std::cout << "Enter min grad norm.\n>";
                                 if (!scanf("%lf", &min_grad_norm))
                                     throw std::invalid_argument("Incorrect min grad norm parameter.");
                                 if (min_grad_norm <= 0.)
                                     incorrect_input = 1;
                             }
                             break;
                         case 2:
                             incorrect_input = 1;
                             while (incorrect_input) {
                                 incorrect_input = 0;
                                 std::cout << "Enter min last step norm.\n>";
                                 if (!scanf("%lf", &min_last_step_norm))
                                     throw std::invalid_argument("Incorrect min last step norm parameter.");
                                 if (min_last_step_norm <= 0.)
                                     incorrect_input = 1;
                             }
                             break;
                         case 3:
                             incorrect_input = 1;
                             while (incorrect_input) {
                                 incorrect_input = 0;
                                 std::cout << "Enter min relative improvment norm.\n>";
                                 if (!scanf("%lf", &min_last_imp_norm))
                                     throw std::invalid_argument("Incorrect min relative improvment norm.");
                                 if (min_last_imp_norm <= 0.)
                                     incorrect_input = 1;
                             }
                             break;
                         }
                     }

                     first_point = std::vector<double>(dims);
                     std::cout << "Enter first point.\n";
                     std::cout << "\t 0: Default \n\t 1: Otherwise\n";
                     if (!scanf("%hhiu", &first_point_type))
                         throw std::invalid_argument("Incorrect first point type parameter input.");
                     switch (first_point_type) {
                     case 0:
                         for (int i = 0; i < dims; ++i) {
                             first_point[i] = limits[i].lower + (limits[i].upper - limits[i].lower) / 2;
                         }
                         break;
                     case 1:
                         incorrect_input = 1;
                         while (incorrect_input) {
                             std::cout << "Enter first point coordinates: ";
                             incorrect_input = 0;
                             for (int i = 0; i < dims; ++i) {
                                 if (!scanf("%lf", &first_point[i]))
                                     throw std::invalid_argument("Incorrect point coordinate input.");
                             }
                             if (!box_area.is_in(first_point)) {
                                 incorrect_input = 1;
                                 std::cout << "Point is not in the area.\n";
                             }
                         }
                         break;
                     }

                     switch (stop_criterion_index) {
                     case 1:
                         stop_criterion = new MinGradNormGDSC(iter_max_num, min_grad_norm);
                         break;
                     case 2:
                         stop_criterion = new MinStepNormGDSC(iter_max_num, min_last_step_norm);
                         break;
                     case 3:
                         stop_criterion = new MinRelImpNormGDSC(iter_max_num, min_rel_imp_norm);
                         break;
                     default:
                         stop_criterion = nullptr;
                         break;
                     }

                     std::cout << "Enter norm type.\n";
                     std::cout << "\t 0: Default \n\t 1: Otherwise\n";
                     if (!scanf("%hhiu", &norm_type))
                         throw std::invalid_argument("Incorrect norm type parameter input.");

                     switch (norm_type) {
                     case 0:
                         norm_name = "l2";
                         break;
                     case 1:
                         unsigned int norm_name_index;
                         incorrect_input = 1;
                         while (incorrect_input) {
                             std::cout << "Enter norm index. \n\t 1: l1 norm \n\t 2: l2 norm\n>";
                             incorrect_input = 0;
                             if (!scanf("%iu", &norm_name_index))
                                 throw std::invalid_argument("Incorrect norm index parameter input.");
                             if (norm_name_index != 1 && norm_name_index != 2)
                                 incorrect_input = 1;
                         }
                         switch (norm_name_index) {
                         case 1:
                             norm_name = "l1";
                             break;
                         case 2:
                             norm_name = "l2";
                             break;
                         }

                     }

                     optimization_method = new GradientDescent(&func, &grad, &box_area, stop_criterion, first_point, lr, norm_name);
                     break;
                }

                opt_result = optimization_method -> optimize();
                std::cout << "Optimization result: \n";
                std::cout << "\t Iteration num: " << opt_result.get_n_iter();
                std::cout << "\n\t Optimal point: ";
                opt_point = opt_result.get_min_point();
                for (int i = 0; i < dims; ++i) {
                    std::cout << opt_point[i] << ' ';
                }

                std::cout << "\n\t Function value: " << opt_result.get_min_value() << "\n";
                for (int i = 0; i < 15; ++i)
                    std::cout << '=';
                std::cout << '\n';

                if (stop_criterion)
                    delete stop_criterion;
                if (optimization_method)
                    delete optimization_method;
            }
        }

    }
    catch (std::exception& exeption_) {
        std::cerr << exeption_.what();
    }

	return 0;
}
#pragma once
#include <float.h>
#include "OptimizationMethod.h"
#include "IterOnlySSC.h"
#include "IterAfterImpSSC.h"
#include "NormDiffSSC.h"
#include <random>
#include <chrono>

class StohasticOptimization : public OptimizationMethod
{
private:
    std::vector<double> first_point;
    double curr_min_value;
    double delta;
    double p;
    unsigned int n_iter_after_improvment;
    double last_imp_norm = DBL_MAX;
    std::uniform_real_distribution<double> distribution;
    std::default_random_engine generator;
    unsigned long long seed;
    double unif(double lower, double upper);
public:
    StohasticOptimization();
    StohasticOptimization(Function* func_, BoxArea* box_area_, StopCriterion* stop_criterion_, const std::vector<double>& first_point_, 
        double delta_, double p_);
    StohasticOptimization(const StohasticOptimization& other);
    StohasticOptimization(StohasticOptimization&& other);

    void swap(StohasticOptimization& other) noexcept;
    StohasticOptimization& operator=(StohasticOptimization other) noexcept;
    void set_first_point(const std::vector<double>& point);
    void make_step();
    OptResult optimize();
    OptInfo* get_opt_info() const;
    ~StohasticOptimization();
};


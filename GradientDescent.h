#pragma once
#include "OptimizationMethod.h"
#include "MDFunction.h"
#include "GDOptInfo.h"
#include "MinGradNormGDSC.h"
#include "MinStepNormGDSC.h"
#include "MinRelImpNormGDSC.h"

class GradientDescent : public OptimizationMethod {
private:
    std::vector<double> value_history;
    MDFunction* grad = nullptr;
    std::vector<double> first_point;
    double curr_min_value;
    unsigned int lr_steps;
    double grad_norm;
    double last_step_norm;
    double rel_imp_norm;
    unsigned int max_updates_lr = 10;
    double scale = 10.;
    std::string norm_name;
public:
    GradientDescent();
    GradientDescent(Function* func_, MDFunction* grad_, BoxArea* box_area_, StopCriterion* stop_criterion_, 
        const std::vector<double>& first_point_, unsigned int lr_steps_, std::string norm_name_);
    GradientDescent(const GradientDescent& other);
    GradientDescent(GradientDescent&& other) noexcept;

    void swap(GradientDescent& other) noexcept;
    GradientDescent& operator=(GradientDescent other) noexcept;
    void set_first_point(const std::vector<double>& point);
    void make_step();
    OptResult optimize();
    OptInfo* get_opt_info() const;
    ~GradientDescent();
};

double l2_norm(std::vector<double> arg);
double l1_norm(std::vector<double> arg);

std::vector<double> operator*(double scalar, const std::vector<double>& vec);
std::vector<double> operator+(const std::vector<double>& vec1, const std::vector<double>& vec2);
std::vector<double> operator-(const std::vector<double>& vec1, const std::vector<double>& vec2);


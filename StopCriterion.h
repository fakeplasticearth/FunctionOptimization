#pragma once
#include "OptInfo.h"
#include <vector>

class StopCriterion
{
protected:
    unsigned int max_n_iter;

    StopCriterion();
    StopCriterion(unsigned int max_n_iter_);
    StopCriterion(const StopCriterion& other);
    StopCriterion(StopCriterion&& other) noexcept;

    void swap(StopCriterion& other) noexcept;
public:
    virtual bool criterion(OptInfo* opt_info) = 0;
    bool operator()(OptInfo* opt_info);
    virtual ~StopCriterion() {};
};


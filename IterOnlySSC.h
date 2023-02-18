#pragma once
#include "StopCriterion.h"
#include "StohOptInfo.h"

class IterOnlySSC : public StopCriterion{
public:
    IterOnlySSC();
    IterOnlySSC(unsigned int max_n_iter_);
    IterOnlySSC(const IterOnlySSC& other);
    IterOnlySSC(IterOnlySSC&& other) noexcept;

    void swap(IterOnlySSC& other) noexcept;
    IterOnlySSC& operator=(IterOnlySSC other) noexcept;
    bool criterion(OptInfo* opt_info);
    ~IterOnlySSC();
};


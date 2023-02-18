#pragma once
#include "StopCriterion.h"
#include "StohOptInfo.h"

class IterAfterImpSSC : public StopCriterion {
private:
    unsigned int max_n_iter_after_imp;
public:
    IterAfterImpSSC();
    IterAfterImpSSC(unsigned int max_n_iter_, unsigned int max_n_iter_after_imp_);
    IterAfterImpSSC(const IterAfterImpSSC& other);
    IterAfterImpSSC(IterAfterImpSSC&& other) noexcept;

    void swap(IterAfterImpSSC& other) noexcept;
    IterAfterImpSSC& operator=(IterAfterImpSSC other) noexcept;
    bool criterion(OptInfo* status);
    unsigned int get_max_n_iter_after_imp() const;
    ~IterAfterImpSSC();
};


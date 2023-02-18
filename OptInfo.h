#pragma once
#include <stdexcept>

class OptInfo {
protected:
    unsigned int n_iter;
public:
    OptInfo();
    OptInfo(const OptInfo& other);
    OptInfo(OptInfo&& other) noexcept;
    OptInfo(unsigned int n_iter_);

    void swap(OptInfo& other) noexcept;
    OptInfo& operator=(OptInfo other);
    unsigned int get_n_iter() const;
    virtual ~OptInfo() {};
};


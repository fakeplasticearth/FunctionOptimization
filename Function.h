#pragma once
#include <vector>

class Function
{
protected:
    unsigned int n_dim = 0; // размерность аргумента 
    double (*func)(std::vector<double>) = nullptr; // Указатель на функцию
public:
    Function(); // Базовый конструктор
    Function(double (*func_)(std::vector<double>), unsigned int n_dim_);
    Function(const Function& other); // Конструктор копирования
    Function(Function&& other) noexcept;

    Function& operator=(Function other);
    void swap(Function& other) noexcept;
    double operator()(const std::vector<double>& argument) const;
    void set_func(double (*new_func)(std::vector<double>), unsigned int new_n_dim);
    unsigned int get_n_dim() const;
    ~Function();
};




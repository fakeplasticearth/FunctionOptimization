#pragma once
#include<vector>

class MDFunction
{
protected:
    unsigned int arg_n_dim = 0; // размерность аргумента
    std::vector<double> (*func)(std::vector<double>) = nullptr; // Указатель на функцию
public:
    MDFunction(); // Базовый конструктор
    MDFunction(std::vector<double>(*func_)(std::vector<double>), unsigned int arg_n_dim_);
    MDFunction(const MDFunction& other); // Конструктор копирования
    MDFunction(MDFunction&& other) noexcept;

    MDFunction& operator=(MDFunction other);
    void swap(MDFunction& other) noexcept;
    std::vector<double> operator()(const std::vector<double>& argument) const;
    void set_func(std::vector<double> (*new_func)(std::vector<double>), unsigned int new_arg_n_dim);
    unsigned int get_arg_n_dim() const;
    ~MDFunction();
};


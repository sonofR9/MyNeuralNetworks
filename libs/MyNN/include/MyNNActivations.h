#ifndef My_NN_Activations
#define My_NN_Activations

#include <cmath>

namespace MyNN
{
    //--------------------------------------Activations--------------------------------
    template<typename T> //type of signals and internal param, e.g. float, double
    struct BaseActivation
    {
        virtual T calculate(T input){return input;}
        virtual T calculate_feedback(T input){return input;}
    };

    template<typename T>
    struct ReLu: public BaseActivation
    {
        T calculate(T input); 
        T calculate_feedback(T input);
    };

    template<typename T>
    struct SameOutput: public BaseActivation
    {
        T calculate(T input); 
        T calculate_feedback(T input);
    };

    template<typename T>
    struct Sigmoid: public BaseActivation
    {
        T calculate(T input); 
        T calculate_feedback(T input);
    };

    template<typename T>
    struct SoftMax: public BaseActivation
    {
        T calculate(T input, T total_layer_exp_input); 
        T calculate_feedback(T input);
        T calculate_feedback(T input1, T input2, bool same);
    };
}

#endif //My_NN_Activations
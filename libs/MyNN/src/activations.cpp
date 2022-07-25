#include <MyNNActivations.h>

namespace MyNN
{
//---------------------------------------------------------------------------------
//--------------------------------------Activations--------------------------------
//---------------------------------------------------------------------------------
    template<typename T>
    T ReLu<T>::calculate (T input)
    {
        return (input<=0 ? 0: input);
    }

    template<typename T>
    T ReLu<T>::calculate_feedback (T input)
    {
        return (input<=0 ? 0: 1);
    }

    template<typename T>
    T SameOutput<T>::calculate (T input)
    {
        return (input);
    }

    template<typename T>
    T SameOutput<T>::calculate_feedback (T input)
    {
        return (input);
    }

    template<typename T>
    T Sigmoid<T>::calculate (T input)
    {
        return (1/(1+std::exp(-input)));
    }

    template<typename T>
    T Sigmoid<T>::calculate_feedback (T input)
    {
        return (calculate(input)*(1-calculate(input)));
    }

    template<typename T>
    T SoftMax<T>::calculate (T input, T total_layer_exp_input)
    {
        return std::exp(input)/total_layer_exp_input;
    }

    template<typename T>
    T SoftMax<T>::calculate_feedback (T output)
    {
        return (output*(1-output));
    }

    template<typename T>
    T SoftMax<T>::calculate_feedback (T output1, T output2, bool same)
    {
        if (same)
        {
            return (output1*(1-output1));
        }
        else
        {
            return (output1*(-output2));
        }
    }
}
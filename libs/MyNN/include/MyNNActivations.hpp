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
        virtual T calculate_feedback(T input, T feedback){return input;}
    };

    template<typename T>
    struct ReLu: public BaseActivation<T>
    {
        T calculate(T input); 
        T calculate_feedback(T input, T feedback);
    };

    template<typename T>
    struct SameOutput: public BaseActivation<T>
    {
        T calculate(T input); 
        T calculate_feedback(T input, T feedback);
    };

    template<typename T>
    struct Sigmoid: public BaseActivation<T>
    {
        T calculate(T input); 
        T calculate_feedback(T input, T feedback);
    };

    template<typename T>
    struct SoftMax: public BaseActivation<T>
    {
        T calculate(T input, T total_layer_exp_input); 
        T calculate_feedback(T output, T feedback);
        T calculate_feedback(T output1, T output2, T feedback, bool same);
    };

//---------------------------------------------------------------------
//------------------------------definitions----------------------------
//---------------------------------------------------------------------
    template<typename T>
    T ReLu<T>::calculate (T input)
    {
        return (input<=0 ? input/5: input);
    }

    template<typename T>
    T ReLu<T>::calculate_feedback (T input, T feedback)
    {
        return (input<=0 ? feedback/5: feedback);
    }

    template<typename T>
    T SameOutput<T>::calculate (T input)
    {
        return (input);
    }

    template<typename T>
    T SameOutput<T>::calculate_feedback (T input, T feedback)
    {
        return (feedback);
    }

    template<typename T>
    T Sigmoid<T>::calculate (T input)
    {
        return (1/(1+std::exp(-input)));
    }

    template<typename T>
    T Sigmoid<T>::calculate_feedback (T input, T feedback)
    {
        return (calculate(input)*(1-calculate(input))*feedback);
    }

    template<typename T>
    T SoftMax<T>::calculate (T input, T total_layer_exp_input)
    {
        return std::exp(input)/total_layer_exp_input;
    }

    template<typename T>
    T SoftMax<T>::calculate_feedback (T output, T feedback)
    {
        return (output*(1-output)*feedback);
    }

    template<typename T>
    T SoftMax<T>::calculate_feedback (T output1, T output2,  T feedback, bool same)
    {
        if (same)
        {
            return (output1*(1-output1)*feedback);
        }
        else
        {
            return (output1*(-output2)*feedback);
        }
    }

}

#endif //My_NN_Activations
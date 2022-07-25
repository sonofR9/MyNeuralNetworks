#include <MyNNNeurons.h>

namespace MyNN
{
//----------------------------------------------------------------------------
//---------------------------------Neurons------------------------------------
//----------------------------------------------------------------------------

//--------------------------------BaseNeuron----------------------------------
    template <typename T>
    void BaseNeuron<T>::change_error(T error_diff)
    {
        error+=error_diff;
    }

    template <typename T>
    void BaseNeuron<T>::change_input(T input_diff)
    {
        input+=input_diff;
    }

    template <typename T>
    void BaseNeuron<T>::nullify_input()
    {
        input=0;
    }

    template <typename T>
    void BaseNeuron<T>::nullify_error()
    {
        error=0;
    }

    template <typename T>
    BaseNeuron<T>::BaseNeuron(): error(0), input(0) {};

//------------------------------SimpleNeuron---------------------------------

    template <typename T, class activation>
    SimpleNeuron<T, activation>::SimpleNeuron(): BaseNeuron() {};

    template <typename T, class activation>
    T SimpleNeuron<T, activation>::get_response()
    {
        activation A;
        return A.calculate(input);
    }

    template <typename T, class activation>
    T SimpleNeuron<T, activation>::get_feedback()
    {
        activation A;
        return A.calculate_feedback(input);
    }

//-------------------------SoftMaxSimpleNeuron---------------------------------
    template <typename T>
    SimpleNeuron<T, SoftMax<T>>::SimpleNeuron():
            BaseNeuron(), total_layer_exp_input(0){};

    template <typename T>
    T SimpleNeuron<T, SoftMax<T>>::get_exp_input ()
    {
        return std::exp(input);
    }

    template <typename T>
    T SimpleNeuron<T, SoftMax<T>>::get_response ()
    {
        SoftMax<T> A;
        return (A.calculate(input, total_layer_exp_input));
     } 

    template <typename T>
    T SimpleNeuron<T, SoftMax<T>>::get_feedback()
    {
        SoftMax<T> A;
        return (A.calculate_feedback(get_response, total_layer_exp_input));
    } 
    
    template <typename T>
    void SimpleNeuron<T, SoftMax<T>>::set_total_input(T sum_of_all_layer_exp_inputs)
    {
        total_layer_exp_input=sum_of_all_layer_exp_inputs;
    }
}
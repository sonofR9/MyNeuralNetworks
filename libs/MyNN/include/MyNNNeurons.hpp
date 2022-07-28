#ifndef My_NN_Neurons
#define My_NN_Neurons

#include "MyNNActivations.hpp"

namespace MyNN
{
//----------------------------------------Neurons------------------------------------
    template<typename T> //type of signals and internal param, e.g. float, double
    class BaseNeuron
    {
        public:
        BaseNeuron();
        virtual T get_response(){return 0;} 
        virtual T get_feedback(){return 0;}
        T get_error();
        virtual ~BaseNeuron() = default;

        void change_error(T error_diff);
        void change_input(T input_diff);
        void nullify_input();
        void nullify_error();

        //for compatability with softmax((
        virtual T get_exp_input(){return 0;}
        virtual T get_feedback_with_correlation(std::vector<T> &outputs, std::vector<T> &errors){return 0;}
        virtual void set_total_input(T sum_of_all_layer_exp_inputs){;}

        protected:
        T error;
        T input;
    };

    template<typename T, class activation>
    struct SimpleNeuron: public BaseNeuron<T>
    {
        SimpleNeuron();
        T get_response(); 
        T get_feedback();
        ~SimpleNeuron() = default;
    };

    template<typename T>
    struct SimpleNeuron<T, SoftMax<T>>: public BaseNeuron<T>
    {
        SimpleNeuron();
        T get_exp_input();
        T get_response();
        T get_feedback_with_correlation(std::vector<T> &outputs, std::vector<T> &errors);

        void set_total_input(T sum_of_all_layer_exp_inputs);
        private:
        T total_layer_exp_input;
    };

//----------------------------------------------------------------------------
//--------------------------------Definitions------------------------------------
//----------------------------------------------------------------------------

//--------------------------------BaseNeuron----------------------------------
    template <typename T>
    T BaseNeuron<T>::get_error()
    {
        return error;
    }

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
        return A.calculate_feedback(input, error);
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
    T SimpleNeuron<T, SoftMax<T>>::get_feedback_with_correlation(std::vector<T> &outputs, std::vector<T> &errors)
    {
        SoftMax<T> A;
        T feedback{0};
        for (size_t i{0}; i<outputs.size(); ++i)
            feedback+=A.calculate_feedback(outputs[i], get_response(), errors[i], false);
        feedback-=A.calculate_feedback(get_response(), get_response(), error, false);
        feedback+=A.calculate_feedback(get_response(), get_response(), error, true);
        //softmax i-neuron feedback by j-th neuron
        return (feedback);
    } 
        
    template <typename T>
    void SimpleNeuron<T, SoftMax<T>>::set_total_input(T sum_of_all_layer_exp_inputs)
    {
        total_layer_exp_input=sum_of_all_layer_exp_inputs;
    }
}
#endif //My_NN_Neurons
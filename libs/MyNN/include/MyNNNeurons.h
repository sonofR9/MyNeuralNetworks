#ifndef My_NN_Neurons
#define My_NN_Neurons

#include <MyNNActivations.h>

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
        virtual ~BaseNeuron() = default;

        void change_error(T error_diff);
        void change_input(T input_diff);
        void nullify_input();
        void nullify_error();

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
    struct SimpleNeuron<T, SoftMax<T>>: public BaseNeuron
    {
        SimpleNeuron();
        T get_exp_input();
        T get_response();
        T get_feedback();

        void set_total_input(T sum_of_all_layer_exp_inputs);
        private:
        T total_layer_exp_input;
    };
}
#endif //My_NN_Neurons
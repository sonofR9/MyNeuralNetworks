#include <vector>
#include <map>
#include <utility>
#include <algorithm>

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

//----------------------------------------Layers-----------------------------------
    template<typename T>
    class BaseLayer
    //neurons identified by their number
    {
        public:
        BaseLayer(int num_of_neurons);

        virtual void propagate_signal(BaseLayer<T> &next_layer);
        virtual void backpropagate_error(BaseLayer<T> &previous_layer){;}
        virtual void update_weights(BaseLayer<T> &previous_layer, T learning_rate){;}
        virtual std::vector<T> get_output();

        void nullify_neurons();
        void change_neuron_input(int num, T input_diff);
        void change_neuron_error(int num, T error_diff);

        int get_size_of_layer();
        virtual void add_weight_in(int neuron_from, int neuron_to, T* weight){;}
        virtual void add_weights_out(BaseLayer<T> &next_layer, int size_of_next_layer){;}
        virtual ~BaseLayer();

        private:       

        protected:
        std::vector<BaseNeuron<T>*> neurons;
        std::vector<std::vector<std::pair<int, T*>>> weights_in;
        //weight between i neuron in this layer and [i][j].first in previous
        std::vector<std::vector<std::pair<int, T*>>> weights_out;
    };

    template<typename T, class activation>
    class DenseLayer: public BaseLayer<T>
    {
        public:
        DenseLayer(int num_of_neurons);

        void backpropagate_error(BaseLayer<T> &previous_layer);
        void update_weights(BaseLayer<T> &previous_layer, T learning_rate);

        void add_weights_out(BaseLayer<T> &next_layer, int size_of_next_layer);
        ~DenseLayer() = default;
    };

    template<typename T>
    class DenseLayer<T, SoftMax<T>>: public BaseLayer<T>
    {
        public:
        DenseLayer(int num_of_neurons);

        void propagate_signal(BaseLayer<T> &next_layer);
        void backpropagate_error(BaseLayer<T> &previous_layer);
        void update_weights(BaseLayer<T> &previous_layer, T learning_rate);

        void add_weights_out(BaseLayer<T> &next_layer, int size_of_next_layer);
        ~DenseLayer() = default;

        private:
        std::vector<SimpleNeuron<T, SoftMax<T>>*> neurons;
        T get_neuron_error(int num);
    };
//------------------------------------FirstLayer---------------------------------
    template<typename T>
    class FirstLayer: public BaseLayer<T>
    {
        public:
        FirstLayer(int input_size);

        void set_input(std::vector<T>& input);
        void add_weights_out(BaseLayer<T> &next_layer, int size_of_next_layer);

        ~FirstLayer() = default;
    };
//-------------------------------------LastLayer---------------------------------
    template<typename T>
    class LastLayerBase: public BaseLayer<T>
    {
        public:
        LastLayerBase(int num_of_neurons):{};

        virtual std::vector<T> get_output();
        void set_errors(std::vector<T> &errors);
        virtual void backpropagate_error(BaseLayer<T> &previous_layer){;}

        virtual ~LastLayerBase() = default;
    };

    template<typename T, class Activation>
    class LastLayer: public LastLayerBase<T>
    {
        public:
        LastLayer(int num_of_neurons):{};

        std::vector<T> get_output();
        void backpropagate_error(BaseLayer<T> &previous_layer){;}

        ~LastLayer() = default;
    };

    template<typename T>
    class LastLayer<T, SoftMax<T>>: public LastLayerBase<T>
    {
        public:
        LastLayer(int num_of_neurons);

        std::vector<T> get_output();
        void backpropagate_error(BaseLayer<T> &previous_layer);

        ~LastLayer() = default;
    };

    template<typename T>
    class LastLayer<T, SameOutput<T>>: public LastLayerBase<T>
    {
        public:
        LastLayer(int num_of_neurons);

        std::vector<T> get_output();
        void backpropagate_error(BaseLayer<T> &previous_layer);

        ~LastLayer() = default;
    };

//-----------------------------------------Model-----------------------------------

    template<typename T, class losse>
    class Model
    {
        public:

        Model(int input_size_);
        template<class layer> void add_layer(int num_of_neurons);
        template<class layer> void add_last_layer(int num_of_neurons);

        void backpropagate(std::vector<T> &expected_outputs);
        std::vector<T> get_output(std::vector<T>& input);

        ~Model();

        private:

        int input_size, output_size;
        int num_of_layers;
        T learning_rate;
        FirstLayer<T>* first_layer;
        LastLayerBase<T>* last_layer;
        std::vector<BaseLayer<T>*> layers;
    };

//----------------------------------------Losses---------------------------------
    template <typename T>
    struct BaseLosse
    {
        public:
        virtual std::vector<T> backpropagate_errors 
            (std::vector<T> &outputs, std::vector<T> &expected_outputs) {;}
        virtual std::vector<T> get_errors 
            (std::vector<T> &outputs, std::vector<T> &expected_outputs) {;}
    };

    template <typename T>
    struct MeanSquarredError: public BaseLosse
    {
        public:
        virtual std::vector<T> backpropagate_errors 
            (std::vector<T> &outputs, std::vector<T> &expected_outputs);
        virtual std::vector<T> get_errors 
            (std::vector<T> &outputs, std::vector<T> &expected_outputs);
    };

    template <typename T>
    struct MeanAbsoluteError: public BaseLosse
    {
        public:
        virtual std::vector<T> backpropagate_errors 
           (std::vector<T> &outputs, std::vector<T> &expected_outputs) {;}
        virtual std::vector<T> get_errors 
           (std::vector<T> &outputs, std::vector<T> &expected_outputs) {;}
    };

    template <typename T>
    struct BinaryCrossentropy: public BaseLosse
    {
        public:
        virtual std::vector<T> backpropagate_errors 
           (std::vector<T> &outputs, std::vector<T> &expected_outputs) {;}
        virtual std::vector<T> get_errors 
           (std::vector<T> &outputs, std::vector<T> &expected_outputs) {;}
    };
}


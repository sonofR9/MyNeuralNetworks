#ifndef My_NN_Layers
#define My_NN_Layers

#include <vector>
#include <utility>

#include <MyNNNeurons.h>

namespace MyNN
{
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
}

#endif //My_NN_Layers
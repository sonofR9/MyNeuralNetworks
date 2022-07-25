#ifndef My_NN_Model
#define My_NN_Model

#include <vector>

#include <MyNNLayers.h>
#include <MyNNLosses.h>

namespace MyNN
{
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
}

#endif //My_NN_Model


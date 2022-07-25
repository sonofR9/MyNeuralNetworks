#include <MyNNModel.h>

namespace MyNN
{
//---------------------------------------------------------------------------------
//-----------------------------------------Model-----------------------------------
//---------------------------------------------------------------------------------
    template<typename T, class losse>
    Model<T, losse>::Model(int input_size_):input_size{input_size_}, output_size{0},
        num_of_layers{1} 
    {
        first_layer = new FirstLayer(input_size_);
        layers.push_back(first_layer);
    }

    template<typename T, class losse>
    template<class layer>
    void Model<T, losse>::add_layer(int num_of_neurons)
    {
        layers.push_back(new layer(num_of_neurons));
        layers[num_of_layers-1]->add_weights_out(*(layers[num_of_layers]), num_of_neurons);
        num_of_layers++;
    }

    template<typename T, class losse>
    template<class layer>
    void Model<T, losse>::add_layer(int num_of_neurons)
    {
        last_layer = new layer(num_of_neurons);
        layers.push_back(last_layer);
        layers[num_of_layers-1]->add_weights_out(*(layers[num_of_layers]), num_of_neurons);
        num_of_layers++;
    }

    template<typename T, class losse>
    void Model<T, losse>::backpropagate(std::vector<T> &expected_outputs)
    {
        losse losse_obj;
        std::vector<T> errors = losse_obj.backpropagate_errors(last_layer->get_output(), expected_outputs);
        last_layer->set_errors(errors);   
        for (int i{layers.size()-1}; i>1; --i)
        {
            layers[i]->backpropagate_error(*layers[i-1]);
            layers[i]->update_weights(*layers[i-1], learning_rate);
        }
    }

    template<typename T, class losse>
    std::vector<T> Model<T,losse>::get_output(std::vector<T>& input)
    {
        std::vector<T> output;
        first_layer->set_input(input);
        for (int i{0}; y<layers.size()-1; ++i)
        {
            layers[i]->propagate_signal(*layers[i+1]);
        }
        output = last_layer->get_ouput();
        return output;
    }

    template<typename T, class losse>
    Model<T, losse>::~Model()
    {
        for (int i{0}; i<num_of_layers; ++i)
            delete layers[i];
        layers.clear();        
    }
}
#include <MyNNLayers.h>

namespace MyNN
{
//---------------------------------------------------------------------------------
//----------------------------------------Layers-----------------------------------
//---------------------------------------------------------------------------------
    
//--------------------------------------BaseLayer----------------------------------
    template <typename T>
    BaseLayer<T>::BaseLayer(int num_of_neurons): 
    {
        std::vector<std::pair<int, T*>> buff{int, T*>{nullptr, nullptr}};
        weights_in{num_of_neurons, buff};
        weights_out{num_of_neurons, buff};
        buff.clear();
    }

    template <typename T>
    int BaseLayer<T>::get_size_of_layer()
    {
        return static_cast<int>(neurons.size());
    }
    
    template <typename T>
    void BaseLayer<T>::change_neuron_input(int num, T input_diff)
    {
        neurons[num]->change_input(input_diff);
    }

    template <typename T>
    void BaseLayer<T>::change_neuron_error(int num, T error_diff)
    {
        neurons[num]->change_error(error_diff);
    }

    template <typename T>
    void BaseLayer<T>::nullify_neurons()
    {
        for (auto neuron: neurons)
        {
            neuron->nullify_error();
            neuron->nullify_input();
        }
    }

    template <typename T>
    void BaseLayer<T>::propagate_signal(BaseLayer<T> &next_layer)
    {
        T response;
        for (int i{0}; i<get_size_of_layer(); ++i)
        {
            response = neurons[i]->get_response();
            size_t j{0};
            while (j<weights_out.size() && weights_out[i][j].first!=-1)
            {
                next_layer.change_neuron_input(weights_out[i][j].first,
                        response*weights_out[i][j].second); 
            }
        }
    }

    template <typename T>
    std::vector<T> BaseLayer<T>::get_output()
    {
        std::vector<T> output;
        for (int i{0}; i<get_size_of_layer(); ++i)
            output.push_back(neurons[i]->get_response());
        return output;
    }

    template <typename T>
    void BaseLayer<T>::add_weight_in
    (int neuron_from, int neuron_to, T* weight)
    {
        if (weights_in[neuron_to][0].first==-1) 
        {
            weights_in[neuron_to][0].first = neuron_from;
            weights_in[neuron_to][0].second = weight;        
        }
        else
            weights_in[neuron_to].push_back(std::pair<int, T*>{neuron_from, weight});
    }

    template <typename T>
    BaseLayer<T>::~BaseLayer()
    {
        int i = get_size_of_layer()-1;
        while (i>=0)
        {
            delete neurons[i];
            --i;
        }
        neurons.clear();
        int i = get_size_of_layer()-1;
        while (i>=0)
        {
            int j = static_cast<int>(weights_in[i].size())-1;
            while (j>=0)
            {
                delete weights_in[i][j].second;
                --j;
            }
            --i;
        }
        weights_in.clear();
        weights_out.clear();
    }

//---------------------------------------DenseLayer---------------------------------
    template <typename T, class activation>
    DenseLayer<T, activation>::DenseLayer(int num_of_neurons):
    BaseLayer(num_of_neurons) 
    {
        for (int i{0}; i<num_of_neurons; ++i)
            neurons.push_back(new SimpleNeuron<T,activation>);
    };

    template <typename T, class activation>
    void DenseLayer<T, activation>::backpropagate_error(BaseLayer<T> &previous_layer)
    {
        T feedback;
        for (int i{0}; i<get_size_of_layer(); ++i)
        {
            feedback = neurons[i]->get_feedback();
            size_t j{0};
            while (j<weights_in.size() && weights_in[i][j].first!=-1)
            {
                previous_layer.change_neuron_error(weights_in[i][j].first,
                        feedback*weights_in[i][j].second); 
            }
        }
    }

    template <typename T, class activation>
    void DenseLayer<T, activation>::update_weights
        (BaseLayer<T> &previous_layer, T learning_rate)
    {
        T feedback;
        T output_previous;
        for (int i{0}; i<get_size_of_layer(); ++i)
        {
            feedback = neurons[i]->get_feedback();
            size_t j{0};
            while (j<weights_in.size() && weights_in[i][j].first!=-1)
            {
                output_previous = previous_layer.get_ouput(weights_in[i][j].first);
                weights_in[i][j].second += -output_previous*feedback*learning_rate;
            }
        }
    }

    template <typename T, class activation>
    void DenseLayer<T, activation>::add_weights_out 
        (BaseLayer<T> &next_layer, int size_of_next_layer)
    {
        T* buff;
        for (int i{0}; i<get_size_of_layer(); ++i)
        {
            weights_out[i][0].first = 0;
            weights_out[i][0].second = new int(0);
            for (int j{1}; j<size_of_next_layer; ++j)
            {
                buff = new T(0)
                next_layer.add_weight_in(i, j, buff);
                weights_out[i].push_back(std::pair<int, T*>{j, buff});
            }
        }
    }
//------------------------------------SoftMaxDenseLayer-----------------------------
    template <typename T>
    DenseLayer<T, SoftMax<T>>::DenseLayer(int num_of_neurons):
    BaseLayer(num_of_neurons)
    {
        for (int i{0}; i<num_of_neurons; ++i)
        {
            neurons.push_back(new SimpleNeuron<T,SoftMax<T>>);
            BaseLayer::neurons.push_back(neurons[i]);
        }
    };

    template <typename T>
    T DenseLayer<T, SoftMax<T>>::get_neuron_error(int num)
    {
        T output_num = neurons[num]->get_response();
        MyNN::SoftMax<T> soft_max;
        T error {0};
        bool same;
        for (int i{0}; i<get_size_of_layer(); ++i)
        {
            T output_i = neurons[i]->get_response();
            same = (i==num);
            error+=soft_max.calculate_feedback(output_num, output_i, same);
        }
        return error;
    }

    template <typename T>
    void DenseLayer<T, SoftMax<T>>::propagate_signal(BaseLayer<T> &next_layer)
    {
        T total_exp_input{0};
        for (int i{0}; i<get_size_of_layer(); ++i)
        {
            total_exp_input += neurons[i]->get_exp_input();
        }
        for (int i{0}; i<get_size_of_layer(); ++i)
        {
            neurons[i]->set_total_input(sum_of_all_layer_exp_inputs);
        }
        T response;
        for (int i{0}; i<get_size_of_layer(); ++i)
        {
            response = neurons[i]->get_response();
            size_t j{0};
            while (j<weights_out.size() && weights_out[i][j].first!=-1)
            {
                next_layer.change_neuron_input(weights_out[i][j].first,
                        response*weights_out[i][j].second); 
            }
        }
    }

    template <typename T>
    void DenseLayer<T, SoftMax<T>>::backpropagate_error(BaseLayer<T> &previous_layer)
    {
        T feedback;
        for (int i{0}; i<get_size_of_layer(); ++i)
        {
            feedback = get_neuron_error(i);
            size_t j{0};
            while (j<weights_in.size() && weights_in[i][j].first!=-1)
            {
                previous_layer.change_neuron_error(weights_in[i][j].first,
                        feedback*weights_in[i][j].second); 
                ++j;
            }
        }
    }

    template <typename T>
    void DenseLayer<T, SoftMax<T>>::update_weights
        (BaseLayer<T> &previous_layer, T learning_rate)
    {
        T feedback;
        T output_previous;
        for (int i{0}; i<get_size_of_layer(); ++i)
        {
            feedback = get_neuron_error(i);
            size_t j{0};
            while (j<weights_in.size() && weights_in[i][j].first!=-1)
            {
                output_previous = previous_layer.get_ouput(weights_in[i][j].first);
                weights_in[i][j].second += -output_previous*feedback*learning_rate;
            }
        }
    }

    template <typename T>
    void DenseLayer<T, SoftMax<T>>::add_weights_out 
        (BaseLayer<T> &next_layer, int size_of_next_layer)
    {
        T* buff;
        for (int i{0}; i<get_size_of_layer(); ++i)
        {
            weights_out[i][0].first = 0;
            weights_out[i][0].second = new int(0);
            for (int j{1}; j<size_of_next_layer; ++j)
            {
                buff = new T(0)
                next_layer.add_weight_in(i, j, buff);
                weights_out[i].push_back(std::pair<int, T*>{j, buff});
            }
        }
    }
//--------------------------------------FirstLayer------------------------------------
    template<typename T>
    FirstLayer<T>::FirstLayer(int input_size): 
        BaseLayer(input_size) 
    {
        for (int i{0}; i<input_size; ++i)
            neurons.push_back(new SimpleNeuron<T,BaseActivation<T>>);
    }

    template<typename T>
    void FirstLayer<T>::set_input(std::vector<T>& input)
    {
        nullify_neurons();
        for (int i{0}; i<get_size_of_layer; ++i)
            change_neuron_input(i, input[i]);
    }

    template <typename T>
    void FirstLayer<T>::add_weights_out(BaseLayer<T> &next_layer, int size_of_next_layer)
    {
        for (int i{0}; i<get_size_of_layer(); ++i)
        {
            weights_out[i][0].first = i;
            weights_out[i][0].second = new int(0);
            next_layer.add_weight_in(i, i, buff);
        }
    }
//---------------------------------------LastLayer------------------------------------
    template<typename T>
    void LastLayerBase<T>::set_errors(std::vector<T>& errors)
    {
        for (int i{0}; i<get_size_of_layer(); ++i)
            neurons[i]->change_error(errors[i]);
    }
    
    template<typename T>
    std::vector<T> LastLayerBase<T>::get_output()
    {
        std::vector<T> empty; 
        return empty;
    }

    template<typename T, class activation>
    std::vector<T> LastLayer<T, activation>::get_output()
    {
        std::vector<T> empty; 
        return empty;
    }
//------------------------------------LastLayer<Same>---------------------------------
    template <typename T>
    LastLayer<T, SameOutput<T>>::LastLayer(int num_of_neurons):
    BaseLayer(num_of_neurons) 
    {
        for (int i{0}; i<num_of_neurons; ++i)
            neurons.push_back(new SimpleNeuron<T,SameOutput<T>>);
    };

    template <typename T>
    std::vector<T> LastLayer<T, SameOutput<T>>::get_output()
    {
        std::vector<T> output;
        for (int i{0}; i<get_size_of_layer(); ++i)
        {
            output.push_back(neurons[i]->get_response());

        }
    }

    template <typename T>
    void LastLayer<T,SameOutput<T>>::backpropagate_error(BaseLayer<T> &previous_layer)
    {
        T feedback;
        for (int i{0}; i<get_size_of_layer(); ++i)
        {
            feedback = neurons[i]->get_feedback();
            size_t j{0};
            while (j<weights_in.size() && weights_in[i][j].first!=-1)
            {
                previous_layer.change_neuron_error(weights_in[i][j].first,
                        feedback*weights_in[i][j].second); 
            }
        }
    }
//-----------------------------------LastLayer<SoftMax>---------------------------------
    template <typename T>
    LastLayer<T, SoftMax<T>>::LastLayer(int num_of_neurons):
    BaseLayer(num_of_neurons) 
    {
        for (int i{0}; i<num_of_neurons; ++i)
            neurons.push_back(new SimpleNeuron<T,SoftMax<T>>);
    };

    template <typename T>
    std::vector<T> LastLayer<T, SoftMax<T>>::get_output()
    {
        std::vector<T> output;
        T total_exp_input{0};
        for (int i{0}; i<get_size_of_layer(); ++i)
            total_exp_input += neurons[i]->get_exp_input();
        for (int i{0}; i<get_size_of_layer(); ++i)
            neurons[i]->set_total_input(sum_of_all_layer_exp_inputs);
        for (int i{0}; i<get_size_of_layer(); ++i)
            output.push_back(neurons[i]->get_response());
    }

    template <typename T>
    void LastLayer<T,SoftMax<T>>::backpropagate_error(BaseLayer<T> &previous_layer)
    {
        T feedback;
        for (int i{0}; i<get_size_of_layer(); ++i)
        {
            T output_num = neurons[i]->get_response();
            MyNN::SoftMax<T> soft_max;
            T feedback {0};
            bool same;
            for (int z{0}; z<get_size_of_layer(); ++z)
            {
                T output_i = neurons[z]->get_response();
                same = (z==i);
                feedback+=soft_max.calculate_feedback(output_num, output_i, same);
            }
            size_t j{0};
            while (j<weights_in.size() && weights_in[i][j].first!=-1)
            {
                previous_layer.change_neuron_error(weights_in[i][j].first,
                        feedback*weights_in[i][j].second); 
                ++j;
            }
        }
    }
}
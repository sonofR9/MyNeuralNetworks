#include <MyNNLibrary.h>
#include <cmath>

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
        std::vector<T> errors = 
            losse_obj.backpropagate_errors(last_layer->get_output(), expected_outputs);
        last_layer->set_errors(errors);   
        for (int i{layers.size()-1; i>1; --i)
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

//---------------------------------------------------------------------------------
//----------------------------------------Losses-----------------------------------
//---------------------------------------------------------------------------------
    template <typename T>
    std::vector<T> MeanSquarredError<T>::backpropagate_errors(std::vector<T> &outputs, 
                                                    std::vector<T> &expected_outputs)
    {
        std::vector<T> errors_derivatives;
        for (int i{0}; i<outputs.size(); ++i)
            errors_derivatives.push_back(2*(outputs[i]-expected_outputs[i]));
        return errors_derivatives;
    }

    template <typename T>
    std::vector<T> MeanSquarredError<T>::get_errors(std::vector<T> &outputs, 
                                                std::vector<T> &expected_outputs)
    {
        std::vector<T> errors;
        for (int i{0}; i<outputs.size(); ++i)
        {
            errors.push_back(outputs[i]-expected_outputs[i]);
            errors[i]*=errors[i];
        }
        return errors;
    }
}
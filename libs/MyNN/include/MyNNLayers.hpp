#ifndef My_NN_Layers
#define My_NN_Layers

#include <vector>
#include <utility>
#include <random>

#include "MyNNNeurons.hpp"

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
        virtual void backpropagate_error(BaseLayer<T> &previous_layer);
        virtual void update_weights_in(BaseLayer<T> &previous_layer, T learning_rate);
        virtual std::vector<T> get_output();

        T get_neuron_response(int num);

        void nullify_neurons();
        void change_neuron_input(int num, T input_diff);
        void change_neuron_error(int num, T error_diff);

        int get_size_of_layer();
        virtual void add_weight_in(int neuron_from, int neuron_to, T* weight);
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
        void update_weights_in(BaseLayer<T> &previous_layer, T learning_rate);

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
        void update_weights_in(BaseLayer<T> &previous_layer, T learning_rate);

        void add_weights_out(BaseLayer<T> &next_layer, int size_of_next_layer);
        ~DenseLayer() = default;

        private:
        std::vector<SimpleNeuron<T, SoftMax<T>>*> neurons;
        T get_neuron_feedback(int num, std::vector<T> &neurons_outputs, std::vector<T> &neurons_errors);
    };
//------------------------------------FirstLayer---------------------------------
    template<typename T>
    class FirstLayer: public BaseLayer<T>
    {
        public:
        FirstLayer(int input_size);

        void backpropagate_error(BaseLayer<T> &previous_layer){;}
        void update_weights_in(BaseLayer<T> &previous_layer, T learning_rate){;}

        void set_input(std::vector<T>& input);
        void add_weights_out(BaseLayer<T> &next_layer, int size_of_next_layer);

        ~FirstLayer() = default;
    };
//-------------------------------------LastLayer---------------------------------
    template<typename T>
    class LastLayerBase: public BaseLayer<T>
    {
        public:
        LastLayerBase(int num_of_neurons);

        virtual std::vector<T> get_output();
        void set_errors(std::vector<T> &errors);
        virtual void backpropagate_error(BaseLayer<T> &previous_layer){;}

        virtual ~LastLayerBase() = default;
    };

    template<typename T, class Activation>
    class LastLayer: public LastLayerBase<T>
    {
        public:
        LastLayer(int num_of_neurons){};

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

//---------------------------------------------------------------------------------
//--------------------------------------Definitions---------------------------------
//---------------------------------------------------------------------------------

//---------------------------------------BaseLayer---------------------------------
    template <typename T>
    BaseLayer<T>::BaseLayer(int num_of_neurons)
    {
        std::vector<std::pair<int, T*>> buff;
        buff.push_back(std::pair<int, T*>{-1, nullptr});
        for (size_t i{0}; i<num_of_neurons; ++i)
        {
            weights_in.push_back(buff);
            weights_out.push_back(buff);
        }
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
        for (size_t i{0}; i<get_size_of_layer(); ++i)
        {
            response = neurons[i]->get_response();
            size_t j{0};
            while (j<weights_out[i].size() && weights_out[i][j].first!=-1)
            {
                next_layer.change_neuron_input(weights_out[i][j].first, 
                            response*(*weights_out[i][j].second)); 
                ++j;
            }
        }
    }

    template <typename T>
    void BaseLayer<T>::backpropagate_error(BaseLayer<T> &previous_layer)
    {
        T feedback;
        for (size_t i{0}; i<get_size_of_layer(); ++i)
        {
            feedback = neurons[i]->get_feedback();
            size_t j{0};
            while (j<weights_in[i].size() && weights_in[i][j].first!=-1)
            {
                previous_layer.change_neuron_error(weights_in[i][j].first,
                        feedback*(*weights_in[i][j].second)); 
                ++j;
            }
        }
    }

    template <typename T>
    void BaseLayer<T>::update_weights_in
        (BaseLayer<T> &previous_layer, T learning_rate)
    {
        T feedback;
        T output_previous;
        for (size_t i{0}; i<get_size_of_layer(); ++i)
        {
            feedback = neurons[i]->get_feedback();
            size_t j{0};
            while (j<weights_in[i].size() && weights_in[i][j].first!=-1)
            {
                output_previous = previous_layer.get_neuron_response(weights_in[i][j].first);
                (*weights_in[i][j].second) += -output_previous*feedback*learning_rate;
                ++j;
            }
        }
    }

    template <typename T>
    std::vector<T> BaseLayer<T>::get_output()
    {
        std::vector<T> output;
        for (size_t i{0}; i<get_size_of_layer(); ++i)
            output.push_back(neurons[i]->get_response());
        return output;
    }

    template <typename T>
    T BaseLayer<T>::get_neuron_response(int num)
    {
        return neurons[num]->get_response();
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
        i = get_size_of_layer()-1;
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
        for (size_t i{0}; i<num_of_neurons; ++i)
            neurons.push_back(new SimpleNeuron<T,activation>);
    };

    template <typename T, class activation>
    void DenseLayer<T, activation>::backpropagate_error(BaseLayer<T> &previous_layer)
    {
        T feedback;
        for (size_t i{0}; i<get_size_of_layer(); ++i)
        {
            feedback = neurons[i]->get_feedback();
            size_t j{0};
            while (j<weights_in[i].size() && weights_in[i][j].first!=-1)
            {
                previous_layer.change_neuron_error(weights_in[i][j].first,
                        feedback*(*weights_in[i][j].second)); 
                ++j;
            }
        }
    }

    template <typename T, class activation>
    void DenseLayer<T, activation>::update_weights_in
        (BaseLayer<T> &previous_layer, T learning_rate)
    {
        T feedback;
        T output_previous;
        for (size_t i{0}; i<get_size_of_layer(); ++i)
        {
            feedback = neurons[i]->get_feedback();
            size_t j{0};
            while (j<weights_in[i].size() && weights_in[i][j].first!=-1)
            {
                output_previous = previous_layer.get_neuron_response(weights_in[i][j].first);
                (*weights_in[i][j].second) += -output_previous*feedback*learning_rate;
                ++j;
            }
        }
    }

    template <typename T, class activation>
    void DenseLayer<T, activation>::add_weights_out 
        (BaseLayer<T> &next_layer, int size_of_next_layer)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        T* buff;
        for (size_t i{0}; i<get_size_of_layer(); ++i)
        {
            weights_out[i][0].first = 0;
            weights_out[i][0].second = new T(dis(gen));
            next_layer.add_weight_in(i, 0, weights_out[i][0].second);
            for (size_t j{1}; j<size_of_next_layer; ++j)
            {
                buff = new T(dis(gen));
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
        for (size_t i{0}; i<num_of_neurons; ++i)
        {
            neurons.push_back(new SimpleNeuron<T,SoftMax<T>>);
            BaseLayer::neurons.push_back(neurons[i]);
        }
    };

    template <typename T>
    T DenseLayer<T, SoftMax<T>>::get_neuron_feedback(int num, std::vector<T> &neurons_outputs, std::vector<T> &neurons_errors)
    {
        return neurons[num]->get_feedback_with_correlation(neurons_outputs, neurons_errors);
    }

    template <typename T>
    void DenseLayer<T, SoftMax<T>>::propagate_signal(BaseLayer<T> &next_layer)
    {
        T total_exp_input{0};
        for (size_t i{0}; i<get_size_of_layer(); ++i)
        {
            total_exp_input += neurons[i]->get_exp_input();
        }
        for (size_t i{0}; i<get_size_of_layer(); ++i)
        {
            neurons[i]->set_total_input(total_exp_input);
        }
        T response;
        for (size_t i{0}; i<get_size_of_layer(); ++i)
        {
            response = neurons[i]->get_response();
            size_t j{0};
            while (j<weights_out[i].size() && weights_out[i][j].first!=-1)
            {
                next_layer.change_neuron_input(weights_out[i][j].first,
                        response*(*weights_out[i][j].second));
                ++j;
            }
        }
    }

    template <typename T>
    void DenseLayer<T, SoftMax<T>>::backpropagate_error(BaseLayer<T> &previous_layer)
    {
        T feedback;
        std::vector<T> neurons_outputs;
        std::vector<T> neurons_errors;
        for (size_t i{0}; i<get_size_of_layer(); ++i)
            neurons_outputs.push_back(neurons[i]->get_response());
        for (size_t i{0}; i<get_size_of_layer(); ++i)
            neurons_errors.push_back(neurons[i]->get_error());
        for (size_t i{0}; i<get_size_of_layer(); ++i)
        {
            feedback = get_neuron_feedback(i, neurons_outputs, neurons_errors);
            size_t j{0};
            while (j<weights_in[i].size() && weights_in[i][j].first!=-1)
            {
                previous_layer.change_neuron_error(weights_in[i][j].first,
                        feedback*(*weights_in[i][j].second)); 
                ++j;
            }
        }
    }

    template <typename T>
    void DenseLayer<T, SoftMax<T>>::update_weights_in
        (BaseLayer<T> &previous_layer, T learning_rate)
    {
        T feedback;
        T output_previous;
        std::vector<T> neurons_outputs;
        std::vector<T> neurons_errors;
        for (size_t i{0}; i<get_size_of_layer(); ++i)
            neurons_outputs.push_back(neurons[i]->get_response());
        for (size_t i{0}; i<get_size_of_layer(); ++i)
            neurons_errors.push_back(neurons[i]->get_error());
        for (size_t i{0}; i<get_size_of_layer(); ++i)
        {
            feedback = get_neuron_feedback(i,neurons_outputs, neurons_errors);
            size_t j{0};
            while (j<weights_in[i].size() && weights_in[i][j].first!=-1)
            {
                output_previous = previous_layer.get_neuron_response(weights_in[i][j].first);
                (*weights_in[i][j].second) += -output_previous*feedback*learning_rate;
                ++j;
            }
        }
    }

    template <typename T>
    void DenseLayer<T, SoftMax<T>>::add_weights_out 
        (BaseLayer<T> &next_layer, int size_of_next_layer)
    {
        T* buff;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (size_t i{0}; i<get_size_of_layer(); ++i)
        {
            weights_out[i][0].first = 0;
            weights_out[i][0].second = new T(dis(gen));
            next_layer.add_weight_in(i, 0, weights_out[i][0].second);
            for (size_t j{1}; j<size_of_next_layer; ++j)
            {
                buff = new T(dis(gen));
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
        for (size_t i{0}; i<input_size; ++i)
            neurons.push_back(new SimpleNeuron<T,BaseActivation<T>>);
    }

    template<typename T>
    void FirstLayer<T>::set_input(std::vector<T>& input)
    {
        nullify_neurons();
        for (size_t i{0}; i<get_size_of_layer(); ++i)
            change_neuron_input(i, input[i]);
    }

    template <typename T>
    void FirstLayer<T>::add_weights_out(BaseLayer<T> &next_layer, int size_of_next_layer)
    {
        for (size_t i{0}; i<get_size_of_layer(); ++i)
        {
            weights_out[i][0].first = i;
            weights_out[i][0].second = new T(1);
            next_layer.add_weight_in(i, i, weights_out[i][0].second);
        }
    }
//---------------------------------------LastLayer------------------------------------
    template<typename T>
    LastLayerBase<T>::LastLayerBase(int num_of_neurons):
    BaseLayer(num_of_neurons) {}
    
    
    template<typename T>
    void LastLayerBase<T>::set_errors(std::vector<T>& errors)
    {
        for (size_t i{0}; i<get_size_of_layer(); ++i)
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
    LastLayerBase(num_of_neurons) 
    {
        for (size_t i{0}; i<num_of_neurons; ++i)
            neurons.push_back(new SimpleNeuron<T,SameOutput<T>>);
    };

    template <typename T>
    std::vector<T> LastLayer<T, SameOutput<T>>::get_output()
    {
        std::vector<T> output;
        for (size_t i{0}; i<get_size_of_layer(); ++i)
        {
            output.push_back(neurons[i]->get_response());
        }
        return output;
    }

    template <typename T>
    void LastLayer<T,SameOutput<T>>::backpropagate_error(BaseLayer<T> &previous_layer)
    {
        T feedback;
        for (size_t i{0}; i<get_size_of_layer(); ++i)
        {
            feedback = neurons[i]->get_feedback();
            size_t j{0};
            while (j<weights_in[i].size() && weights_in[i][j].first!=-1)
            {
                previous_layer.change_neuron_error(weights_in[i][j].first,
                        feedback*(*weights_in[i][j].second)); 
                ++j;
            }
        }
    }
//-----------------------------------LastLayer<SoftMax>---------------------------------
    template <typename T>
    LastLayer<T, SoftMax<T>>::LastLayer(int num_of_neurons):
    LastLayerBase(num_of_neurons) 
    {
        for (size_t i{0}; i<num_of_neurons; ++i)
            neurons.push_back(new SimpleNeuron<T,SoftMax<T>>);
    };

    template <typename T>
    std::vector<T> LastLayer<T, SoftMax<T>>::get_output()
    {
        std::vector<T> output;
        T total_exp_input{0};
        for (size_t i{0}; i<get_size_of_layer(); ++i)
            total_exp_input += neurons[i]->get_exp_input();
        for (size_t i{0}; i<get_size_of_layer(); ++i)
            neurons[i]->set_total_input(total_exp_input);
        for (size_t i{0}; i<get_size_of_layer(); ++i)
            output.push_back(neurons[i]->get_response());
        return output;
    }

    template <typename T>
    void LastLayer<T,SoftMax<T>>::backpropagate_error(BaseLayer<T> &previous_layer)
    {
        T feedback;
        std::vector<T> neurons_outputs;
        std::vector<T> neurons_errors;
        for (size_t i{0}; i<get_size_of_layer(); ++i)
            neurons_outputs.push_back(neurons[i]->get_response());
        for (size_t i{0}; i<get_size_of_layer(); ++i)
            neurons_errors.push_back(neurons[i]->get_error());
        for (size_t i{0}; i<get_size_of_layer(); ++i)
        {
            MyNN::SoftMax<T> soft_max;
            feedback = neurons[i]->get_feedback_with_correlation(neurons_outputs, neurons_errors);
            size_t j{0};
            while (j<weights_in[i].size() && weights_in[i][j].first!=-1)
            {
                previous_layer.change_neuron_error(weights_in[i][j].first,
                        feedback*(*weights_in[i][j].second)); 
                ++j;
            }
        }
    }
}

#endif //My_NN_Layers
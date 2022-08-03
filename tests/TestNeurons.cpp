#include "gtest/gtest.h"
#include "MyNNNeurons.hpp"

using namespace MyNN;

TEST (Neurons, simple_neuron_ReLu)
{
    BaseNeuron<double>* ReLu_neuron = new SimpleNeuron<double, ReLu<double>>;
    ReLu_neuron->change_input(10);
    EXPECT_EQ(ReLu_neuron->get_response(), 10)<<"get_response error";
    ReLu_neuron->nullify_input();
    EXPECT_EQ(ReLu_neuron->get_response(), 0)<<"nullify_input error";
    ReLu_neuron->change_input(10);
    ReLu_neuron->change_error(10);
    EXPECT_EQ(ReLu_neuron->get_feedback(), 10)<<"get feedback error";
    ReLu_neuron->nullify_error();
    EXPECT_EQ(ReLu_neuron->get_feedback(), 0)<<"nullify_error error";
}
#include "gtest/gtest.h"
#include "MyNNModel.hpp"
#include "MyNNLayers.hpp"
#include <vector>
#include <math.h>

using namespace MyNN;
using namespace std;

class TestModel: public ::testing::Test
{
    protected:
    void SetUp() override 
    {
        m_mean_sq_er_.set_learning_rate(0.01);
        //<=0.01 for SameOutput
        //>=0.01 for SoftMax
        using T = double;
        m_mean_sq_er_.template add_layer<DenseLayer<T, ReLu<T>>>(10);
        m_mean_sq_er_.template add_layer<DenseLayer<T, SoftMax<T>>>(10);
        m_mean_sq_er_.template add_layer<DenseLayer<T, ReLu<T>>>(10);
        m_mean_sq_er_.template add_last_layer<LastLayer<double, SameOutput<double>>>(2);
    }

    Model<double, MeanSquarredError<double>> m_mean_sq_er_{4};
};

//it's just showcase: fixture class here is unneccesary
TEST_F (TestModel, BasicTest)
{
    vector<double> test_input(4,1);
    vector<double> test_output;
    test_output.push_back(30);
    test_output.push_back(17);
    vector<double> output_before, output_after;
    vector<double> errors_before, errors_after;
    EXPECT_NO_FATAL_FAILURE(m_mean_sq_er_.get_output(test_input));
    output_before = m_mean_sq_er_.get_output(test_input);
    for (int i{0}; i<2; ++i)
        errors_before.push_back(abs(test_output[i]-output_before[i]));
    for (int i{0}; i<10; ++i)
    {
        output_before = m_mean_sq_er_.get_output(test_input);
        m_mean_sq_er_.backpropagate(test_output);
    }
    output_after = m_mean_sq_er_.get_output(test_input);
    for (int i{0}; i<2; ++i)
        errors_after.push_back(test_output[i]-output_after[i]);
    for (int i{0}; i<2; ++i)
        EXPECT_GE(errors_before[i], errors_after[i])<<i;
}
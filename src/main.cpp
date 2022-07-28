#include "MyNNModel.hpp"
#include "MyNNLosses.hpp"
#include "MyNNLayers.hpp"
#include "MyNNActivations.hpp"
#include <vector>

int main(void)
{
    using namespace std;
    using namespace MyNN;
    Model<double, MeanSquarredError<double>> test_model(4);
    vector<double> test_input(4,1);
    vector<double> test_output;
    test_output.push_back(30);
    test_output.push_back(17);
    vector<double> output;
    test_model.set_learning_rate(0.01);
    //<=0.01 for SameOutput
    //>=0.01 for SoftMax
    using T = double;
    test_model.template add_layer<DenseLayer<T, ReLu<T>>>(10);
    test_model.template add_layer<DenseLayer<T, SoftMax<T>>>(10);
    test_model.template add_layer<DenseLayer<T, ReLu<T>>>(10);
    test_model.template add_last_layer<LastLayer<double, SameOutput<double>>>(2);
    for (int i{0}; i<50; ++i)
    {
        output = test_model.get_output(test_input);
        test_model.backpropagate(test_output);
    }
    
    return 0;
}
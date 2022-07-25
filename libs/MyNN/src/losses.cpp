#include <MyNNLosses.h>

namespace MyNN
{
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
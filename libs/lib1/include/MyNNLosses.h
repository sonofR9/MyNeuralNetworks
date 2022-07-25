#ifndef My_NN_Losses
#define My_NN_Losses

#include <vector>

namespace MyNN
{
//----------------------------------------Losses---------------------------------
    template <typename T>
    struct BaseLosse
    {
        public:
        virtual std::vector<T> backpropagate_errors 
            (std::vector<T> &outputs, std::vector<T> &expected_outputs) {;}
        virtual std::vector<T> get_errors 
            (std::vector<T> &outputs, std::vector<T> &expected_outputs) {;}
    };

    template <typename T>
    struct MeanSquarredError: public BaseLosse
    {
        public:
        virtual std::vector<T> backpropagate_errors 
            (std::vector<T> &outputs, std::vector<T> &expected_outputs);
        virtual std::vector<T> get_errors 
            (std::vector<T> &outputs, std::vector<T> &expected_outputs);
    };

    template <typename T>
    struct MeanAbsoluteError: public BaseLosse
    {
        public:
        virtual std::vector<T> backpropagate_errors 
           (std::vector<T> &outputs, std::vector<T> &expected_outputs) {;}
        virtual std::vector<T> get_errors 
           (std::vector<T> &outputs, std::vector<T> &expected_outputs) {;}
    };

    template <typename T>
    struct BinaryCrossentropy: public BaseLosse
    {
        public:
        virtual std::vector<T> backpropagate_errors 
           (std::vector<T> &outputs, std::vector<T> &expected_outputs) {;}
        virtual std::vector<T> get_errors 
           (std::vector<T> &outputs, std::vector<T> &expected_outputs) {;}
    };
}

#endif
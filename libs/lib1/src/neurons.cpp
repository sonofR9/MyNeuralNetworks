#include <MyNNLibrary.h>

namespace MyNN
{
    template<class T> //type of signals and internal param, e.g. float, double
    class SimpleNeuron
    {
        public:

        virtual SimpleNeuron();
        virtual T get_response(); 
        virtual T get_feedback();
        virtual ~SimpleNeuron();

        void change_error(T error_diff)
        {
            error+=error_diff;
        }
        void change_input(T input_diff)
        {
            input+=input_diff;
        }
        void nullify_input()
        {
            input=0;
        }
        void nullify_error()
        {
            error=0;
        }

        private:

        T error;
        T input;

        protected:
    }
}
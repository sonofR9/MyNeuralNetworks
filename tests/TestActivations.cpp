#include "gtest/gtest.h"
#include "MyNNActivations.hpp"

using namespace MyNN;

TEST (Activations, ReLu_test)
{
    BaseActivation<double>* act_reLu = new ReLu<double>;
    for (int i{0}; i<100; ++i)
        EXPECT_EQ(act_reLu->calculate(i), 1.0*i)<<"Error: ReLu calculate positive";
    for (int i{0}; i>-100; --i)
        EXPECT_EQ(act_reLu->calculate(i), 1.0*i/5)<<"Error: ReLu calculate negative";
    for (int i{1}; i<100; ++i)
        EXPECT_EQ(act_reLu->calculate_feedback(i,1), 1)<<"Error: ReLu calculate_feedback positive"<<i;
    for (int i{0}; i>-100; --i)
        EXPECT_EQ(act_reLu->calculate_feedback(i,5), 1)<<"Error: ReLu calculate_feedback negative";
}

TEST (Activations, SameOutput_test)
{
    BaseActivation<double>* SameOutput_act = new SameOutput<double>;
    for (int i{0}; i<100; ++i)
        EXPECT_EQ(SameOutput_act->calculate(i), i)<<"Error: SameOutput calculate";
    for (int i{0}; i<100; ++i)
        EXPECT_EQ(SameOutput_act->calculate_feedback(i,1), 1)<<"Error: SameOutput calculate_feedback";
}
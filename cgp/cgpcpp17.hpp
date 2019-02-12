
// MIT License
//
// Copyright (c) 2019 degski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <experimental/fixed_capacity_vector> // https://github.com/gnzlbg/static_vector
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>

namespace cgp {

namespace functions {

// Node functions defines in CGP-Library
template<typename Real = float> Real f_add ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_sub ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_mul ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_divide ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_and ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_absolute ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_squareRoot ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_square ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_cube ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_power ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_exponential ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_sine ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_cosine ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_tangent ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_randFloat ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_constTwo ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_constOne ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_constZero ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_constPI ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_nand ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_or ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_nor ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_xor ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_xnor ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_not ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_wire ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_sigmoid ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_gaussian ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_step ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_softsign ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
template<typename Real = float> Real f_hyperbolicTangent ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept;
}

template<typename Real = float>
using FunctionPointer = Real ( * ) ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights );


template<typename Real = float>
struct FunctionSet {

    std::vector<std::string> functionNames;
    std::vector<int> maxNumInputs;
    std::vector<FunctionPointer<Real>> functions;

    template<typename ... Args>
    void addNodeFunction ( Args ... args_ ) {


    }

    int numFunctions ( ) const noexcept { return static_cast< int > ( functionNames.size ( ) ); }

    void print ( ) const noexcept {
        std::printf ( "Function Set:" );
        for ( const auto & name : functionNames ) {
            std::printf ( " %s", name );
        }
        std::printf ( " (%d)\n", numFunctions ( ) );
    }
};


template<typename Real = float>
struct Node {

    int function;
    std::vector<int> inputs;
    std::vector<Real> weights;
    int active;
    Real output;
    int maxArity;
    int actArity;
};


template<typename Real = float>
struct Chromosome {

    int numInputs;
    int numOutputs;
    int numNodes;
    int numActiveNodes;
    int arity;
    Node<Real> **nodes;
    int *outputNodes;
    int *activeNodes;
    Real fitness;
    Real *outputValues;
    FunctionSet<Real> & funcSet;
    Real *nodeInputsHold;
    int generation;

};


template<typename Real = float>
struct Parameters {

    int mu;
    int lambda;
    char evolutionaryStrategy;
    Real mutationRate;
    Real recurrentConnectionProbability;
    Real connectionWeightRange;
    int numInputs;
    int numNodes;
    int numOutputs;
    int arity;
    FunctionSet<Real> funcSet;
    Real targetFitness;
    int updateFrequency;
    int shortcutConnections;
    void ( *mutationType )( struct parameters *params, struct chromosome *chromo );
    std::string mutationTypeName;
    Real ( *fitnessFunction )( struct parameters *params, struct chromosome *chromo, struct dataSet *dat );
    std::string fitnessFunctionName;
    void ( *selectionScheme )( struct parameters *params, struct chromosome **parents, struct chromosome **candidateChromos, int numParents, int numCandidateChromos );
    std::string selectionSchemeName;
    void ( *reproductionScheme )( struct parameters *params, struct chromosome **parents, struct chromosome **children, int numParents, int numChildren );
    std::string reproductionSchemeName;
    int numThreads;

    Parameters ( const int numInputs_, const int numNodes_, const int numOutputs_, const int arity_ ) noexcept :

        mu { 1 },
        lambda { 4 },
        evolutionaryStrategy { '+' },
        mutationRate { Real { 0.05 } },
        recurrentConnectionProbability { Real { 0 } },
        connectionWeightRange { Real { 1 } },
        numInputs { numInputs_ },
        numNodes { numNodes_ },
        numOutputs { numOutputs_ },
        arity { arity_ },
        targetFitness { Real { 0 } },
        updateFrequency { 1 },
        shortcutConnections { 1 },
        mutationType { probabilisticMutation },
        mutationTypeName { "probabilistic" },
        fitnessFunction { supervisedLearning },
        fitnessFunctionName { "supervisedLearning" },
        selectionScheme { selectFittest },
        selectionSchemeName { "selectFittest" },
        reproductionScheme { mutateRandomParent },
        reproductionSchemeName { "mutateRandomParent" },
        numThreads { 1 } {

        assert ( numInputs > 0 );
        assert ( numNodes >= 0 );
        assert ( numOutputs > 0 );
        assert ( arity > 0 );
    }

    template<typename ... Args>
    void addNodeFunction ( Args ... args_ ) {
        funcSet.addNodeFunction ( std::forward<Args> ( args_ ) ...  )
        assert ( funcSet.numFunctions ( ) > 0 );
    }

    void print ( ) const noexcept {

        std::printf ( "-----------------------------------------------------------\n" );
        std::printf ( "                       Parameters                          \n" );
        std::printf ( "-----------------------------------------------------------\n" );
        std::printf ( "Evolutionary Strategy:\t\t\t(%d%c%d)-ES\n", mu, evolutionaryStrategy, lambda );
        std::printf ( "Inputs:\t\t\t\t\t%d\n", numInputs );
        std::printf ( "Nodes:\t\t\t\t\t%d\n", numNodes );
        std::printf ( "Outputs:\t\t\t\t%d\n", numOutputs );
        std::printf ( "Node Arity:\t\t\t\t%d\n", arity );
        std::printf ( "Connection weights range:\t\t+/- %f\n", connectionWeightRange );
        std::printf ( "Mutation Type:\t\t\t\t%s\n", mutationTypeName );
        std::printf ( "Mutation rate:\t\t\t\t%f\n", mutationRate );
        std::printf ( "Recurrent Connection Probability:\t%f\n", recurrentConnectionProbability );
        std::printf ( "Shortcut Connections:\t\t\t%d\n", shortcutConnections );
        std::printf ( "Fitness Function:\t\t\t%s\n", fitnessFunctionName );
        std::printf ( "Target Fitness:\t\t\t\t%f\n", targetFitness );
        std::printf ( "Selection scheme:\t\t\t%s\n", selectionSchemeName );
        std::printf ( "Reproduction scheme:\t\t\t%s\n", reproductionSchemeName );
        std::printf ( "Update frequency:\t\t\t%d\n", updateFrequency );
        std::printf ( "Threads:\t\t\t%d\n", numThreads );
        funcSet.print ( );
        std::printf ( "-----------------------------------------------------------\n\n" );
    }
};


template<typename Real = float>
struct DataSet {
    int numSamples;
    int numInputs;
    int numOutputs;
    Real **inputData;
    Real **outputData;
};


template<typename Real = float>
struct Results {
    int numRuns;
    Chromosome<Real> **bestChromosomes;
};

namespace functions {


/*
    Node function add. Returns the sum of all the inputs.
*/
template<typename Real = float> Real f_add ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    int i;
    Real sum = inputs [ 0 ];

    for ( i = 1; i < numInputs; i++ ) {
        sum += inputs [ i ];
    }

    return sum;
}

/*
    Node function sub. Returns the first input minus all remaining inputs.
*/
template<typename Real = float> Real f_sub ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    int i;
    Real sum = inputs [ 0 ];

    for ( i = 1; i < numInputs; i++ ) {
        sum -= inputs [ i ];
    }

    return sum;
}


/*
    Node function mul. Returns the multiplication of all the inputs.
*/
template<typename Real = float> Real f_mul ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    int i;
    Real multiplication = inputs [ 0 ];

    for ( i = 1; i < numInputs; i++ ) {
        multiplication *= inputs [ i ];
    }

    return multiplication;
}


/*
    Node function div. Returns the first input divided by the second input divided by the third input etc
*/
template<typename Real = float> Real f_divide ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    int i;
    Real divide = inputs [ 0 ];

    for ( i = 1; i < numInputs; i++ ) {
        divide /= inputs [ i ];
    }

    return divide;
}


/*
    Node function abs. Returns the absolute of the first input
*/
template<typename Real = float> Real f_absolute ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    return std::abs ( inputs [ 0 ] );
}


/*
    Node function sqrt.  Returns the square root of the first input
*/
template<typename Real = float> Real f_squareRoot ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    return std::sqrt ( inputs [ 0 ] );
}


/*
    Node function squ.  Returns the square of the first input
*/
template<typename Real = float> Real f_square ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    return std::pow ( inputs [ 0 ], 2 );
}


/*
    Node function cub.  Returns the cube of the first input
*/
template<typename Real = float> Real f_cube ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    return std::pow ( inputs [ 0 ], 3 );
}


/*
    Node function power.  Returns the first output to the power of the second
*/
template<typename Real = float> Real f_power ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    return std::pow ( inputs [ 0 ], inputs [ 1 ] );
}

/*
    Node function exp.  Returns the exponential of the first input
*/
template<typename Real = float> Real f_exponential ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    return std::exp ( inputs [ 0 ] );
}


/*
    Node function sin.  Returns the sine of the first input
*/
template<typename Real = float> Real f_sine ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    return std::sin ( inputs [ 0 ] );
}

/*
    Node function cos.  Returns the cosine of the first input
*/
template<typename Real = float> Real f_cosine ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    return std::cos ( inputs [ 0 ] );
}

/*
    Node function tan.  Returns the tangent of the first input
*/
template<typename Real = float> Real f_tangent ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    return std::tan ( inputs [ 0 ] );
}

/*
    Node function one.  Always returns 1
*/
template<typename Real = float> Real f_constTwo ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    return 2.0;
}

/*
    Node function one.  Always returns 1
*/
template<typename Real = float> Real f_constOne ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    return 1.0;
}

/*
    Node function one.  Always returns 0
*/
template<typename Real = float> Real f_constZero ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    return 0.0;
}

/*
    Node function one.  Always returns PI
*/
template<typename Real = float> Real f_constPI ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    return 3.141592653589793116;
}


/*
    Node function rand.  Returns a random number between minus one and positive one
*/
template<typename Real = float> Real f_randFloat ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    return std::uniform_real_distribution<Real> ( -1.0, 1.0 ) ( rgen );
}

/*
    Node function and. logical AND, returns '1' if all inputs are '1'
    else, '0'
*/
template<typename Real = float> Real f_and ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    int i;

    for ( i = 0; i < numInputs; i++ ) {

        if ( inputs [ i ] == 0.0 ) {
            return 0.0;
        }
    }

    return 1.0;
}

/*
    Node function and. logical NAND, returns '0' if all inputs are '1'
    else, '1'
*/
template<typename Real = float> Real f_nand ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    int i;

    for ( i = 0; i < numInputs; i++ ) {

        if ( inputs [ i ] == 0.0 ) {
            return 1.0;
        }
    }

    return 0.0;
}


/*
    Node function or. logical OR, returns '0' if all inputs are '0'
    else, '1'
*/
template<typename Real = float> Real f_or ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    int i;

    for ( i = 0; i < numInputs; i++ ) {

        if ( inputs [ i ] == 1.0 ) {
            return 1.0;
        }
    }

    return 0.0;
}


/*
    Node function nor. logical NOR, returns '1' if all inputs are '0'
    else, '0'
*/
template<typename Real = float> Real f_nor ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    int i;

    for ( i = 0; i < numInputs; i++ ) {

        if ( inputs [ i ] == 1.0 ) {
            return 0.0;
        }
    }

    return 1.0;
}


/*
    Node function xor. logical XOR, returns '1' iff one of the inputs is '1'
    else, '0'. AKA 'one hot'.
*/
template<typename Real = float> Real f_xor ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    int i;
    int numOnes = 0;
    int out;

    for ( i = 0; i < numInputs; i++ ) {

        if ( inputs [ i ] == 1 ) {
            numOnes++;
        }

        if ( numOnes > 1 ) {
            break;
        }
    }

    return numOnes == 1;
}

/*
    Node function xnor. logical XNOR, returns '0' iff one of the inputs is '1'
    else, '1'.
*/
template<typename Real = float> Real f_xnor ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    int i;
    int numOnes = 0;

    for ( i = 0; i < numInputs; i++ ) {

        if ( inputs [ i ] == 1 ) {
            numOnes++;
        }

        if ( numOnes > 1 ) {
            break;
        }
    }

    return numOnes != 1;
}

/*
    Node function not. logical NOT, returns '1' if first input is '0', else '1'
*/
template<typename Real = float> Real f_not ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    return inputs [ 0 ] == 0.0;
}


/*
    Node function wire. simply acts as a wire returning the first input
*/
template<typename Real = float> Real f_wire ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    return inputs [ 0 ];
}


/*
    Node function sigmoid. returns the sigmoid of the sum of weighted inputs.
    The specific sigmoid function used in the logistic function.
    range: [0,1]
*/
template<typename Real = float> Real f_sigmoid ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    Real weightedInputSum;
    Real out;

    weightedInputSum = sumWeigtedInputs ( numInputs, inputs, connectionWeights );

    out = 1 / ( 1 + exp ( -weightedInputSum ) );

    return out;
}

/*
    Node function Gaussian. returns the Gaussian of the sum of weighted inputs.
    range: [0,1]
*/
template<typename Real = float> Real f_gaussian ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    Real weightedInputSum;
    Real out;

    int centre = 0;
    int width = 1;

    weightedInputSum = sumWeigtedInputs ( numInputs, inputs, connectionWeights );

    out = exp ( -( pow ( weightedInputSum - centre, 2 ) ) / ( 2 * pow ( width, 2 ) ) );

    return out;
}


/*
    Node function step. returns the step function of the sum of weighted inputs.
    range: [0,1]
*/
template<typename Real = float> Real f_step ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    return sumWeigtedInputs ( numInputs, inputs, connectionWeights ) >= 0.0;
}


/*
    Node function step. returns the step function of the sum of weighted inputs.
    range: [-1,1]
*/
template<typename Real = float> Real f_softsign ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    Real weightedInputSum;
    Real out;

    weightedInputSum = sumWeigtedInputs ( numInputs, inputs, connectionWeights );

    out = weightedInputSum / ( 1.0 + fabs ( weightedInputSum ) );

    return out;
}


/*
    Node function tanh. returns the tanh function of the sum of weighted inputs.
    range: [-1,1]
*/
template<typename Real = float> Real f_hyperbolicTangent ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {

    Real weightedInputSum;
    Real out;

    weightedInputSum = sumWeigtedInputs ( numInputs, inputs, connectionWeights );

    out = tanh ( weightedInputSum );

    return out;
}

} // namespace functions
} // namespace cgp

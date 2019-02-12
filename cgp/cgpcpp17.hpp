
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
#include <limits>
#include <map>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>

#include <frozen/unordered_map.h>
#include <frozen/string.h>


namespace cgp {

template<typename ResultType = std::uint64_t>
class splitmix64 {

    public:

    using result_type = ResultType;

    static inline constexpr result_type min ( ) noexcept {
        return std::numeric_limits<result_type>::min ( );
    }
    static inline constexpr result_type max ( ) noexcept {
        return std::numeric_limits<result_type>::max ( );
    }

    // Seed by default.
    splitmix64 ( ) noexcept :
        m_state { static_cast<std::uint64_t> ( std::random_device { } ( ) ) << 32 | static_cast<std::uint64_t> ( std::random_device { } ( ) ) } {
    }

    splitmix64 ( const std::uint64_t s_ ) noexcept {
        seed ( s_ );
    }

    void seed ( const std::uint64_t s_ ) noexcept {
        m_state = s_;
    }

    bool operator == ( const splitmix64 & rhs_ ) const noexcept {
        return m_state == rhs_.m_state;
    }
    bool operator != ( const splitmix64 & rhs_ ) const noexcept {
        return not ( operator == ( rhs_ ) );
    }

    result_type operator ( ) ( ) noexcept {
        return hash ( next ( ) );
    }

    private:

    std::uint64_t next ( ) noexcept {
        return ( m_state += std::uint64_t { 0x9E3779B97F4A7C15 } );
    }

    std::uint64_t hash ( std::uint64_t z ) const noexcept {
        z = ( z ^ ( z >> 30 ) ) * std::uint64_t { 0xBF58476D1CE4E5B9 };
        z = ( z ^ ( z >> 27 ) ) * std::uint64_t { 0x94D049BB133111EB };
        return z ^ ( z >> 31 );
    }

    std::uint64_t m_state;
};


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

    struct functionData {
        FunctionPointer<Real> function;
        int maxNumInputs;
    };

    std::vector<std::string> functionNames;
    std::vector<FunctionPointer<Real>> functions;
    std::vector<int> maxNumInputs;

    int numFunctions = 0;

    template<typename ... Args>
    void addNodeFunction ( Args && ... args_ ) {
        ( addPresetNodeFunction ( args_ ), ... );
    }

    void addPresetNodeFunction ( const std::string & functionName_ ) {
        const functionData & data = function_set.at ( functionName_ );
        addCustomNodeFunction ( functionName_, data.pointer, data.maxNumInputs );
    }

    void addCustomNodeFunction ( const std::string & functionName_, FunctionPointer<Real> function_, int maxNumInputs_ ) {
        functionNames.push_back ( functionName_ );
        functions.push_back ( function_ );
        maxNumInputs.push_back ( maxNumInputs_ );
        ++numFunctions;
    }

    void clear ( ) noexcept {
        functionNames.clear ( );
        functions.clear ( );
        maxNumInputs.clear ( );
        numFunctions = 0;
    }

    void print ( ) const noexcept {
        std::printf ( "Function Set:" );
        for ( const auto & name : functionNames ) {
            std::printf ( " %s", name );
        }
        std::printf ( " (%d)\n", numFunctions ( ) );
    }

    static constexpr frozen::unordered_map<frozen::string, functionData, 31> function_set {
        { "add", { functions::f_add, -1 } },
        { "sub", { functions::f_sub, -1 } },
        { "mul", { functions::f_mul, -1 } },
        { "div", { functions::f_divide, -1 } },
        { "and", { functions::f_and, -1 } },
        { "abs", { functions::f_absolute, 1 } },
        { "sqrt", { functions::f_squareRoot, 1 } },
        { "sq", { functions::f_square, 1 } },
        { "cube", { functions::f_cube, 1 } },
        { "pow", { functions::f_power, 2 } },
        { "exp", { functions::f_exponential, 1 } },
        { "sin", { functions::f_sine, 1 } },
        { "cos", { functions::f_cosine, 1 } },
        { "tan", { functions::f_tangent, 1 } },
        { "rand", { functions::f_randFloat,  } },
        { "2", { functions::f_constTwo, 0 } },
        { "1", { functions::f_constOne, 0 } },
        { "0", { functions::f_constZero, 0 } },
        { "pi", { functions::f_constPI, 0 } },
        { "nand", { functions::f_nand, -1 } },
        { "or", { functions::f_or, -1 } },
        { "nor", { functions::f_nor, -1 } },
        { "xor", { functions::f_xor, -1 } },
        { "xnor", { functions::f_xnor, -1 } },
        { "not", { functions::f_not, 1 } },
        { "wire", { functions::f_wire, 1 } },
        { "sig", { functions::f_sigmoid, -1 } },
        { "gauss", { functions::f_gaussian, -1 } },
        { "step", { functions::f_step, -1 } },
        { "soft", { functions::f_softsign, -1 } },
        { "tanh", { functions::f_hyperbolicTangent, -1 } }
    };
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
struct Parameters;

template<typename Real = float>
void probabilisticMutation ( const Parameters<Real> & params, Chromosome<Real> & chromo ) noexcept;


template<typename Real>
struct Parameters {

    int mu;
    int lambda;
    char evolutionaryStrategy;
    Real mutationRate;
    std::bernoulli_distribution mutationDistribution;
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
    void ( *mutationType )( const Parameters & params, Chromosome<Real> & chromo );
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
        mutationDistribution { mutationRate },
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
        mutationTypeName { "probabilisticMutation" },
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
    void addNodeFunction ( Args && ... args_ ) {
        funcSet.addNodeFunction ( std::forward<Args> ( args_ ) ...  )
        assert ( funcSet.numFunctions ( ) > 0 );
    }

    void addCustomNodeFunction ( FunctionPointer<Real> function, const std::string functionName, int maxNumInputs ) {


    }

    // Mutation.

    void setMutationRate ( const Real mutationRate_ ) noexcept {
        assert ( mutationRate_ >= Real { 0 } and mutationRate_ <= Real { 1 } );
        mutationRate = mutationRate_;
        mutationDistribution = std::bernoulli_distribution { mutationRate };
    }

    [[ nodiscard ]] bool mutate ( ) const noexcept {
        return mutationDistribution ( Parameters::rng );
    }

    [[ nodiscard ]] int getRandomFunction ( ) const noexcept {
        return Parameters::randInt ( funcSet.numFunctions );
    }

    [[ nodiscard ]] Real getRandomConnectionWeight ( ) const noexcept {
        return std::uniform_real_distribution<Real> ( -connectionWeightRange, connectionWeightRange ) ( Parameters::rng );
    }

    [[ nodiscard ]] int getRandomNodeInput ( const Chromosome<Real> & chromo, const int nodePosition ) const noexcept {
        return std::bernoulli_distribution ( recurrentConnectionProbability ) ( Parameters::rng ) ?
            Parameters::randInt ( chromo.numNodes - nodePosition ) + nodePosition + chromo.numInputs :
            Parameters::randInt ( chromo.numInputs + nodePosition );
    }

    [[ nodiscard ]] int getRandomChromosomeOutput ( const Chromosome<Real> & chromo ) const noexcept {
        return shortcutConnections ? Parameters::randInt ( chromo.numInputs + chromo.numNodes ) : Parameters::randInt ( chromo.numNodes ) + chromo.numInputs;
    }

    // Random generator.

    [[ nodiscard ]] static int randInt ( const int n_ ) noexcept {
        return std::uniform_int_distribution<int> ( 0, n_ - 1 ) ( Parameters::rng );
    }

    void seed_rng ( const std::uint64_t s_ ) noexcept {
        rng.seed ( s_ );
    }

    static splitmix64<std::uint64_t> rng;

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

template<typename Real>
splitmix64<std::uint64_t> Parameters<Real>::rng { };


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


// Conductions probabilistic mutation on the given chromosome. Each chromosome
// gene is changed to a random valid allele with a probability specified in
// parameters.
template<typename Real>
void probabilisticMutation ( const Parameters<Real> & params, Chromosome<Real> & chromo ) noexcept {
    for ( auto & node : chromo.nodes ) {
        // mutate the function gene
        if ( params.mutate ( ) ) {
            node.function = params.getRandomFunction ( );
        }
        int i = 0;
        for ( auto & input : node.inputs ) {
            if ( params.mutate ( ) ) {
                input = params.getRandomNodeInput ( chromo, i );
            }
            ++i;
        }
        for ( auto & weight : node.weights ) {
            if ( params.mutate ( ) ) {
                weight = params.getRandomConnectionWeight ( );
            }
        }
    }
    for ( auto & output : chromo.outputNodes ) {
        if ( params.mutate ( ) ) {
            output = params.getRandomChromosomeOutput ( chromo );
        }
    }
}


namespace functions {

// Node function add. Returns the sum of all the inputs.
template<typename Real> Real f_add ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    int i;
    Real sum = inputs [ 0 ];
    for ( i = 1; i < inputs.size ( ); i++ ) {
        sum += inputs [ i ];
    }
    return sum;
}

// Node function sub. Returns the first input minus all remaining inputs.
template<typename Real> Real f_sub ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    int i;
    Real sum = inputs [ 0 ];
    for ( i = 1; i < inputs.size ( ); i++ ) {
        sum -= inputs [ i ];
    }
    return sum;
}

// Node function mul. Returns the multiplication of all the inputs.
template<typename Real> Real f_mul ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    int i;
    Real multiplication = inputs [ 0 ];
    for ( i = 1; i < inputs.size ( ); i++ ) {
        multiplication *= inputs [ i ];
    }
    return multiplication;
}

// Node function div. Returns the first input divided by the second input divided by the third input etc
template<typename Real> Real f_divide ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    int i;
    Real divide = inputs [ 0 ];
    for ( i = 1; i < inputs.size ( ); i++ ) {
        divide /= inputs [ i ];
    }
    return divide;
}

// Node function abs. Returns the absolute of the first input
template<typename Real> Real f_absolute ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    return std::abs ( inputs [ 0 ] );
}

// Node function sqrt.  Returns the square root of the first input
template<typename Real> Real f_squareRoot ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    return std::sqrt ( inputs [ 0 ] );
}

// Node function squ.  Returns the square of the first input
template<typename Real> Real f_square ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    return std::pow ( inputs [ 0 ], 2 );
}

// Node function cub.  Returns the cube of the first input
template<typename Real> Real f_cube ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    return std::pow ( inputs [ 0 ], 3 );
}

// Node function power.  Returns the first output to the power of the second
template<typename Real> Real f_power ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    return std::pow ( inputs [ 0 ], inputs [ 1 ] );
}

// Node function exp.  Returns the exponential of the first input
template<typename Real> Real f_exponential ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    return std::exp ( inputs [ 0 ] );
}

// Node function sin.  Returns the sine of the first input
template<typename Real> Real f_sine ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    return std::sin ( inputs [ 0 ] );
}

// Node function cos.  Returns the cosine of the first input
template<typename Real> Real f_cosine ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    return std::cos ( inputs [ 0 ] );
}

// Node function tan.  Returns the tangent of the first input
template<typename Real> Real f_tangent ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    return std::tan ( inputs [ 0 ] );
}

// Node function one.  Always returns 1
template<typename Real> Real f_constTwo ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    return 2.0;
}

// Node function one.  Always returns 1
template<typename Real> Real f_constOne ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    return 1.0;
}

// Node function one.  Always returns 0
template<typename Real> Real f_constZero ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    return 0.0;
}

// Node function one.  Always returns PI
template<typename Real> Real f_constPI ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    return 3.141592653589793116;
}

// Node function rand.  Returns a random number between minus one and positive one
template<typename Real> Real f_randFloat ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    return std::uniform_real_distribution<Real> ( -1.0, 1.0 ) ( Parameters<Real>::rng );
}

// Node function and. logical AND, returns '1' if all inputs are '1'
//    else, '0'
template<typename Real> Real f_and ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    int i;
    for ( i = 0; i < inputs.size ( ); i++ ) {
        if ( inputs [ i ] == 0.0 ) {
            return 0.0;
        }
    }
    return 1.0;
}

// Node function and. logical NAND, returns '0' if all inputs are '1'
//    else, '1'
template<typename Real> Real f_nand ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    int i;
    for ( i = 0; i < inputs.size ( ); i++ ) {
        if ( inputs [ i ] == 0.0 ) {
            return 1.0;
        }
    }
    return 0.0;
}

// Node function or. logical OR, returns '0' if all inputs are '0'
//    else, '1'
template<typename Real> Real f_or ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    int i;
    for ( i = 0; i < inputs.size ( ); i++ ) {
        if ( inputs [ i ] == 1.0 ) {
            return 1.0;
        }
    }
    return 0.0;
}

// Node function nor. logical NOR, returns '1' if all inputs are '0'
//    else, '0'
template<typename Real> Real f_nor ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    int i;
    for ( i = 0; i < inputs.size ( ); i++ ) {
        if ( inputs [ i ] == 1.0 ) {
            return 0.0;
        }
    }
    return 1.0;
}

// Node function xor. logical XOR, returns '1' iff one of the inputs is '1'
//    else, '0'. AKA 'one hot'.
template<typename Real> Real f_xor ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    int i;
    int numOnes = 0;
    int out;
    for ( i = 0; i < inputs.size ( ); i++ ) {
        if ( inputs [ i ] == 1 ) {
            numOnes++;
        }
        if ( numOnes > 1 ) {
            break;
        }
    }
    return numOnes == 1;
}

// Node function xnor. logical XNOR, returns '0' iff one of the inputs is '1'
//    else, '1'.
template<typename Real> Real f_xnor ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    int i;
    int numOnes = 0;
    for ( i = 0; i < inputs.size ( ); i++ ) {
        if ( inputs [ i ] == 1 ) {
            numOnes++;
        }
        if ( numOnes > 1 ) {
            break;
        }
    }
    return numOnes != 1;
}

// Node function not. logical NOT, returns '1' if first input is '0', else '1'
template<typename Real> Real f_not ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    return inputs [ 0 ] == 0.0;
}

// Node function wire. simply acts as a wire returning the first input
template<typename Real> Real f_wire ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    return inputs [ 0 ];
}

// Node function sigmoid. returns the sigmoid of the sum of weighted inputs.
//    The specific sigmoid function used in the logistic function.
//    range: [0,1]
template<typename Real> Real f_sigmoid ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    Real weightedInputSum;
    Real out;
    weightedInputSum = sumWeigtedInputs ( inputs.size ( ), inputs, connectionWeights );
    out = 1 / ( 1 + exp ( -weightedInputSum ) );
    return out;
}

// Node function Gaussian. returns the Gaussian of the sum of weighted inputs.
//    range: [0,1]
template<typename Real> Real f_gaussian ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    Real weightedInputSum;
    Real out;
    int centre = 0;
    int width = 1;
    weightedInputSum = sumWeigtedInputs ( inputs.size ( ), inputs, connectionWeights );
    out = exp ( -( pow ( weightedInputSum - centre, 2 ) ) / ( 2 * pow ( width, 2 ) ) );
    return out;
}

// Node function step. returns the step function of the sum of weighted inputs.
//    range: [0,1]
template<typename Real> Real f_step ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    return sumWeigtedInputs ( inputs.size ( ), inputs, connectionWeights ) >= 0.0;
}

// Node function step. returns the step function of the sum of weighted inputs.
//    range: [-1,1]
template<typename Real> Real f_softsign ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    Real weightedInputSum;
    Real out;
    weightedInputSum = sumWeigtedInputs ( inputs.size ( ), inputs, connectionWeights );
    out = weightedInputSum / ( 1.0 + fabs ( weightedInputSum ) );
    return out;
}

// Node function tanh. returns the tanh function of the sum of weighted inputs.
//    range: [-1,1]
template<typename Real> Real f_hyperbolicTangent ( const std::vector<Real> & inputs, const std::vector<Real> & connectionWeights ) noexcept {
    Real weightedInputSum;
    Real out;
    weightedInputSum = sumWeigtedInputs ( inputs.size ( ), inputs, connectionWeights );
    out = tanh ( weightedInputSum );
    return out;
}

} // namespace functions
} // namespace cgp


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

#include <iostream>

#include <random>
#include <splitmix.hpp>

#include "cgp.hpp"

double radius ( const int numInputs, const double *inputs, const double *connectionWeights ) noexcept {
    return 3.0;
}
double radius2 ( const int numInputs, const double *inputs, const double *connectionWeights ) noexcept {
    return 6.0;
}
double int_div ( const int numInputs, const double *inputs, const double *connectionWeights ) noexcept {
    if ( 0 != static_cast<int> ( inputs [ 1 ] ) )
        return static_cast<double> ( static_cast<int> ( inputs [ 0 ] ) % static_cast<int> ( inputs [ 1 ] ) );
    return 0.0;
}
double int_rem ( const int numInputs, const double *inputs, const double *connectionWeights ) noexcept {
    if ( 0 != static_cast<int> ( inputs [ 1 ] ) )
        return static_cast< double > ( static_cast<int> ( inputs [ 0 ] ) % static_cast<int> ( inputs [ 1 ] ) );
    return 0.0;
}
double negate ( const int numInputs, const double *inputs, const double *connectionWeights ) noexcept {
    return -inputs [ 0 ];
}

void tournament ( struct parameters *params, struct chromosome **parents, struct chromosome **candidateChromos, int numParents, int numCandidateChromos ) {

    int i;

    struct chromosome *candidateA;
    struct chromosome *candidateB;

    for ( i = 0; i < numParents; i++ ) {

        candidateA = candidateChromos [ rand ( ) % numCandidateChromos ];
        candidateB = candidateChromos [ rand ( ) % numCandidateChromos ];

        if ( getChromosomeFitness ( candidateA ) <= getChromosomeFitness ( candidateB ) ) {
            copyChromosome ( parents [ i ], candidateA );
        }
        else {
            copyChromosome ( parents [ i ], candidateB );
        }
    }
}


double fitnessWithSizePressure ( struct parameters *params, struct chromosome *chromo, struct dataSet *data ) noexcept {
    static splitmix64 rgen { std::random_device { } ( ) };
    static int best = INT_MAX;
    const double svl_fitness = supervisedLearning ( params, chromo, data );
    if ( svl_fitness == 0.0 ) {
        const int b = getNumChromosomeActiveNodes ( chromo );
        if ( b < best ) {
            best = b;
            printChromosome ( chromo, 0 );
            std::cout << "Active number of nodes " << b << "\n\n";
            saveChromosome ( chromo, "./../data/best.chro" );
        }
    }
    return svl_fitness;// +getNumChromosomeActiveNodes ( chromo ) / 24.0;
}


int main ( ) {

    struct parameters *params = nullptr;
    struct dataSet *trainingData = nullptr;
    struct chromosome *chromo = nullptr;

    int numInputs = 2;
    int numNodes = 24;
    int numOutputs = 1;
    int nodeArity = 2;

    int numGens = 100'000'000;
    double targetFitness = 0.0;
    int updateFrequency = 10'000;

    params = initialiseParameters ( numInputs, numNodes, numOutputs, nodeArity );

    setMu ( params, 1 );
    setLambda ( params, 4 );

    addNodeFunction ( params, "add, sub, mul, abs, 1, 2" );
    addCustomNodeFunction ( params, radius, "rad", 0 );
    addCustomNodeFunction ( params, radius2, "2xrad", 0 );
    addCustomNodeFunction ( params, int_div, "idiv", 2 );
    addCustomNodeFunction ( params, int_rem, "irem", 2 );
    addCustomNodeFunction ( params, negate, "neg", 1 );

    setTargetFitness ( params, targetFitness );

    setUpdateFrequency ( params, updateFrequency );
    setCustomFitnessFunction ( params, fitnessWithSizePressure, "fitnessWithSizePressure" );

    printParameters ( params );

    trainingData = initialiseDataSetFromFile ( "./../data/table.data" );

    chromo = runCGP ( params, trainingData, numGens );

    printChromosome ( chromo, 0 );

    freeDataSet ( trainingData );
    freeChromosome ( chromo );
    freeParameters ( params );

    return EXIT_SUCCESS;
}

/*
( 0 ) : input
( 1 ) : input
( 2 ) : mul     1 1 *
( 3 ) : idiv    2 1 *
( 4 ) : rad *
( 5 ) : mul     4 3 *
( 6 ) : idiv    4 5
( 7 ) : add     0 5 *
( 8 ) : sub     5 2 *
( 9 ) : abs     8 *
( 10 ) : irem    9 5 *
( 11 ) : idiv    10 8 *
( 12 ) : sub     7 4 *
( 13 ) : irem    9 7
( 14 ) : idiv    7 6
( 15 ) : sub     5 11 *
( 16 ) : irem    2 8
( 17 ) : add     12 15 *
*/
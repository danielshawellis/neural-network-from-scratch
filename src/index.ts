import * as R from 'ramda';
import { randomNormal } from 'd3-random';

// Define the hyper-parameters
const LAYERS: readonly { size: number }[] = [{ size: 784 }, { size: 20 }, { size: 10 }];
const LEARNING_RATE: number = 3;

// Define types
type Biases = number[][];
type Weights = number[][][];

// Define the activation (squishification) functions
const sigmoid = (x: number) => 1 / (1 + (Math.E ** -x)); // Explanation: https://en.wikipedia.org/wiki/Sigmoid_function

// Define getter functions for the weights and biases
const getLayerBiases = (biases: Biases, layerIndex: number) => biases[layerIndex -1]; // Add an offset since there are no biases for the input layer
const getWeightBetweenNodes = (weights: Weights, currentLayerIndex: number, currentNodeIndex: number, previousNodeIndex: number) => weights[currentLayerIndex - 1][previousNodeIndex][currentNodeIndex]; // Add an offset since the adjacent layers are paired

// Define how each subsequent layer's activations are computed
const computeLayerActivations = (weights: Weights, biases: Biases, currentLayerIndex: number, previousLayerActivations: number[]) => {
	// Map over each node in this layer via the layer's biases array (it directly corresponds to the layer's nodes)
	return getLayerBiases(biases, currentLayerIndex).map((currentNodeBias, currentNodeIndex) => {
		// Compute weighted inputs between this node and each node in the previous layer
		const weightedInputs = previousLayerActivations.map((previousNodeActivation, previousNodeIndex) => {
			const weightBetweenNodes = getWeightBetweenNodes(weights, currentLayerIndex, currentNodeIndex, previousNodeIndex);
			return (previousNodeActivation * weightBetweenNodes) + currentNodeBias;
		});

		// Sum up all of the weighted inputs and compress the product into a range between zero and one using the sigmoid activation function 
		return sigmoid(R.sum(weightedInputs));
	});
};

// Define the feed forward process (to compute the output activations through the entire net)
const feedForward = (weights: Weights, biases: Biases, inputLayer: number[]) => {
	const layerIndecies = Array.from({ length: biases.length }, (_, i) => i + 1); // Use the biases length since it corresponds to the number of non-input layers
	return layerIndecies.reduce((previousLayerActivations, currentLayerIndex) => computeLayerActivations(weights, biases, currentLayerIndex, previousLayerActivations), inputLayer);
};

// Randomly initialize the weights and biases
const sampleStandardNormal = randomNormal(0, Math.sqrt(1)); // Generate random numbers that follow a normal (Gaussian) distribution with a mean of 0 and a variance of 1
const initialWeights: Weights = R.pipe(
	(x: typeof LAYERS) => R.zip(R.slice(0, -1, x), R.slice(1, Infinity, x)), // Get pairs of adjacent layers
	R.map(([leftLayer, rightLayer]) => Array.from({ length: leftLayer.size }, () => Array.from({ length: rightLayer.size }, sampleStandardNormal))), // Create 2-dimentional arrays representing all possibe connections between nodes in adjacent layers, then fill it with normally distributed random weights 
)(LAYERS);
const initialBiases: Biases = R.pipe(
	R.slice(1, Infinity), // Leave off the input layer since it needs no biases
	R.map(({ size }) => Array.from({ length: size }, sampleStandardNormal)) // For each subsequent layer, create a 1-dimentional array with normally distributed random biases for each node
)(LAYERS);

// TEMPORARY: Create a sample input and compute the output
const input = Array.from({ length: LAYERS[0].size }, Math.random);
const output = feedForward(initialWeights, initialBiases, input);
console.log(output);
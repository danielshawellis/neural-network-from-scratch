import * as R from 'ramda';
import { randomNormal } from 'd3-random';

// Define the hyper-parameters
const LAYERS: readonly { size: number }[] = [{ size: 784 }, { size: 20 }, { size: 10 }];
const LEARNING_RATE: number = 3;

// Define the activation (squishification) functions
const sigmoid = (x: number) => 1 / (1 + (Math.E ** -x)); // Explanation: https://en.wikipedia.org/wiki/Sigmoid_function

// Randomly initialize the weights and biases
const sampleStandardNormal = randomNormal(0, Math.sqrt(1)); // Generate random numbers that follow a normal (Gaussian) distribution with a mean of 0 and a variance of 1
const biases: number[][] = R.pipe(
	R.slice(1, Infinity), // Leave off the input layer since it needs no biases
	R.map(({ size }) => Array.from({ length: size }, sampleStandardNormal)) // For each subsequent layer, create a 1-dimentional array with normally distributed random biases for each node
)(LAYERS);
const weights: number[][][] = R.pipe(
	(x: typeof LAYERS) => R.zip(R.slice(0, -1, x), R.slice(1, Infinity, x)), // Get pairs of adjacent layers
	R.map(([leftLayer, rightLayer]) => Array.from({ length: leftLayer.size }, () => Array.from({ length: rightLayer.size }, sampleStandardNormal))), // Create 2-dimentional arrays representing all possibe connections between nodes in adjacent layers, then fill it with normally distributed random weights 
)(LAYERS);

// Define getter functions for the weights and biases
const getLayerBiases = (layerIndex: number) => biases[layerIndex -1]; // Add an offset since there are no biases for the input layer
const getWeightBetweenNodes = (currentLayerIndex: number, currentNodeIndex: number, previousNodeIndex: number) => weights[currentLayerIndex - 1][previousNodeIndex][currentNodeIndex]; // Add an offset since the adjacent layers are paired

// Define how each subsequent layer's activations are computed
const computeLayerActivations = (currentLayerIndex: number, previousLayerActivations: number[]) => {	
	const currentLayerBiases = getLayerBiases(currentLayerIndex);
	const activations = currentLayerBiases.map((currentNodeBias, currentNodeIndex) => {
		const weightedInputs = previousLayerActivations.map((previousNodeActivation, previousNodeIndex) => {
			const weightBetweenNodes = getWeightBetweenNodes(currentLayerIndex, currentNodeIndex, previousNodeIndex);
			return (previousNodeActivation * weightBetweenNodes) + currentNodeBias;
		})
		return sigmoid(R.sum(weightedInputs));
	});
	return activations;
};

// Define the feed forward process (to compute the output activations through the entire net)
const feedForward = (inputLayer: number[]) => {
	const layerIndecies = Array.from({ length: LAYERS.length - 1 }, (_, i) => i + 1);
	return layerIndecies.reduce((previousLayerActivations, currentLayerIndex) => computeLayerActivations(currentLayerIndex, previousLayerActivations), inputLayer);
};

// TEMPORARY: Create a sample input and compute the output
const input = Array.from({ length: LAYERS[0].size }, Math.random);
const output = feedForward(input);
console.log(output);
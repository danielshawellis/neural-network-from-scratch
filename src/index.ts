import * as R from 'ramda';
import { randomNormal } from 'd3-random';

// Define types
type Biases = number[][];
type Weights = number[][][];
type ImageData = { image: number[], label: number };

// Define the hyper-parameters
const LAYERS: readonly { size: number }[] = [{ size: 784 }, { size: 20 }, { size: 10 }];
const LEARNING_RATE: number = 3;
const EPOCHS: number = 5;
const MINI_BATCH_SIZE: number = 10;
const ADJUSTMENT_DISTANCE: number = 1; // Magnitude of adjustments made during experimental gradient estimation

// Import the MNIST image dataset and split the training and test data
import _mnistDataset from '../data/mnist-dataset.json';
const mnistDataset = _mnistDataset as ImageData[];
const [trainingData, testData] = R.splitAt(50_000, mnistDataset);

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

// Define the cost function (which measures the error for a given data point)
const computeCost = (weights: Weights, biases: Biases, { image, label }: ImageData) => {
	const predictedActivations = feedForward(weights, biases, image);
	const actualActivations = Array.from({ length: 10 }, (_, index) => (index === label) ? 1 : 0);
	const cost = R.pipe(
		R.map(([predictedActivation, actualActivation]) => Math.abs(predictedActivation - actualActivation)), // Calculate the error for each prediction
		R.sum // Sum up all of the prediction errors
	)(R.zip(predictedActivations, actualActivations));
	return cost;
};
const computeSumCost = (weights: Weights, biases: Biases, images: ImageData[]) => R.pipe(
	R.map((image: ImageData) => computeCost(weights, biases, image)),
	R.sum
)(images);

// Experimentally compute the cost gradients of the weights and biases for some specified images
const computeCostGradients = (weights: Weights, biases: Biases, images: ImageData[], adjustmentDistance: number) => {
	// Calculate the cost (error) of the current weights and biases
	const currentCost = computeSumCost(weights, biases, images);

	// Experimentally estimate the cost gradient for each of the weights
	const weightsCostGradient = weights.map((x, layerPairIndex) => x.map((y, leftNodeIndex) => y.map((weight, rightNodeIndex) => {
		const adjustedWeights = R.assocPath([layerPairIndex, leftNodeIndex, rightNodeIndex], weight + adjustmentDistance, weights);
		const adjustedCost = computeSumCost(adjustedWeights, biases, images);
		return (adjustedCost - currentCost) / adjustmentDistance;
	})));

	// Experimentally estimate the cost gradient for the biases on each node
	const biasesCostGradient = biases.map((layerBiases, layerIndex) => layerBiases.map((bias, nodeIndex) => {
		const adjustedBiases = R.assocPath([layerIndex, nodeIndex], bias + adjustmentDistance, biases);
		const adjustedCost = computeSumCost(weights, adjustedBiases, images);
		return (adjustedCost - currentCost) / adjustmentDistance;
	}));

	// Return both cost gradients
	return { weightsCostGradient, biasesCostGradient };
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

// On every epoch, compute the cost gradients and descend
const [finalWeights, finalBiases] = R.reduce(
	([weights, biases]: [Weights, Biases], epoch): [Weights, Biases] => {
		// Get a subset of the images for training (note that these aren't true epochs since we aren't using all of the training data each time)
		const trainingDataRandomSubset = R.sort(() => Math.random() - 0.5, trainingData).slice(0, MINI_BATCH_SIZE);

		// Calculate the cost (error) of the current weights and biases
		const { weightsCostGradient, biasesCostGradient } = computeCostGradients(weights, biases, trainingDataRandomSubset, ADJUSTMENT_DISTANCE);

		// Improve the weights and biases based on their cost gradient (nudge them in the right direction to reduce error)
		const improvedWeights = weights.map((x, layerPairIndex) => x.map((y, leftNodeIndex) => y.map((weight, rightNodeIndex) => {
			const weightCostGradient = weightsCostGradient[layerPairIndex][leftNodeIndex][rightNodeIndex];
			return weight + (weightCostGradient * LEARNING_RATE);
		})));
		const improvedBiases = biases.map((layerBiases, layerIndex) => layerBiases.map((bias, nodeIndex) => {
			const biasCostGradient = biasesCostGradient[layerIndex][nodeIndex];
			return bias + (biasCostGradient * LEARNING_RATE);
		}));

		// Log the completion of the epoch
		console.log(`Epoch ${epoch} Complete`);
		
		// Return the improved weights and biases
		return [improvedWeights, improvedBiases];
	},
	[initialWeights, initialBiases] as [Weights, Biases],
	Array.from({ length: EPOCHS }, (_, index) => index + 1)
);

// Compare the total error before and after training to determine whether the accuracy improved
console.log("Initial Cost: ", computeSumCost(initialWeights, initialBiases, testData));
console.log("Final Cost: ", computeSumCost(finalWeights, finalBiases, testData));

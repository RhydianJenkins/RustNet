use std::f64::consts::E;

use super::training_data::TrainingData;

pub struct Perceptron {
    weights: Vec<f64>,
    bias: f64,
}

const LEARNING_RATE: f64 = 0.01;

impl Perceptron {
    pub fn new(weights: Vec<f64>, bias: f64) -> Perceptron {
        Perceptron { weights, bias }
    }

    /*
     * For every input, multiply that input by its weight.
     * Sum all of the weighted inputs.
     * Compute the output of the perceptron based on that sum passed through an activation function
     * (the sign of the sum).
     */
    pub fn feed_forward(&self, inputs: &Vec<f64>) -> f64 {
        let sum: f64 = inputs
            .iter()
            .zip(self.weights.iter())
            .map(|(input, weight)| input * weight + self.bias)
            .sum();

        self.activate(sum)
    }

    /*
     * Takes a value from the feed_forward and decides what value it needs to output.
     * This can get really complicated, involving calculus and other things in more complex neural networks.
     * Here, we're KISSing it and returning a normalized value between 0 and 1.
     */
    fn activate(&self, sum: f64) -> f64 {
        let normalized_sum = sum / (1.0 + E.powf(-sum));

        normalized_sum
    }

    /*
     * Updates the weights and bias of this perceptron based on the error of the guess.
     * Returns the guess (the output of the perceptron).
     */
    pub fn train(&mut self, training_data: &TrainingData) -> f64 {
        let guess = self.feed_forward(&training_data.inputs);
        let error = training_data.desired_answer - guess;
        let weights: Vec<f64> = self
            .weights
            .iter()
            .zip(training_data.inputs.iter())
            .map(|(weight, training_input)| weight + training_input * error * LEARNING_RATE)
            .collect();

        self.weights = weights;
        self.bias = self.bias + error * LEARNING_RATE;

        guess
    }
}

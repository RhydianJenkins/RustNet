use crate::neural_network::training_data::TrainingData;

use super::{perceptron::Perceptron, random_float_generator::gen_random_floats};

const NUM_INPUTS: usize = 2;
const STARTING_BIAS: f64 = 1.0;
const NUM_TRAINING_ITERATIONS: i32 = 10000;

pub struct Network {
    input_layer: Vec<Perceptron>,
    hidden_layer: Vec<Perceptron>,
    output_layer: Vec<Perceptron>,
}

impl Network {
    pub fn new() -> Network {
        let input_perceptron = Perceptron::new(gen_random_floats(NUM_INPUTS), STARTING_BIAS);
        let hidden_perceptron = Perceptron::new(gen_random_floats(1), STARTING_BIAS);
        let output_perceptron = Perceptron::new(gen_random_floats(1), STARTING_BIAS);

        Network {
            input_layer: vec![input_perceptron],
            hidden_layer: vec![hidden_perceptron],
            output_layer: vec![output_perceptron],
        }
    }

    /*
     * For every input, multiply that input by its weight.
     * Sum all of the weighted inputs.
     * Compute the output of the perceptron based on that sum passed through an activation function
     * Repeat for every perceptron in the network.
     * Return the output of the last perceptron in the network.
     * This is the prediction.
     */
    pub fn feed_forward(&self, inputs: &Vec<f64>) -> Vec<f64> {
        if inputs.len() != NUM_INPUTS {
            panic!("Expected {} inputs, got {}", NUM_INPUTS, inputs.len());
        }

        let mut input_layer_results: Vec<f64> = vec![];
        let mut hidden_layer_results: Vec<f64> = vec![];
        let mut output_layer_results: Vec<f64> = vec![];

        for perceptron in &self.input_layer {
            let output = perceptron.feed_forward(inputs);
            input_layer_results.push(output);
        }

        for perceptron in &self.hidden_layer {
            let output = perceptron.feed_forward(&input_layer_results);
            hidden_layer_results.push(output);
        }

        for perceptron in &self.output_layer {
            let output = perceptron.feed_forward(&hidden_layer_results);
            output_layer_results.push(output);
        }

        output_layer_results
    }

    fn back_propagate(&mut self, feed_forward_results: Vec<f64>, desired_answer: f64) {
        let mut output_back_propagation_results = vec![];
        let mut hidden_back_propagation_results = vec![];

        for perceptron in &mut self.output_layer {
            let output_back_propagation_result = perceptron.train(&TrainingData {
                inputs: feed_forward_results.clone(),
                desired_answer,
            });
            output_back_propagation_results.push(output_back_propagation_result);
        }

        for perceptron in &mut self.hidden_layer {
            let hidden_back_propagation_result = perceptron.train(&TrainingData {
                inputs: output_back_propagation_results.clone(),
                desired_answer,
            });
            hidden_back_propagation_results.push(hidden_back_propagation_result);
        }

        for perceptron in &mut self.input_layer {
            perceptron.train(&TrainingData {
                inputs: hidden_back_propagation_results.clone(),
                desired_answer,
            });
        }
    }

    /*
     * TODO train the network and feed forward training to the later layers
     */
    pub fn train(&mut self, inputs: &Vec<f64>, desired_answer: f64) {
        for _ in 0..NUM_TRAINING_ITERATIONS {
            let feed_forward_results = self.feed_forward(inputs);
            self.back_propagate(feed_forward_results, desired_answer);
        }
    }
}

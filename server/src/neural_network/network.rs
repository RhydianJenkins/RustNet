use crate::neural_network::training_data::TrainingData;

use super::{perceptron::Perceptron, random_float_generator::gen_random_floats};

const NUM_INPUTS: usize = 2;
const STARTING_BIAS: f64 = 1.0;

pub struct Network {
    input_layer: Vec<Perceptron>,
}

impl Network {
    pub fn new() -> Network {
        let random_weights = gen_random_floats(NUM_INPUTS);
        let perceptron1 = Perceptron::new(random_weights, STARTING_BIAS);

        Network {
            input_layer: vec![perceptron1],
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

        let mut outputs: Vec<f64> = vec![];

        for perceptron in &self.input_layer {
            let output = perceptron.feed_forward(inputs);
            outputs.push(output);
        }

        outputs
    }

    pub fn train(&mut self) {
        for _ in 0..10000 {
            let training_data = TrainingData::new();
            self.input_layer[0].train(&training_data);
        }
    }
}

use crate::neural_network::network::{LEARNING_RATE, STARTING_BIAS};
use serde::Serialize;
use std::f64::consts::E;

#[derive(Clone, Debug, Serialize)]
pub struct Perceptron {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl Perceptron {
    pub fn new(weights: Vec<f64>) -> Perceptron {
        Perceptron {
            weights,
            bias: STARTING_BIAS,
        }
    }

    /*
     * Weighted sum all of the inputs.
     * Activate the weighted sum with a sigmoid function.
     */
    pub fn activate(&self, inputs: &Vec<f64>) -> f64 {
        debug_assert_eq!(self.weights.len(), inputs.len());

        let weighted_sum: f64 = inputs
            .iter()
            .zip(self.weights.iter())
            .map(|(input, weight)| input * weight)
            .sum();

        sigmoid(weighted_sum + self.bias)
    }

    pub fn update_weights(&mut self, average_error: f64, prev_layer_results: &Vec<f64>) {
        let new_weights = self
            .weights
            .iter()
            .zip(prev_layer_results.iter())
            .map(|(weight, prev_layer_result)| {
                let new_weight = weight - (LEARNING_RATE * average_error * prev_layer_result);

                new_weight
            })
            .collect::<Vec<f64>>();

        let perceptron_output = self.activate(prev_layer_results);
        self.bias = calculate_new_bias(self.bias, average_error, perceptron_output);

        debug_assert_eq!(new_weights.len(), self.weights.len());

        self.weights = new_weights;
    }
}

fn calculate_new_bias(prev_bias: f64, average_error: f64, perceptron_output: f64) -> f64 {
    let derivative = perceptron_output * (1.0 - perceptron_output);
    let new_bias = prev_bias + LEARNING_RATE * average_error * derivative;

    new_bias
}

fn sigmoid(x: f64) -> f64 {
    let result = 1.0 / (1.0 + E.powf(-x));

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! sigmoid_tests {
        ($($name:ident: $value:expr,)*) => {$(
            #[test]
            fn $name() {
                let result = sigmoid($value);
                assert!(result >= 0.0 && result <= 1.0);
            }
        )*}
    }

    sigmoid_tests! {
        sig_0: -10.0,
        sig_1: -0.0,
        sig_2: 0.0,
        sig_3: 0.5,
        sig_4: 1.0,
        sig_5: 1000.0,
    }
}

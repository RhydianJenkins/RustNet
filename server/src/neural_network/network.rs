use super::{perceptron::Perceptron, random_float_generator::gen_random_floats};
use indicatif::ProgressBar;
use serde::Serialize;

pub const NUM_RAW_INPUTS: usize = 10; // TODO 784 inputs in a 28x28 canvas
const NUM_HIDDEN_NEURONS: usize = 16;
const NUM_OUTPUTS: usize = 9;
const NUM_TRAINING_ITERATIONS: i32 = 100;

#[derive(Clone, Debug, Serialize)]
pub struct Network {
    input_layer: Vec<Perceptron>,
    hidden_layer: Vec<Perceptron>,
    output_layer: Vec<Perceptron>,
    trained: bool,
}

impl Network {
    pub fn new() -> Network {
        Network {
            input_layer: (0..NUM_HIDDEN_NEURONS)
                .map(|_| Perceptron::new(gen_random_floats(NUM_RAW_INPUTS)))
                .collect(),
            hidden_layer: (0..NUM_HIDDEN_NEURONS)
                .map(|_| Perceptron::new(gen_random_floats(NUM_HIDDEN_NEURONS)))
                .collect(),
            output_layer: (0..NUM_OUTPUTS)
                .map(|_| Perceptron::new(gen_random_floats(NUM_HIDDEN_NEURONS)))
                .collect(),
            trained: false,
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
    pub fn feed_forward(&self, inputs: &Vec<f64>) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut input_layer_results: Vec<f64> = vec![];
        let mut hidden_layer_results: Vec<f64> = vec![];
        let mut output_layer_results: Vec<f64> = vec![];

        for perceptron in &self.input_layer {
            let output = perceptron.activate(inputs);
            input_layer_results.push(output);
        }

        for perceptron in &self.hidden_layer {
            let output = perceptron.activate(&input_layer_results);
            hidden_layer_results.push(output);
        }

        for perceptron in &self.output_layer {
            let output = perceptron.activate(&hidden_layer_results);
            output_layer_results.push(output);
        }

        (
            input_layer_results,
            hidden_layer_results,
            output_layer_results,
        )
    }

    /*
     * for every node in the output layer
     *   calculate the error signal
     * end
     *
     * for all hidden layers
     *   for every node in the layer
     *     1. Calculate the node's signal error
     *     2. Update each node's weight in the network
     *   end
     * end
     */
    fn back_propagate(
        &mut self,
        raw_input: &Vec<f64>,
        input_layer_results: &Vec<f64>,
        hidden_layer_results: &Vec<f64>,
        output_layer_results: &Vec<f64>,
        desired_results: &Vec<f64>,
    ) {
        let output_error_signal =
            self.calculate_error_signal(output_layer_results, desired_results);
        self.update_output_weights(&output_error_signal, hidden_layer_results);

        let hidden_error_signal =
            self.calculate_error_signal(hidden_layer_results, &output_error_signal);
        self.update_hidden_weights(&hidden_error_signal, input_layer_results);

        let input_error_signal =
            self.calculate_error_signal(input_layer_results, &hidden_error_signal);
        self.update_input_weights(&input_error_signal, raw_input);
    }

    fn calculate_error_signal(
        &self,
        actual_results: &Vec<f64>,
        desired_results: &Vec<f64>,
    ) -> Vec<f64> {
        let error_signal = actual_results
            .iter()
            .zip(desired_results.iter())
            .map(|(output, desired)| output - desired)
            .collect::<Vec<f64>>();

        error_signal
    }

    fn update_output_weights(&mut self, error_signal: &Vec<f64>, prev_layer_results: &Vec<f64>) {
        for perceptron in &mut self.output_layer {
            perceptron.update_weights(error_signal, prev_layer_results);
        }
    }

    fn update_hidden_weights(&mut self, error_signal: &Vec<f64>, prev_layer_results: &Vec<f64>) {
        for perceptron in &mut self.hidden_layer {
            perceptron.update_weights(error_signal, prev_layer_results);
        }
    }

    fn update_input_weights(&mut self, error_signal: &Vec<f64>, prev_layer_results: &Vec<f64>) {
        for perceptron in &mut self.input_layer {
            perceptron.update_weights(error_signal, prev_layer_results);
        }
    }

    pub fn train(&mut self) {
        println!("Training with {} iterations...", NUM_TRAINING_ITERATIONS);

        let pb = ProgressBar::new(NUM_TRAINING_ITERATIONS.try_into().unwrap());

        for (_i, _training_iteration) in (0..NUM_TRAINING_ITERATIONS).enumerate() {
            // TODO: Make this come from the grid values of the user's canvas
            let training_data = &gen_random_floats(NUM_RAW_INPUTS);

            // pretend the user drew a 2
            let desired_outputs = &vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

            pb.inc(1);

            self.train_once(training_data, desired_outputs);
        }

        self.trained = true;

        pb.finish_with_message("Done");
        println!("Training complete.");
    }

    fn train_once(&mut self, inputs: &Vec<f64>, desired_results: &Vec<f64>) {
        let (input_layer_results, hidden_layer_results, output_layer_answers) =
            self.feed_forward(inputs);

        self.back_propagate(
            &inputs,
            &input_layer_results,
            &hidden_layer_results,
            &output_layer_answers,
            desired_results,
        );
    }
}

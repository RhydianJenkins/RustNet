use super::{perceptron::Perceptron, random_float_generator::gen_random_floats};

const NUM_INPUTS: usize = 16;
const STARTING_BIAS: f64 = 1.0;
const NUM_TRAINING_ITERATIONS: i32 = 10000;

pub struct Network {
    input_layer: Vec<Perceptron>,
    hidden_layer: Vec<Perceptron>,
    output_layer: Vec<Perceptron>,
}

fn gen_layer(num_perceptrons: usize, num_inputs: usize) -> Vec<Perceptron> {
    (0..num_perceptrons)
        .map(|_| Perceptron::new(gen_random_floats(num_inputs), STARTING_BIAS))
        .collect()
}

impl Network {
    pub fn new() -> Network {
        Network {
            input_layer: gen_layer(16, NUM_INPUTS),
            hidden_layer: gen_layer(16, 16),
            output_layer: gen_layer(9, 16),
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

        (
            input_layer_results,
            hidden_layer_results,
            output_layer_results,
        )
    }

    fn back_propagate(
        &mut self,
        input_layer_results: Vec<f64>,
        hidden_layer_results: Vec<f64>,
        output_layer_results: Vec<f64>,
        desired_answer: &Vec<f64>,
    ) {
        let output_back_propagation_results = self
            .output_layer
            .iter_mut()
            .enumerate()
            .map(|(index, perceptron)| {
                let desired_answer_for_this_perceptron = desired_answer
                    .get(index)
                    .expect("could not get desired answer for perceptron in output layer");

                perceptron.train(&output_layer_results, *desired_answer_for_this_perceptron)
            })
            .collect::<Vec<f64>>();

        let hidden_back_propagation_results = self
            .hidden_layer
            .iter_mut()
            .enumerate()
            .map(|(index, perceptron)| {
                let desired_answer_for_this_perceptron = hidden_layer_results
                    .get(index)
                    .expect("could not get desired answer for perceptron in hidden layer");

                perceptron.train(
                    &output_back_propagation_results,
                    *desired_answer_for_this_perceptron,
                )
            })
            .collect::<Vec<f64>>();

        self.input_layer
            .iter_mut()
            .enumerate()
            .for_each(|(index, perceptron)| {
                let desired_answer_for_this_perceptron = input_layer_results
                    .get(index)
                    .expect("could not get desired answer for perceptron in input layer");

                perceptron.train(
                    &hidden_back_propagation_results,
                    *desired_answer_for_this_perceptron,
                );
            });
    }

    pub fn train(&mut self, inputs: &Vec<f64>, desired_answer: &Vec<f64>) {
        for _ in 0..NUM_TRAINING_ITERATIONS {
            let (input_layer_results, hidden_layer_results, output_layer_results) =
                self.feed_forward(inputs);

            self.back_propagate(
                input_layer_results,
                hidden_layer_results,
                output_layer_results,
                desired_answer,
            );
        }
    }
}

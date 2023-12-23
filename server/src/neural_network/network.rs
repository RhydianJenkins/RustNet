use super::data_loader::MnistImage;
use super::{perceptron::Perceptron, random_float_generator::gen_random_floats};
use indicatif::ProgressBar;
use serde::Serialize;

pub const STARTING_BIAS: f64 = 0.01;
pub const LEARNING_RATE: f64 = 0.001;
const NUM_RAW_INPUTS: usize = 784;
const NUM_TO_TRAIN: usize = 50_000;
const NUM_HIDDEN_NEURONS: usize = 16;
const NUM_OUTPUTS: usize = 10;

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

    pub fn feed_forward(&self, inputs: &Vec<f64>) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut input_layer_results: Vec<f64> = vec![];
        let mut hidden_layer_results: Vec<f64> = vec![];
        let mut output_layer_results: Vec<f64> = vec![];

        for perceptron in &self.input_layer {
            let output = perceptron.activate(inputs);
            input_layer_results.push(output);
        }
        debug_assert_eq!(input_layer_results.len(), NUM_HIDDEN_NEURONS);

        for perceptron in &self.hidden_layer {
            let output = perceptron.activate(&input_layer_results);
            hidden_layer_results.push(output);
        }
        debug_assert_eq!(hidden_layer_results.len(), NUM_HIDDEN_NEURONS);

        for perceptron in &self.output_layer {
            let output = perceptron.activate(&hidden_layer_results);
            output_layer_results.push(output);
        }
        debug_assert_eq!(output_layer_results.len(), NUM_OUTPUTS);

        (
            input_layer_results,
            hidden_layer_results,
            output_layer_results,
        )
    }

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

    fn update_output_weights(&mut self, error_signals: &Vec<f64>, prev_layer_results: &Vec<f64>) {
        let average_error = error_signals.iter().sum::<f64>() / error_signals.len() as f64;
        for perceptron in &mut self.output_layer {
            perceptron.update_weights(average_error, prev_layer_results);
        }
    }

    fn update_hidden_weights(&mut self, error_signals: &Vec<f64>, prev_layer_results: &Vec<f64>) {
        let average_error = error_signals.iter().sum::<f64>() / error_signals.len() as f64;
        for perceptron in &mut self.hidden_layer {
            perceptron.update_weights(average_error, prev_layer_results);
        }
    }

    fn update_input_weights(&mut self, error_signals: &Vec<f64>, prev_layer_results: &Vec<f64>) {
        let average_error = error_signals.iter().sum::<f64>() / error_signals.len() as f64;
        for perceptron in &mut self.input_layer {
            perceptron.update_weights(average_error, prev_layer_results);
        }
    }

    pub fn train(&mut self, training_dataset: &Vec<MnistImage>) {
        println!("Training...");
        let data = if training_dataset.len() < NUM_TO_TRAIN {
            training_dataset
        } else {
            &training_dataset[0..NUM_TO_TRAIN]
        };

        let pb = ProgressBar::new((data.len() - 1).try_into().unwrap());

        data.iter().for_each(|training_data| {
            self.train_once(training_data);

            pb.inc(1);
        });

        self.trained = true;

        pb.finish_with_message("Done");
        println!("Training complete.");
    }

    fn train_once(&mut self, training_data: &MnistImage) {
        let inputs = &training_data.image;
        let (input_layer_results, hidden_layer_results, output_layer_answers) =
            self.feed_forward(inputs);

        self.back_propagate(
            inputs,
            &input_layer_results,
            &hidden_layer_results,
            &output_layer_answers,
            &training_data.desired_output,
        );
    }
}

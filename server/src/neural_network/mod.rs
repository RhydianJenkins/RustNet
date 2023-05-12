pub mod data_loader;
pub mod network;
mod perceptron;
pub mod random_float_generator;

use data_loader::load_data;
use network::Network;

pub fn generate_trained_network() -> Network {
    let training_dataset = load_data("train").unwrap();
    let mut network = Network::new();

    network.train(&training_dataset);
    network
}

pub fn generate_predictions(network: &Network, input_values: &Vec<f64>) -> Result<Vec<f64>, ()> {
    let (_, _, predictions) = network.feed_forward(input_values);

    Ok(predictions)
}

#[cfg(test)]
mod tests {
    use crate::neural_network::random_float_generator::gen_random_floats;

    use super::*;

    #[test]
    fn generate_predictions_is_ok() {
        let network = Network::new();
        let random_inputs = gen_random_floats(network::NUM_RAW_INPUTS);
        let predictions = generate_predictions(&network, &random_inputs);
        assert!(predictions.is_ok());
    }
}

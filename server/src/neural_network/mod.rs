pub mod network;
mod perceptron;
mod random_float_generator;

use network::Network;

pub fn generate_trained_network() -> Network {
    let mut network = Network::new();
    network.train();
    network
}

pub fn generate_predictions(network: &Network) -> Result<Vec<f64>, ()> {
    let input_values = &vec![0.1]; // TODO raw pixel values from canvas
    let (_, _, predictions) = network.feed_forward(&input_values);

    Ok(predictions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_predictions_is_ok() {
        let network = Network::new();
        let predictions = generate_predictions(&network);
        assert!(predictions.is_ok());
    }
}

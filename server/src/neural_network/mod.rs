mod network;
mod perceptron;
mod random_float_generator;

use network::Network;

pub fn generate_predictions() -> Result<Vec<f64>, ()> {
    let network = &mut Network::new();

    network.train();

    let input_values = &vec![0.1]; // TODO raw pixel values from canvas
    let (_, _, predictions) = network.feed_forward(&input_values);

    Ok(predictions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_predictions_is_ok() {
        let predictions = generate_predictions();
        assert!(predictions.is_ok());
    }
}

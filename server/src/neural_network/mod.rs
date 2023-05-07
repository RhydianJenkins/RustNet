mod network;
mod perceptron;
mod random_float_generator;

use network::Network;

use self::random_float_generator::gen_random_floats;

pub fn generate_predictions() -> Result<Vec<f64>, ()> {
    let network = &mut Network::new();

    let inputs = gen_random_floats(16);
    let desired_outputs = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    network.train(&inputs, &desired_outputs);

    let (_, _, predictions) = network.feed_forward(&inputs);

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

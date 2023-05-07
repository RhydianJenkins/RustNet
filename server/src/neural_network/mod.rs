mod network;
mod perceptron;
mod random_float_generator;
mod training_data;

use network::Network;

use self::random_float_generator::gen_random_floats;

pub fn generate_predictions() -> Result<Vec<f64>, ()> {
    let network = &mut Network::new();

    let inputs = gen_random_floats(2);
    network.train(&inputs, 0.5);

    let predictions = network.feed_forward(&inputs);

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

mod connection;
mod network;
mod perceptron;
mod random_float_generator;
mod training_data;

use network::Network;

use self::random_float_generator::gen_random_floats;

pub fn generate_predictions() -> Result<Vec<f64>, ()> {
    let network = &mut Network::new();

    network.train(); // TODO no need to train the network every time

    let input = gen_random_floats(2);
    let predictions = network.feed_forward(&input);

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

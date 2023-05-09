mod network;
mod perceptron;
mod random_float_generator;

use network::Network;

pub fn generate_predictions() -> Result<Vec<f64>, ()> {
    let network = &mut Network::new();

    // TODO: Make this come from the grid values of the user's canvas
    let pixel_values = vec![
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
    ];

    // pretend the user drew a 9
    let desired_outputs = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    network.train(&pixel_values, &desired_outputs);

    let (_, _, predictions) = network.feed_forward(&pixel_values);

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

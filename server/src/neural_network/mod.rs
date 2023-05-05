mod perceptron;

use perceptron::Perceptron;
use rand::{thread_rng, Rng};

const NUM_INPUTS: usize = 20;

pub fn generate_predictions() -> Result<f64, ()> {
    let random_weights = gen_random_floats(NUM_INPUTS);
    let bias = 1.0;
    let perceptron = Perceptron::new(random_weights, bias);

    // TODO: The input coordinates' (or previous neurons'?) values
    let input = vec![1.0, 2.0];

    Ok(perceptron.feed_forward(input))
}

fn gen_random_floats(number: usize) -> Vec<f64> {
    let values: Vec<f64> = (0..number).map(|_| gen_random_float()).collect();

    values
}

fn gen_random_float() -> f64 {
    thread_rng().gen_range::<f64, _>(0.0..1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_predictions_is_ok() {
        let predictions = generate_predictions();
        assert!(predictions.is_ok());
    }

    #[test]
    fn random_floats_are_generated() {
        let expected_length = 5;
        let actual_length = gen_random_floats(expected_length).len();
        assert_eq!(actual_length, expected_length);
    }
}

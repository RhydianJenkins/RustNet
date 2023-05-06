mod perceptron;
mod random_float_generator;
mod trainer;

use perceptron::Perceptron;
use random_float_generator::gen_random_floats;
use trainer::Trainer;

const NUM_INPUTS: usize = 20;

pub fn generate_predictions() -> Result<f64, ()> {
    let random_weights = gen_random_floats(NUM_INPUTS);
    let bias = 1.0;
    let mut perceptron = Perceptron::new(random_weights, bias);

    // train (TODO loop until error is low enough)
    for _ in 0..10000 {
        let trainer = Trainer::new();
        perceptron.train(&trainer);
    }

    // predict
    let input = vec![1.0, 2.0];
    Ok(perceptron.feed_forward(&input))
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

mod perceptron;

use perceptron::Perceptron;

pub fn generate_predictions() -> Result<f64, ()> {
    let random_weights = gen_random_floats()?;
    let bias = 1.0;
    let perceptron = Perceptron::new(random_weights, bias);

    // TODO: The input coordinates' (or previous neurons'?) values
    let input = &[1.0, 2.0];

    Ok(perceptron.feed_forward(input))
}

fn gen_random_floats() -> Result<Vec<f64>, ()> {
    let random_floats = vec![0.1, 0.2, 0.3, 0.4, 0.5];

    Ok(random_floats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_floats_are_generated() {
        let result = gen_random_floats().unwrap();
        let expected = vec![0.1, 0.3, 0.3, 0.4, 0.5];
        assert_eq!(result, expected);
    }
}

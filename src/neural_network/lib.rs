mod perceptron;

use perceptron::Perceptron;

pub fn public_function() -> f64 {
    let perceptron = Perceptron::new(vec![1.0, 2.0, 3.0], 4.0);

    perceptron.predict(&[1.0, 2.0])
}

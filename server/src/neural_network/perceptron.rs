use std::f64::consts::E;

#[derive(Debug)]
pub struct Perceptron {
    weights: Vec<f64>,
    bias: f64,
}

const LEARNING_RATE: f64 = 0.01;

impl Perceptron {
    pub fn new(weights: Vec<f64>, bias: f64) -> Perceptron {
        Perceptron { weights, bias }
    }

    /*
     * For every input, multiply that input by its weight.
     * Sum all of the weighted inputs.
     * Compute the output of the perceptron based on that sum passed through an activation function
     * (the sign of the sum).
     */
    pub fn feed_forward(&self, inputs: &Vec<f64>) -> f64 {
        let weighted_sum: f64 = inputs
            .iter()
            .zip(self.weights.iter())
            .map(|(input, weight)| input * weight)
            .sum::<f64>();

        sigmoid(weighted_sum)
    }

    pub fn train(&mut self, inputs: &Vec<f64>, cost: f64) -> f64 {
        let new_weights = self
            .weights
            .iter()
            .zip(inputs.iter())
            .map(|(weight, training_input)| {
                sigmoid(weight * training_input * (1.0 - cost) * LEARNING_RATE)
            })
            .collect::<Vec<f64>>();

        self.weights = new_weights;

        self.feed_forward(&inputs)
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! sigmoid_tests {
        ($($name:ident: $value:expr,)*) => {$(
            #[test]
            fn $name() {
                let result = sigmoid($value);
                assert!(result >= 0.0 && result <= 1.0);
            }
        )*}
    }

    sigmoid_tests! {
        sig_0: -10.0,
        sig_1: -0.0,
        sig_2: 0.0,
        sig_3: 0.5,
        sig_4: 1.0,
        sig_5: 1000.0,
    }
}

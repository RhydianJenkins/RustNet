pub struct Perceptron {
    weights: Vec<f64>,
    bias: f64,
}

impl Perceptron {
    pub fn new(weights: Vec<f64>, bias: f64) -> Perceptron {
        Perceptron { weights, bias }
    }

    pub fn predict(&self, input: &[f64]) -> f64 {
        let sum: f64 = input
            .iter()
            .zip(self.weights.iter())
            .map(|(x, w)| x * w)
            .sum();

        sum + self.bias
    }
}

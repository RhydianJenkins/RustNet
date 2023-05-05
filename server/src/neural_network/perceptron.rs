pub struct Perceptron {
    weights: Vec<f64>,
    bias: f64,
}

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
    pub fn feed_forward(&self, input: Vec<f64>) -> f64 {
        let sum: f64 = input
            .iter()
            .zip(self.weights.iter())
            .map(|(x, w)| x * w)
            .sum();

        self.activate(sum)
    }

    /*
     * TODO
     */
    fn activate(&self, sum: f64) -> f64 {
        sum + self.bias
    }
}

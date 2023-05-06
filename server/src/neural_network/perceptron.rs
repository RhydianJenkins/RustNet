use super::training_data::TrainingData;

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
    pub fn feed_forward(&self, input: &Vec<f64>) -> f64 {
        let sum: f64 = input
            .iter()
            .zip(self.weights.iter())
            .map(|(x, w)| x * w)
            .sum();

        self.activate(sum)
    }

    fn activate(&self, sum: f64) -> f64 {
        sum + self.bias
    }

    pub fn train(&mut self, training_data: &TrainingData) {
        let guess = self.feed_forward(&training_data.inputs);
        let error = training_data.desired_answer - guess;
        let weights: Vec<f64> = self
            .weights
            .iter()
            .zip(training_data.inputs.iter())
            .map(|(w, x)| w + x * error * LEARNING_RATE)
            .collect();

        self.weights = weights;
        self.bias = self.bias + error * LEARNING_RATE;
    }
}

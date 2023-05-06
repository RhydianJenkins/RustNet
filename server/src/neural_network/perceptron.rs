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
     *
     * TODO This will feed forward to all perceptrons in the next layer.
     */
    pub fn feed_forward(&self, inputs: &Vec<f64>) -> f64 {
        let sum: f64 = inputs
            .iter()
            .zip(self.weights.iter())
            .map(|(input, weight)| input * weight)
            .sum();

        self.activate(sum)
    }

    fn activate(&self, sum: f64) -> f64 {
        sum + self.bias
    }

    /*
     * TODO Do we also feed forward to all perceptrons in the next layer?
     */
    pub fn train(&mut self, training_data: &TrainingData) -> f64 {
        let guess = self.feed_forward(&training_data.inputs);
        let error = training_data.desired_answer - guess;
        let weights: Vec<f64> = self
            .weights
            .iter()
            .zip(training_data.inputs.iter())
            .map(|(weight, training_input)| weight + training_input * error * LEARNING_RATE)
            .collect();

        self.weights = weights;
        self.bias = self.bias + error * LEARNING_RATE;

        guess
    }
}

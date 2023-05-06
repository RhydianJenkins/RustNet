use super::random_float_generator::gen_random_floats;

pub struct TrainingData {
    pub inputs: Vec<f64>,
    pub desired_answer: f64,
}

impl TrainingData {
    pub fn new() -> TrainingData {
        let inputs = gen_random_floats(20);
        let desired_answer = calculate_correct_answer(&inputs);

        TrainingData {
            inputs,
            desired_answer,
        }
    }
}

/*
 * Supervised learning.
 * This can be anything.
 * Whatever answer we *want* the perceptrons to give us with these given inputs.
 */
fn calculate_correct_answer(inputs: &Vec<f64>) -> f64 {
    inputs.iter().sum()
}

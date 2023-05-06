use super::random_float_generator::gen_random_floats;

pub struct Trainer {
    pub inputs: Vec<f64>,
    pub desired_answer: f64,
}

impl Trainer {
    pub fn new() -> Trainer {
        let inputs = gen_random_floats(20);
        let desired_answer = calculate_correct_answer(&inputs);

        Trainer {
            inputs,
            desired_answer,
        }
    }
}

fn calculate_correct_answer(inputs: &Vec<f64>) -> f64 {
    inputs.iter().sum()
}

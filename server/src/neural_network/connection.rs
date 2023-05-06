use super::perceptron::Perceptron;

pub struct Connection<'a> {
    pub weight: f64,
    pub perceptron_1: &'a Perceptron,
    pub perceptron_2: &'a Perceptron,
}

impl<'a> Connection<'a> {
    pub fn new(
        weight: f64,
        perceptron_1: &'a Perceptron,
        perceptron_2: &'a Perceptron,
    ) -> Connection<'a> {
        Connection {
            weight,
            perceptron_1,
            perceptron_2,
        }
    }

    pub fn get_left_perceptron(&self) -> &'a Perceptron {
        self.perceptron_1
    }

    pub fn get_right_perceptron(&self) -> &'a Perceptron {
        self.perceptron_2
    }
}

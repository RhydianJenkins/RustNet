use rand::{thread_rng, Rng};

const MIN: f64 = 0.0;
const MAX: f64 = 1.0;

pub fn gen_random_floats(number: usize) -> Vec<f64> {
    let values: Vec<f64> = (0..number).map(|_| gen_random_float()).collect();

    values
}

pub fn gen_random_float() -> f64 {
    thread_rng().gen_range::<f64, _>(MIN..MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_floats_are_generated() {
        let expected_length = 5;
        let actual_length = gen_random_floats(expected_length).len();
        assert_eq!(actual_length, expected_length);
    }
}

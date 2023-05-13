pub mod data_loader;
pub mod network;
mod perceptron;
pub mod random_float_generator;

use data_loader::load_data;
use network::Network;

pub fn generate_trained_network() -> Network {
    let training_dataset = load_data("train").unwrap();
    let mut network = Network::new();

    network.train(&training_dataset);
    network
}

use std::{fs::read, path::Path, process::Command};

const DOWNLOAD_SCRIPT_PATH: &str = "data/download";

#[derive(Debug)]
pub struct MnistImage {
    pub image: Vec<f64>,
    pub desired_output: Vec<f64>,
}

fn download_dataset() -> Result<(), std::io::Error> {
    println!("Downloading MNIST dataset...");

    let download = Command::new(DOWNLOAD_SCRIPT_PATH).output()?;
    if !download.status.success() {
        panic!(
            "Failed to download MNIST dataset: {}",
            String::from_utf8_lossy(&download.stderr)
        );
    }

    Ok(())
}

fn get_desired_answer(input: u8) -> Vec<f64> {
    (0..10)
        .map(|x| if x == input { 1.0 } else { 0.0 })
        .collect::<Vec<f64>>()
}

fn load_data_from_file(
    image_filename: &str,
    label_filename: &str,
) -> Result<Vec<MnistImage>, std::io::Error> {
    println!("Loading images...");

    let images = read(image_filename).expect("Failed to read images file");
    let labels = read(label_filename).expect("Failed to read images file");

    let mapped_images = images
        .chunks(28 * 28)
        .zip(labels.iter())
        .map(|(chunk, label)| {
            let normalized_chunk = chunk.to_vec().iter().map(|x| *x as f64 / 255.0).collect();
            let desired_output = get_desired_answer(*label);

            MnistImage {
                image: normalized_chunk,
                desired_output,
            }
        })
        .collect::<Vec<MnistImage>>();

    Ok(mapped_images)
}

pub fn load_dataset(dataset_name: &str) -> Result<Vec<MnistImage>, std::io::Error> {
    let images_filename = format!("{}-images-idx3-ubyte", dataset_name);
    let labels_filename = format!("{}-labels-idx1-ubyte", dataset_name);

    if !Path::new(&images_filename).exists() {
        download_dataset().expect("Failed to download MNIST dataset");
    }

    let mnist_images = load_data_from_file(&images_filename, &labels_filename)?;

    Ok(mnist_images)
}

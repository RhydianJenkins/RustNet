use byteorder::{BigEndian, ReadBytesExt};
use flate2::read::GzDecoder;
use serde::Serialize;
use std::fs::File;
use std::io::{Cursor, Read};

const DATA_DIR: &str = "data";

#[derive(Debug, Serialize, Clone)]
pub struct MnistImage {
    pub image: Vec<f64>,
    pub classification: u8,
    pub desired_output: Vec<f64>,
}

#[derive(Debug)]
struct MnistData {
    sizes: Vec<i32>,
    data: Vec<u8>,
}

impl MnistData {
    fn new(f: &File) -> Result<MnistData, std::io::Error> {
        let mut gz = GzDecoder::new(f);
        let mut contents: Vec<u8> = Vec::new();
        gz.read_to_end(&mut contents)?;
        let mut r = Cursor::new(&contents);

        let magic_number = r.read_i32::<BigEndian>()?;

        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        match magic_number {
            2049 => {
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            _ => panic!(),
        }

        r.read_to_end(&mut data)?;

        Ok(MnistData { sizes, data })
    }
}

pub fn load_data(dataset_name: &str) -> Result<Vec<MnistImage>, std::io::Error> {
    let labels_filename = format!("{}/{}-labels-idx1-ubyte.gz", DATA_DIR, dataset_name);
    let images_filename = format!("{}/{}-images-idx3-ubyte.gz", DATA_DIR, dataset_name);
    let label_data = &MnistData::new(&(File::open(labels_filename))?)?;
    let images_data = &MnistData::new(&(File::open(images_filename))?)?;
    let mut images: Vec<Vec<f64>> = Vec::new();
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;

    for i in 0..images_data.sizes[0] as usize {
        let start = i * image_shape;
        let image_data = images_data.data[start..start + image_shape].to_vec();
        let image_data: Vec<f64> = image_data.into_iter().map(|x| x as f64 / 255.).collect();
        images.push(image_data);
    }

    let classifications: Vec<u8> = label_data.data.clone();

    let desired_outputs = classifications.iter().map(|&x| get_desired_answer(x));

    let mut ret: Vec<MnistImage> = Vec::new();

    for ((image, &classification), desired_output) in images
        .into_iter()
        .zip(&classifications)
        .zip(desired_outputs)
    {
        ret.push(MnistImage {
            image,
            classification,
            desired_output,
        })
    }

    Ok(ret)
}

fn get_desired_answer(input: u8) -> Vec<f64> {
    (0..10)
        .map(|x| if x == input { 1.0 } else { 0.0 })
        .collect()
}

mod neural_network;

use actix_cors::Cors;
use actix_web::main;
use actix_web::web::{Data, Path};
use actix_web::{get, middleware::Logger, post, web, App, HttpResponse, HttpServer, Responder};
use neural_network::network::Network;
use neural_network::{generate_predictions, generate_trained_network};
use serde::Serialize;
use std::io::Result;

use crate::neural_network::data_loader::load_data;

struct AppState {
    network: Network,
}

#[derive(Serialize)]
struct PredictionResponse {
    predictions: Vec<f64>,
    desired_output: Vec<f64>,
}

#[get("/")]
async fn get_health() -> impl Responder {
    HttpResponse::Ok().body("Hello World! Use GET /network and POST /predictions.")
}

#[get("/network")]
async fn get_network(data: Data<AppState>) -> impl Responder {
    web::Json(data.network.clone())
}

#[get("/data/{index}")]
async fn get_data(path: Path<usize>) -> impl Responder {
    let index = path.into_inner();
    let training_dataset = load_data("t10k").unwrap();

    if index >= training_dataset.len() {
        return web::Json(Option::None);
    }

    let mnist_image = training_dataset
        .get(index)
        .expect("Failed to read beyond dataset size");

    web::Json(Option::Some(mnist_image.clone()))
}

#[post("/predictions")]
async fn post_predictions(data: Data<AppState>) -> impl Responder {
    let training_dataset = load_data("t10k").unwrap();
    let mnist_image = training_dataset.get(1).unwrap();
    let inputs = &mnist_image.image;
    let desired_output = &mnist_image.desired_output;
    let predictions = generate_predictions(&data.network, inputs).unwrap();

    web::Json(PredictionResponse {
        predictions,
        desired_output: desired_output.clone(),
    })
}

#[main]
async fn main() -> Result<()> {
    let data = Data::new(AppState {
        network: generate_trained_network(),
    });

    HttpServer::new(move || {
        App::new()
            .app_data(data.clone())
            .wrap(Cors::permissive())
            .wrap(Logger::default())
            .service(get_data)
            .service(get_health)
            .service(get_network)
            .service(post_predictions)
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}

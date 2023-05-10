mod neural_network;

use actix_cors::Cors;
use actix_web::main;
use actix_web::web::Data;
use actix_web::{get, middleware::Logger, post, web, App, HttpResponse, HttpServer, Responder};
use neural_network::network::Network;
use neural_network::random_float_generator::gen_random_floats;
use neural_network::{generate_predictions, generate_trained_network};
use serde::Serialize;
use std::io::Result;

use crate::neural_network::network::NUM_RAW_INPUTS;

struct AppState {
    network: Network,
}

#[derive(Serialize)]
struct ResponseBody {
    predictions: Vec<f64>,
}

#[get("/")]
async fn get_health() -> impl Responder {
    HttpResponse::Ok().body("Hello World! Use GET /network and POST /predictions.")
}

#[get("/network")]
async fn get_network(data: Data<AppState>) -> impl Responder {
    web::Json(data.network.clone())
}

#[post("/predictions")]
async fn post_predictions(data: Data<AppState>) -> impl Responder {
    let inputs = gen_random_floats(NUM_RAW_INPUTS);
    let predictions = generate_predictions(&data.network, &inputs).unwrap();

    web::Json(ResponseBody { predictions })
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
            .service(get_health)
            .service(get_network)
            .service(post_predictions)
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}

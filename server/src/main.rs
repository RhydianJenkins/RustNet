mod neural_network;

use actix_cors::Cors;
use actix_web::main;
use actix_web::web::{Data, Path};
use actix_web::{get, middleware::Logger, post, web, App, HttpResponse, HttpServer, Responder};
use neural_network::data_loader::load_data;
use neural_network::generate_trained_network;
use neural_network::network::Network;
use serde::{Deserialize, Serialize};

struct AppState {
    network: Network,
}

#[derive(Debug, Deserialize)]
struct PredictionRequest {
    inputs: Vec<f64>,
}

#[derive(Serialize)]
struct PredictionResponse {
    hidden_1_outputs: Vec<f64>,
    hidden_2_outputs: Vec<f64>,
    outputs: Vec<f64>,
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
async fn post_predictions(
    request: web::Json<PredictionRequest>,
    data: Data<AppState>,
) -> impl Responder {
    let (hidden_1_outputs, hidden_2_outputs, outputs) = data.network.feed_forward(&request.inputs);

    web::Json(PredictionResponse {
        hidden_1_outputs,
        hidden_2_outputs,
        outputs,
    })
}

#[main]
async fn main() -> Result<(), std::io::Error> {
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

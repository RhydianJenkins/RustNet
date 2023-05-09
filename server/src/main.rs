mod neural_network;

use actix_cors::Cors;
use actix_web::main;
use actix_web::web::Data;
use actix_web::{get, middleware::Logger, post, web, App, HttpResponse, HttpServer, Responder};
use neural_network::network::Network;
use neural_network::{generate_predictions, generate_trained_network};
use serde::Serialize;
use std::io::Result;

struct AppState {
    network: Network,
}

#[derive(Serialize)]
struct ResponseBody {
    predictions: Vec<f64>,
}

#[get("/health")]
async fn checkhealth() -> impl Responder {
    HttpResponse::Ok().body("Hello World!")
}

#[post("/predictions")]
async fn test(data: Data<AppState>) -> impl Responder {
    let predictions = generate_predictions(&data.network).unwrap();

    web::Json(ResponseBody { predictions })
}

#[main]
async fn main() -> Result<()> {
    HttpServer::new(|| {
        let data = Data::new(AppState {
            network: generate_trained_network(),
        });

        App::new()
            .app_data(data.clone())
            .wrap(Cors::permissive())
            .wrap(Logger::default())
            .service(checkhealth)
            .service(test)
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}

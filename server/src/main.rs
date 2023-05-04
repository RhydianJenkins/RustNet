mod neural_network;

use actix_cors::Cors;
use actix_web::{get, middleware::Logger, post, web, App, HttpResponse, HttpServer, Responder};
use neural_network::generate_predictions;
use serde::Serialize;

#[derive(Serialize)]
struct ResponseBody {
    testing: f64,
}

#[get("/checkhealth")]
async fn checkhealth() -> impl Responder {
    HttpResponse::Ok().body("Hello World!")
}

#[post("/test")]
async fn test() -> impl Responder {
    let predictions = generate_predictions().unwrap_or(0.0);

    web::Json(ResponseBody {
        testing: predictions,
    })
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .wrap(Cors::permissive())
            .wrap(Logger::default())
            .service(checkhealth)
            .service(test)
    })
    .bind(("0.0.0.0", 8080))?
    .run()
    .await
}

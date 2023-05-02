use actix_web::{get, post, App, HttpResponse, HttpServer, Responder};
use neural_network::generate_predictions;

#[get("/checkhealth")]
async fn checkhealth() -> impl Responder {
    HttpResponse::Ok().body("Hello World!")
}

#[post("/test")]
async fn test() -> impl Responder {
    let predictions = generate_predictions().unwrap_or(0.0);

    HttpResponse::Ok().body(predictions.to_string())
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| App::new().service(checkhealth).service(test))
        .bind(("127.0.0.1", 8080))?
        .run()
        .await
}

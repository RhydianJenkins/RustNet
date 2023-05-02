use actix_web::{get, App, HttpResponse, HttpServer, Responder};
use neural_network::public_function;

#[get("/checkhealth")]
async fn checkhealth() -> impl Responder {
    let returned_value = public_function();

    HttpResponse::Ok().body(returned_value.to_string())
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| App::new().service(checkhealth))
        .bind(("127.0.0.1", 8080))?
        .run()
        .await
}

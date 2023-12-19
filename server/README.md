# Rust backend for the neural network API

## Local Development

```bash
# If you've not run this before, be sure to download the mnist training data.
cd data && ./download

# Combile/run the app, which will kick of the training
cargo run

# ... in another window
curl http://localhost:8080
```

import { Component } from "solid-js";
import DrawableCanvas from "./DrawableCanvas";
import styles from "./App.module.css";

const App: Component = () => {
  return (
    <main class={styles.container}>
      <h1>Rust Net</h1>
      <DrawableCanvas />
    </main>
  );
};

export default App;

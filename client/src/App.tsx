import { Component, onMount } from "solid-js";

import styles from "./App.module.css";

const SCALE = 10;
const CANVAS_WIDTH = 28;
const CANVAS_HEIGHT = 28;
const GET_IMAGE_ENDPOINT = "http://localhost:8080/data/3";

type ImageType = {
    desired_output: number[];
    label: string;
    image: number[];
}

const initCanvas = (): CanvasRenderingContext2D => {
  const canvas = document.getElementById("canvas") as HTMLCanvasElement|null;

  if (!canvas) {
    throw Error("Could not get canvas element");
  }

  const ctx = canvas.getContext("2d");
  canvas.width = CANVAS_WIDTH * SCALE;
  canvas.height = CANVAS_HEIGHT * SCALE;
  canvas.setAttribute("style", "border: 2px solid white");

  if (!ctx) {
    throw Error("Could not get canvas context");
  }

  return ctx;
};

const fetchImage = async(): Promise<ImageType> => {
  const response = await fetch(GET_IMAGE_ENDPOINT);
  return await response.json();
};

const drawImage = (ctx: CanvasRenderingContext2D, imageData: ImageType): void => {
  imageData.image.forEach((pixel, index) => {
    ctx.fillStyle = `rgba(0, 0, 0, ${pixel})`;
    const x = index % CANVAS_WIDTH * SCALE;
    const y = index / CANVAS_HEIGHT * SCALE;
    ctx.fillRect(x, y, SCALE, SCALE);
  });
};

const App: Component = () => {
  onMount(async() => {
    const ctx = initCanvas();
    const imageData = await fetchImage();
    drawImage(ctx, imageData);
  });

  return (
    <main class={styles.container}>
      <h1>Rust Net</h1>
      <canvas id="canvas" />
    </main>
  );
};

export default App;

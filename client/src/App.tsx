import { Component, onMount } from "solid-js";

import styles from "./App.module.css";

const CANVAS_WIDTH = 512;
const CANVAS_HEIGHT = 512;

const NUM_POINTS = 100;

type Point = {
    x: number;
    y: number;
};

const generatePoint = (): Point => {
  const x = Math.random() * CANVAS_WIDTH;
  const y = Math.random() * CANVAS_HEIGHT;
  return {x, y};
};

const generatePoints = (ctx: CanvasRenderingContext2D): Point[] => {
  const points = [];
  for (let i = 0; i < NUM_POINTS; i++) {
    points.push(drawPoint(ctx));
  }
  return points;
};

const drawPoint = (ctx: CanvasRenderingContext2D): Point => {
  const {x, y} = generatePoint();

  ctx.fillStyle = "red";
  ctx.fillRect(x, y, 10, 10);
  ctx.stroke();

  return {x, y};
};

const initCanvas = (): CanvasRenderingContext2D => {
  const canvas = document.getElementById("canvas") as HTMLCanvasElement|null;

  if (!canvas) {
    throw Error("Could not get canvas element");
  }

  const ctx = canvas.getContext("2d");
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;
  canvas.setAttribute("style", "border: 2px solid white");

  if (!ctx) {
    throw Error("Could not get canvas context");
  }

  return ctx;
};

const App: Component = () => {
  onMount(async() => {
    const ctx = initCanvas();
    const points = generatePoints(ctx);
    await fetch("http://localhost:8080/test", {
      method: "POST",
      body: JSON.stringify({
        points,
      }),
    });
  });

  return (
    <main class={styles.container}>
      <h1>Rust Net</h1>
      <canvas id="canvas" />
    </main>
  );
};

export default App;

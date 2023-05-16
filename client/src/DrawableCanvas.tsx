import { onCleanup, onMount } from "solid-js";

const SCALE = 10;
const CANVAS_WIDTH = 28;
const CANVAS_HEIGHT = 28;
const LINE_WIDTH = 20;

type PredictionResponseType = {
    hidden_1_outputs: number[],
    hidden_2_outputs: number[],
    outputs: number[],
}

type CoordinateType = {
    x: number;
    y: number;
}

let canvas: HTMLCanvasElement;
let ctx: CanvasRenderingContext2D;
let smallCanvas: HTMLCanvasElement;
let smallCtx: CanvasRenderingContext2D;
let coord: CoordinateType = { x: 0, y: 0 };

const initCanvas = () => {
  canvas = document.getElementById("drawable-canvas") as HTMLCanvasElement;

  if (!canvas) {
    throw Error("Could not get canvas element");
  }

  canvas.width = CANVAS_WIDTH * SCALE;
  canvas.height = CANVAS_HEIGHT * SCALE;
  canvas.setAttribute("style", "border: 1px solid white");

  ctx = canvas.getContext("2d")!;

  if (!ctx) {
    throw Error("Could not get canvas context");
  }

  ctx.lineWidth = LINE_WIDTH;
  ctx.lineCap = "round";
  ctx.strokeStyle = "#000";
  ctx.imageSmoothingQuality = "low";
};

const initSmallCanvas = () => {
  smallCanvas = document.getElementById("small-canvas") as HTMLCanvasElement;

  if (!smallCanvas) {
    throw Error("Could not get small canvas element");
  }

  smallCanvas.width = CANVAS_WIDTH;
  smallCanvas.height = CANVAS_HEIGHT;
  smallCanvas.setAttribute("style", "border: 1px solid white");

  smallCtx = smallCanvas.getContext("2d")!;

  if (!smallCtx) {
    throw Error("Could not get small canvas context");
  }

  smallCtx.lineWidth = 1;
  smallCtx.lineCap = "round";
  smallCtx.strokeStyle = "#000";
  smallCtx.imageSmoothingQuality = "low";
};

function draw(event: MouseEvent) {
  if (!ctx) {
    console.error("ctx is null");
    return;
  }

  ctx.beginPath();
  ctx.moveTo(coord.x, coord.y);
  reposition(event);
  ctx.lineTo(coord.x, coord.y);
  ctx.stroke();

  smallCtx.beginPath();
  smallCtx.moveTo(coord.x / SCALE, coord.y / SCALE);
  reposition(event);
  smallCtx.lineTo(coord.x / SCALE, coord.y / SCALE);
  smallCtx.stroke();
}

const reposition = (event: MouseEvent) => {
  coord.x = event.clientX - canvas.offsetLeft;
  coord.y = event.clientY - canvas.offsetTop;
};

const start = (event: MouseEvent): void => {
  document.addEventListener("mousemove", draw);
  reposition(event);
};

const stop = (): void => {
  document.removeEventListener("mousemove", draw);
};

const clearCanvas = (): void => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  smallCtx.clearRect(0, 0, smallCanvas.width, smallCanvas.height);
};

const getSmallCanvasData = (): number[] => {
  const { data } = smallCtx.getImageData(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
  const smallImageData = [];

  for (let i = 3; i < data.length; i += 4) {
    const alphaValue = data[i];
    const normalized = alphaValue / 255;
    smallImageData.push(normalized);
  }

  return smallImageData;
};

const fetchPredictions = async(inputs: number[]) => {
  const response = await fetch("http://localhost:8080/predictions", {
    method: "POST",
    headers: {
      ["Content-Type"]: "application/json",
    },
    body: JSON.stringify({ inputs }),
  });
  const predictions = await response.json() as PredictionResponseType;

  let highestPrediction = 0;
  let highestPredictionIndex = 0;

  predictions.outputs.forEach((prediction: number, index: number) => {
    if (prediction > highestPrediction) {
      highestPrediction = prediction;
      highestPredictionIndex = index;
    }
  });

  document.getElementById("prediction-output")!.innerText = `Prediction: ${highestPredictionIndex}`;
};

const DrawableCanvas = () => {
  onMount(async() => {
    initCanvas();
    initSmallCanvas();

    document.addEventListener("mousedown", start);
    document.addEventListener("mouseup", stop);
  });

  onCleanup(() => {
    document.removeEventListener("mousedown", start);
    document.removeEventListener("mouseup", stop);
  });

  return (
    <>
      <canvas id="drawable-canvas" />
      <canvas id="small-canvas" />
      <span>
        <button onClick={clearCanvas}>Clear</button>
        <button onClick={() => fetchPredictions(getSmallCanvasData())}>Predict</button>
      </span>
      <p id="prediction-output" />
    </>
  );
};

export default DrawableCanvas;

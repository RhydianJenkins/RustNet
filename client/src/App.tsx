import { Component, createSignal, onMount } from "solid-js";
import DrawableCanvas from "./DrawableCanvas";
import styles from "./App.module.css";
import NetworkDetails from "./NetworkDetails";

export type NeuronType = {
    weights: number[];
    bias: number;
};

export type NetworkDetailsType = {
    input_layer: NeuronType[];
    hidden_layer: NeuronType[];
    output_layer: NeuronType[];
    trained: boolean;
};

const fetchNetworkDetails = async(): Promise<NetworkDetailsType> => {
  const response = await fetch("http://localhost:8080/network", {
    headers: {
      accept: "application/json",
    },
  });

  return await response.json();
};

const App: Component = () => {
  const [ networkDetails, setNetworkDetails ] = createSignal<NetworkDetailsType>({} as NetworkDetailsType);

  onMount(async() => {
    const netwokrDetails: NetworkDetailsType = await fetchNetworkDetails();
    setNetworkDetails(netwokrDetails);
  });

  return (
    <main class={styles.container}>
      <h1>Rust Net</h1>
      <DrawableCanvas />
      <NetworkDetails networkDetails={networkDetails()}/>
    </main>
  );
};

export default App;

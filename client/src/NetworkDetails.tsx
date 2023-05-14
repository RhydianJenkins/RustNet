import { NetworkDetailsType } from "./App";
import styles from "./App.module.css";

interface Props {
    networkDetails: NetworkDetailsType;
}

const NetworkDetails = (props: Props) => {
  return (
    <section>
      <h2>Network Details</h2>
      <div class={styles.networkDetails}>
        <span>Hidden 1: <pre>{JSON.stringify(props.networkDetails.input_layer, null, 2)}</pre></span>
        <span>Hidden 2: <pre>{JSON.stringify(props.networkDetails.hidden_layer, null, 2)}</pre></span>
        <span>Output: <pre>{JSON.stringify(props.networkDetails.output_layer, null, 2)}</pre></span>
      </div>
    </section>
  );
};

export default NetworkDetails;

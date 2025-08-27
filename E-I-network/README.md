# Randomly Coupled Izhikevich Network

This repository simulates a network of randomly coupled spiking neurons using the Izhikevich model.  
The network consists of **800 excitatory** and **200 inhibitory** neurons with randomly generated synaptic connections.

The simulation follows the original implementation described by **Eugene M. Izhikevich (2003)** and reproduces key emergent dynamics observed in randomly coupled cortical networks.

## Simulation Details

- **Model**: Izhikevich spiking neuron model
- **Network size**: 1000 neurons (800 excitatory, 200 inhibitory)
- **Synapses**: Random connectivity matrix `S`  
  - Excitatory → positive weights  
  - Inhibitory → negative weights
- **Integration**: Euler method, 0.5 ms step size (two half-steps per ms)
- **Duration**: 1000 ms
- **Outputs**: Membrane potentials `v` and spike timings

Spikes are **clamped at +30 mV** for visualization, following Izhikevich’s recommendation:  
> "All spikes were equalized at +30 mV by resetting v first to +30 mV and then to c."

## Results

The figure below shows the raster plot of spikes for all neurons in the network.

![Randomly Coupled Network](./randomly_coupled.png)

As described by Izhikevich, the randomly coupled network exhibits **emergent alpha (~10 Hz)** and **gamma (~40 Hz)** oscillations, demonstrating the characteristic interplay between excitatory and inhibitory populations.

## References

- Izhikevich, E. M. (2003).  
  *Simple model of spiking neurons*.  
  IEEE Transactions on Neural Networks, **14**(6), 1569–1572.  
  [https://www.izhikevich.org/publications/spikes.pdf](https://www.izhikevich.org/publications/spikes.pdf)


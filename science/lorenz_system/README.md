### Lorenz system

The Lorenz system refers to a system of ordinary differential equations that displays chaotic behavior. It was first introduced by Edward Lorenz, a meteorologist, while studying atmospheric convection.

The Lorenz system consists of three nonlinear differential equations:

```
dx/dt = C1 * (y - x)
dy/dt = x * (C2 - z) - y
dz/dt = x * y - C3 * z
```

In these equations, x, y, and z are variables representing the state of the system, and t represents time. The system also has three parameters: C1, C2, and C3, which control the behavior of the system. These parameters are typically set to specific values to observe interesting dynamics.

The Lorenz system is known for its sensitivity to initial conditions, which leads to chaotic behavior. It exhibits complicated trajectories in phase space, characterized by strange attractors. These attractors are non-periodic and can have complex fractal structures.

The Lorenz system has found applications in various fields, such as physics, mathematics, and engineering, serving as a prototypical example of chaotic behavior. It has contributed to the study of dynamical systems, chaos theory, and the exploration of nonlinear dynamics.


### Data

Set the independent variable to be time t, and the dependent variables to be x, y, and z. We give a small amount of data when C1=10, C2=15, and C3=8/3 as training data.


### Training

You can use bash script `train.sh` to train this model.

```bash
sh train.sh
```

During the training process, it will print out the inversion results of the three parameters C1, C2, and C3 every 100 iters.

### Infer

Bash script `infer.sh` is used to infer the trained model.

```bash
sh infer.sh
```


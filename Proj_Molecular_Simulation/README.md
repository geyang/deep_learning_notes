# Tensorflow Implementation of Electrons on Helium Simulation

This project is a Electrons on Helium 2D simulation implemented with Tensorflow. 
As of now, it correctly finds the 2D electron configuration corresponds to the
lowest energy. However some more work needs to be done to tweak the optimizers,
so that the final configuration can be annealed even better.

**Transparent Math**: One of the goals of this project is to expose the parts
that you need to write a simple particle simulation with tensorflow. So the 
code is done in such way that the logic between the moving parts is relatively 
simple. You can take the code to run as-it-is, or you can look through the 
small code base and adapt it to your own.

## Tricky things

### Inverse Pairwise Distance Matrix has poles

When we calculate the pairwise interaction between the charges, the following
matrix pops up:

```
            [ 
                [ 0,           1/ r_{0, 0}, 1/ r_{0, 1}, ... ],
                [ 1/ r_{1, 0}, 0,           1/ r_{1, 2}, ... ],
                [ 1/ r_{3, 0}, 1/ r_{3, 2}, 0,           ... ],
                [ 1/ r_{4, 0}, 1/ r_{4, 2}, 1/ r_{4, 3}, ... ],
                ... 
            ]
```

A naive implementation of the inverse pairwise distance matrix has a automatically
generated gradient that goes to `nan`. This comes mostly from the square root 
that is taken on the Distance^2 matrix. So in this working version, we postpone 
taking the square root.

## TODOs
- [ ] **Sample and Figures** add more figures showing the annealed results.

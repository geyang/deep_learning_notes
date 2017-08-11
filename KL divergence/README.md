# Exploring difference in $\mathrm{KL}(P \Vert Q)$ vs $\mathrm{KL}(Q \Vert P)$ 


**For the full analysis, refer to the notebook here:** 
> [./Investigating KL Divergence as A Loss Metric](./Investigating%20KL%20Divergence%20as%20a%20loss%20metric.ipynb)

In this notebook, we investigate different divergence loss metrics in a simple optimization problem.

## Key Learning
1. The result depends on the form of $P(x)$. 
    
    P || Q and Q || P looked quite similar qualitatively when the inter-modal separation in the posterior is small. 
    
    ![small_inter-modal_separation](./Comparing%20KL%20P%20Q%20vs%20KL%20Q%20P%20with%20small%20separation.png)
    
    The textbook behavior only arises when this separation is large, in which case the Gaussian end up optimize towards one of the modes in a local minimum.
    
    ![large inter-modal separation](./Comparing%20KL%20P%20Q%20vs%20KL%20Q%20P%20with%20large%20separation.png)
    
2. This result above is makes sense:

    - In the case of $\mathrm{KL}(P \Vert Q)$, the reason why the gaussian tries to spread over the entire $P$ distribution is because the loss will be big anywhere $P(x_i) > 0$ if $Q(x_i)$ is small.

    - Now with $\mathrm{KL}(Q \Vert P)$, the gaussian would try to avoid places where $P(x_i)$ is zero for the same reason. As a result, when the separation between the two modes are large the area in between are close to zero which pushes the Gaussian to one side. On the other hand, if the two modes are close together, the inter-modal area have substantial (non-zero) distribution. As a result, the fitted Gaussian does not experience much "push" to either one side, and the fit with $\mathrm{KL}(P\Vert Q)$ is qualitatively similar to that of $\mathrm{KL}(Q\Vert P)$ in the end.
    
This analysis gave some surprising results at the beginning that turned out quite informative : )
     
### Reference:
- Bishop, *Pattern Recognition and Machine Learning, Chapter 10*.
import matplotlib.pyplot as plt
import numpy as np
import torch
import sqfa


mahalanobis = torch.linspace(0, 2, 50)   
mahalanobis_sq = mahalanobis ** 2 
fr_dist = torch.sqrt(torch.tensor(2.0)) * \
    torch.acosh(1 + mahalanobis_sq / 4)

plt.figure(figsize=(8, 6))
plt.plot(mahalanobis, fr_dist, label='Fisher-Rao', color='blue')
plt.plot(mahalanobis, mahalanobis_sq, label='Mahalanobis Squared', color='orange')
plt.xlabel('Mahalanobis Distance')
plt.ylabel('Distance')
plt.legend()
plt.savefig('distances_plot.pdf')


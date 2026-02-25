import torch
import matplotlib.pyplot as plt

#############
# Single dim comparison
#############

lim = 10.0
eigval_log = torch.linspace(-lim, lim, 51)
eigval = 10**eigval_log

fisher = torch.sqrt(torch.log(eigval)**2) / torch.sqrt(torch.tensor(2.0))

# Compute the Bayes error
def ll_ratio(x, sigma2):
    coef = 1/sigma2 - 1
    ll = -0.5 * (coef[:,None] * x**2 + torch.log(sigma2[:,None]))
    return ll.squeeze()


def sample_error_rate(sigma2, n_samples=100000):
    # H0 error rate
    x = torch.randn(n_samples)
    ll0 = ll_ratio(x, sigma2)
    aa = ll0 >= 0
    error_0 = torch.mean(aa.float(), dim=1)
    # H1 error rate
    x = torch.randn(n_samples) * torch.sqrt(sigma2[:,None])
    ll1 = ll_ratio(x, sigma2)
    bb = ll1 < 0
    error_1 = torch.mean(bb.float(), dim=1)
    # Total error rate
    error = 0.5 * (error_0 + error_1)
    return error

def theory_error_rate(sigma2):
    sigma = torch.sqrt(sigma2)
    eps = 1e-8
    num = 2.0 * sigma2 * torch.log(sigma.clamp_min(eps))
    den = (sigma2 - 1.0)

    x0_sq = num / den
    # For sigma extremely close to 1, replace with the limit value x0^2 -> 1
    close = den.abs() < 1e-6
    x0_sq = torch.where(close, torch.ones_like(x0_sq), x0_sq)

    # Guard against tiny negative due to rounding (shouldn't happen theoretically)
    x0 = torch.sqrt(torch.clamp(x0_sq, min=0.0))

    # standard normal CDF via erf
    def Phi(z):
        return 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0, dtype=z.dtype, device=z.device))))

    acc = 0.5 + (Phi(x0) - Phi(x0 / sigma)).abs()
    return 1-acc

#bayes_error = sample_error_rate(eigval, n_samples=1000000)
bayes_error = theory_error_rate(eigval)


accuracy = 1 - bayes_error2

# Set fontsize
plt.rcParams.update({'font.size': 14})

fig, ax = plt.subplots(1, 3, figsize=(13, 3))
ax[0].plot(eigval_log, accuracy)
ax[1].plot(eigval_log, torch.log(accuracy/bayes_error))
ax[2].plot(eigval_log, fisher)
ax[0].set_xlabel(r'$\log_{10}(\sigma^2)$')
ax[1].set_xlabel(r'$\log_{10}(\sigma^2)$')
ax[2].set_xlabel(r'$\log_{10}(\sigma^2)$')
ax[0].set_ylabel('Accuracy')
ax[1].set_ylabel('Log-odds correct')
ax[2].set_ylabel('Fisher-Rao distance')
plt.tight_layout()
#plt.show()
plt.savefig('plots/distances1D.pdf', dpi=300)
plt.close()


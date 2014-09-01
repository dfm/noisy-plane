import numpy as np
import matplotlib.pyplot as pl
import triangle
import h5py

# fname = "CF45"
# fname = "HF45"
# fname = "NF45"
# fname = "PF5"
fname = "NF5"

with h5py.File("samples_%s" %fname, "r") as f:
    samples = f["samples"][:, 50:, :]
nwalkers, n, ndim = samples.shape
flatchain = samples.reshape((-1, ndim))

mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                  zip(*np.percentile(flatchain, [16, 50, 84], axis=0)))
mres = np.array(mcmc_result)[:, 0]
print 'mcmc_result = ', mres
np.savetxt("parameters%s.txt" %fname, np.array(mcmc_result))

pl.clf()
ylabels = ['a', 'n', 'b']
xlims = [(0., 1.), (.2, .8), (0., 1.)]
for i in range(3):
    pl.subplot(3,1,i+1)
    pl.hist(flatchain[:,i], 50, color='w')
    pl.axvline(mres[i], color='r', label='%s = %.3f'%(ylabels[i], mres[i]))
    pl.ylabel(ylabels[i])
    pl.xlim(xlims[i])
    pl.legend()
pl.savefig("hist%s"%fname)

# fig_labels = ["$a$", "$n$", "$b$", "$Y$", "$V$", "$Z$", "$W$", "$X$", "$U$", "$P$"]
# fig = triangle.corner(flatchain, truths=mres, labels=fig_labels)
# fig.savefig("triangle%s.png" %fname)

# print("Plotting traces")
# pl.figure()
# for i in range(ndim):
#     pl.clf()
#     pl.plot(samples[:, :, i].T, 'k-', alpha=0.3)
#     pl.savefig("%s%s.png" %(i, fname))

import numpy as np
import matplotlib.pyplot as pl
import triangle
import h5py

# fname = "small_changes"
# fname = "garcia"
# fname = "_45"
# fname = "_40"
# fname = "_50"
# fname = "_45acf"
# fname = "_45_2acf"
# fname = "just_clusters"
# fname = "no_NGC6811"
# fname = "just_field"
# fname = "no_clusters"
# fname = "hyades"
# fname = "hyadesComa"
# fname = "hyadesSun"
fname = "just_clusters_no_NGC"

with h5py.File("samples_%s" %fname, "r") as f:
    samples = f["samples"][:, 50:, :]
nwalkers, n, ndim = samples.shape
flatchain = samples.reshape((-1, ndim))

mcmc_result = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                  zip(*np.percentile(flatchain, [16, 50, 84], axis=0)))
mres = np.array(mcmc_result)[:, 0]
print 'mcmc_result = ', mres
np.savetxt("parameters%s.txt" %fname, np.array(mcmc_result))

fig_labels = ["$a$", "$n$", "$b$", "$Y$", "$V$", "$Z$", "$W$", "$X$", "$U$", "$P$"]
fig = triangle.corner(flatchain, truths=mres, labels=fig_labels)
fig.savefig("triangle%s.png" %fname)

print("Plotting traces")
pl.figure()
for i in range(ndim):
    pl.clf()
#     pl.axhline(par_true[i], color = "r")
    pl.plot(samples[:, :, i].T, 'k-', alpha=0.3)
    pl.savefig("%s%s.png" %(i, fname))

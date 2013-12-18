from matplotlib import rc
rc("font", family="serif", size=12)
rc("text", usetex=True)

import daft

pgm = daft.PGM([3.6, 3.1], origin=[-0.8, -0.7])

pgm.add_plate(daft.Plate([-0.5, -0.5, 3, 2], label=r"stars $n=1,\cdots,N$",
                         shift=-0.1))

pgm.add_node(daft.Node("tT", r"$\theta_T$", 0, 2, fixed=True))
pgm.add_node(daft.Node("m", r"$m$", 1, 2))
pgm.add_node(daft.Node("tP", r"$\theta_P$", 2, 2, fixed=True))

pgm.add_node(daft.Node("T", r"$T_n$", 0, 1))
pgm.add_node(daft.Node("A", r"$A_n$", 1, 1))
pgm.add_node(daft.Node("P", r"$P_n$", 2, 1))

pgm.add_node(daft.Node("Tobs", r"$\hat{T}_n$", 0, 0, observed=True))
pgm.add_node(daft.Node("Aobs", r"$\hat{A}_n$", 1, 0, observed=True))
pgm.add_node(daft.Node("Pobs", r"$\hat{P}_n$", 2, 0, observed=True))

pgm.add_edge("tT", "T")
pgm.add_edge("tP", "P")
pgm.add_edge("T", "Tobs")
pgm.add_edge("P", "Pobs")
pgm.add_edge("m", "A")
pgm.add_edge("A", "Aobs")
pgm.add_edge("P", "A")
pgm.add_edge("T", "A")

pgm.render()
pgm.figure.savefig("pgm.pdf")

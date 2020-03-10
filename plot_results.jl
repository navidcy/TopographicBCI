fig, axs = subplots(ncols=2, nrows=nlayers, figsize=(10, 10))

MultilayerQG.updatevars!(prob)

for j in 1:nlayers
  sca(axs[j])
  cla()

  if j<nlayers
      q = vs.q[:, :, j]*1e8
  else
      q = (vs.q[:, :, j]-pr2.eta)*1e8
  end

  maxq = maximum(abs.(q))
  lev = range(-maxq, stop=maxq, length=10)
  contourf(x, y, q, lev)

  colorbar()
  title("q")
  xlabel("x")
  ylabel("y")
  # cb = colorbar(axs[j], fraction = 0.05, shrink = 0.5, pad = 0.1)
  # cb[:set_label](label = "Variable [units]", rotation=270)
  # xlim(-L/2, L/2)
  # ylim(-L/2, L/2)

  sca(axs[j+nlayers])
  cla()

  psi = vs.psi[:, :, j]*3*1.3e9
  maxpsi = maximum(abs.(psi))
  lev = range(-maxpsi, stop=maxpsi, length=10)

  contourf(x, y, psi, lev)
  colorbar()
  contour(x, y, psi, lev, colors="k")
  title("psi")
  xlabel("x")
  ylabel("y")
  # pcolormesh(x, y, vs.u[:, :, j])
  # cb = colorbar(axs[j+nlayers], fraction = 0.05, shrink = 0.5, pad = 0.1)
  # cb[:set_label](label = "Variable [units]", rotation=270)
  # xlim(-L/2, L/2)
  # ylim(-L/2, L/2)
end

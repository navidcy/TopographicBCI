# This script intergrates forwards the 2-layer QG equations with an imposed
# mean flow U(y) at the top-layer. By rescaling the amplitute of sol after
# nsubsteps we ensure that nonlinearities are kept small. This way after some
# iterations the flow field converges to the structure of the most unstable
# mode of baroclinic instability.
# 
# (To confirm that one can evolve in parallel the linearized 2-layer QG system.)

using
  PyPlot,
  FourierFlows,
  LinearAlgebra

using Statistics: mean
using Printf: @sprintf
using FFTW
using FFTW: rfft, irfft

global ke
global ke0

import GeophysicalFlows.MultilayerQG
import GeophysicalFlows.MultilayerQG: fwdtransform!, invtransform!, streamfunctionfrompv!, energies, fluxes


nx, ny, L = 128, 128, 2.0
gr = TwoDGrid(nx, L, ny, L)

nlayers = 2       # these choice of parameters give the
f0, g = 1, 1      # desired PV-streamfunction relations
H = [0.2, 0.8]   # q1 = Δψ1 + 25*(ψ2-ψ1), and
ρ = [4.0, 5.0]   # q2 = Δψ2 + 25/4*(ψ1-ψ2).

U = zeros(ny, nlayers)
U[:, 1] = @. sech(gr.y/0.2)^2

x, y = gridpoints(gr)
k0, l0 = gr.k[2], gr.l[2] # fundamental wavenumbers
eta = @. 3cos(10k0*x)*cos(10l0*y)
dt, stepper = 0.005, "FilteredAB3"

prob = MultilayerQG.Problem(nlayers=nlayers, nx=nx, U=U, Lx=L, f0=f0, g=g, H=H, ρ=ρ, eta=eta, dt=dt, stepper=stepper)
sol, cl, pr, vs, gr = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

prob2 = MultilayerQG.Problem(nlayers=nlayers, nx=nx, U=U, Lx=L, f0=f0, g=g, H=H, ρ=ρ, eta=eta, linear=true, dt=dt, stepper=stepper)
sol2, cl2, pr2, vs2, gr2 = prob2.sol, prob2.clock, prob2.params, prob2.vars, prob2.grid


qi1 = @. 1e-8*cos(2k0*x)*cos(3l0*y)      + 2e-8*cos(1k0*x)*cos(2l0*y) - 2e-8*cos(2k0*x)*cos(2l0*y)
qi2 = @. 4e-8*cos(4k0*x)*cos(3l0*y+2π/3) - 1e-8*cos(6k0*x)*cos(3l0*y) + 2e-8*cos(2k0*x)*cos(3l0*y)
qi3 = @. 2e-8*cos(2k0*x)*cos(3l0*y)      - 2e-8*cos(4k0*x)*cos(1l0*y)


qi = zeros(gr.nx, gr.ny, nlayers)
qi[:, :, 1] = 0*qi1 + 1e-5*randn(gr.nx, gr.ny)
qi[:, :, 2] = 0*qi2 + 1e-5*randn(gr.nx, gr.ny)
# qi[40, 40, 2] = 0.00001
# qi[:, :, 3] = 0*qi3 + 1e-10*randn(gr.nx, gr.ny)

T = Float64
effort=FFTW.MEASURE
rfftplan = plan_rfft(Array{T,3}(undef, gr.nx, gr.ny, nlayers), [1, 2]; flags=FFTW.MEASURE)

qh = rfft(qi, [1, 2])
qh2 = zeros(size(qh))*im
qh3= zeros(size(qh))*im

fwdtransform!(qh2, qi, pr)

mul!(qh3, rfftplan, qi)

println(norm(qh - qh2)/norm(qh))
println(norm(qh - qh3)/norm(qh))
q = zeros(size(qi))

invtransform!(q, qh2, pr)

println(norm(q - qi)/norm(qi))


psih = zeros(size(qh))*im
psi = zeros(size(q))

streamfunctionfrompv!(psih, qh, pr.invS, gr)

invtransform!(psi, psih, pr)

problin=prob2

MultilayerQG.set_q!(prob, q)
MultilayerQG.set_q!(problin, q)
ke = MultilayerQG.energies(prob)

fac = 1e-5
@. sol *= prob.timestepper.filter
sol .= fac*sol/maximum(abs.(sol))

@. sol2 *= problin.timestepper.filter
sol2 .= fac*sol2/maximum(abs.(sol2))

E = Diagnostic(energies, prob; nsteps=1)
diags = [E]


startwalltime = time()

nsubsteps = 500
growth = zeros(nsubsteps)
mom1thick = zeros(nsubsteps)
ke1ke2 = zeros(nsubsteps)
growthlin = zeros(nsubsteps)
mom1thicklin = zeros(nsubsteps)
ke1ke2lin = zeros(nsubsteps)

error = 1
i = 1

fig, axs = subplots(ncols=2, nrows=nlayers, figsize=(10, 10), num=1)

while error>1e-5
  MultilayerQG.updatevars!(prob)
  MultilayerQG.updatevars!(problin)
  
  # cfl = prob.clock.dt*maximum([maximum(vs.u)/gr.dx, maximum(vs.v)/gr.dy])
  # log = @sprintf("step: %04d, t: %d, cfl: %.2f, τ: %.2f min",
  #   prob.clock.step, prob.clock.t, cfl,
  #   (time()-startwalltime)/60)
  
  for j in 1:nlayers
  
    maxq = maximum(abs.(vs.q[:, :, j]))
    levels = range(-maxq[1], maxq[1], length=50)
  
    sca(axs[j])
    cla()
    contourf(x, y, vs.q[:, :, j], levels=levels)
    # cb = colorbar(axs[j], fraction = 0.05, shrink = 0.5, pad = 0.1)
    # cb[:set_label](label = "Variable [units]", rotation=270)
    xlim(-L/2, L/2)
    ylim(-L/2, L/2)
    title("PV, layer "*string(j))
  
    maxψ = maximum(abs.(vs.psi[:, :, j]))
    levels = range(-maxψ[1], maxψ[1], length=10)
  
    sca(axs[j+nlayers])
    cla()
    contourf(x, y, vs.psi[:, :, j], levels=levels)
    contour(x, y, vs.psi[:, :, j], levels=levels, colors="k")
    # pcolormesh(x, y, vs.u[:, :, j])
    # cb = colorbar(axs[j+nlayers], fraction = 0.05, shrink = 0.5, pad = 0.1)
    # cb[:set_label](label = "Variable [units]", rotation=270)
    xlim(-L/2, L/2)
    ylim(-L/2, L/2)
    title("streamfunction, layer "*string(j))
  end

  (ke0, pe0) = MultilayerQG.energies(prob)
  stepforward!(prob, diags, nsubsteps)
  (ke, pe) = MultilayerQG.energies(prob)
  (lateralfluxes, verticalfluxes) = fluxes(prob)
  sol .= fac*sol/maximum(abs.(sol))
  growth[i] =  mean(log.(ke[1]/ke0[1])/(2*nsubsteps*dt) + log.(ke[2]/ke0[2])/(2*nsubsteps*dt))
  mom1thick[i] = lateralfluxes[1]/verticalfluxes[1]
  ke1ke2[i] = ke[1]/ke[2]
  
  # (ke0lin, pe0lin) = MultilayerQG.energies(problin)
  # stepforward!(problin, diags, nsubsteps)
  # (kelin, pelin) = MultilayerQG.energies(problin)
  # (lateralfluxeslin, verticalfluxeslin) = fluxes(problin)
  # sol2 .= fac*sol2/maximum(abs.(sol2))
  # growthlin[i] =  mean(log.(kelin[1]/ke0lin[1])/(2*nsubsteps*dt) + log.(kelin[2]/ke0lin[2])/(2*nsubsteps*dt))
  # mom1thicklin[i] = lateralfluxeslin[1]/verticalfluxeslin[1]
  # ke1ke2lin[i] = kelin[1]/kelin[2]

  pause(0.001)
  
  figure(23)
  clf()
  subplot(311)
  plot(growth[1:i], "*")
  # plot(growthlin[1:i], ".r")
  title(L"growth rate")
  subplot(312)
  plot(mom1thick[1:i], "o")
  # plot(mom1thicklin[1:i], ".r")
  title(L"lateralfluxes_1 / verticalfluxes")
  subplot(313)
  plot(ke1ke2[1:i], "s")
  # plot(ke1ke2lin[1:i], ".r")
  title(L"ke_1 /  ke_2")
  if i>1
    global error = abs(growth[i]-growth[i-1])/abs(growth[i])
  end
  println(error)
  global i += 1

end

growth = growth[1:i-1]
mom1thick = mom1thick[1:i-1]
ke1ke2 = ke1ke2[1:i-1]
growthlin = growthlin[1:i-1]
mom1thicklin = mom1thicklin[1:i-1]
ke1ke2lin = ke1ke2lin[1:i-1]



fig, axs = subplots(ncols=2, nrows=nlayers, figsize=(10, 10), num=1)

for j in 1:nlayers

  maxq = maximum(abs.(vs.q[:, :, j]))
  levels = range(-maxq[1], maxq[1], length=50)

  sca(axs[j])
  cla()
  contourf(x, y, vs.q[:, :, j], levels=levels)
  # cb = colorbar(axs[j], fraction = 0.05, shrink = 0.5, pad = 0.1)
  # cb[:set_label](label = "Variable [units]", rotation=270)
  xlim(-L/2, L/2)
  ylim(-L/2, L/2)
  title("PV, layer "*string(j))
  
  maxψ = maximum(abs.(vs.psi[:, :, j]))
  levels = range(-maxψ[1], maxψ[1], length=10)

  sca(axs[j+nlayers])
  cla()
  contourf(x, y, vs.psi[:, :, j], levels=levels)
  contour(x, y, vs.psi[:, :, j], levels=levels, colors="k")
  # pcolormesh(x, y, vs.u[:, :, j])
  # cb = colorbar(axs[j+nlayers], fraction = 0.05, shrink = 0.5, pad = 0.1)
  # cb[:set_label](label = "Variable [units]", rotation=270)
  xlim(-L/2, L/2)
  ylim(-L/2, L/2)
  title("streamfunction, layer "*string(j))
end


pause(0.01)


figure(23)
clf()
subplot(311)
plot(growth, "*")
# plot(growthlin, ".r")
title(L"growth rate")

subplot(312)
plot(mom1thick, "o")
# plot(mom1thicklin, ".r")
title(L"lateralfluxes_1 / verticalfluxes")


subplot(313)
plot(ke1ke2, "s")
# plot(ke1ke2lin, ".r")
title(L"ke_1 /  ke_2")

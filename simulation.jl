using Pkg
Pkg.activate()

include("module/new_att.jl")

# =============== Initial Mixed State ============== #
N = 200
μ = 0.0005
ρ0, sites = initial_rho_mu(N, μ; normalize=true, conserve_qns=true)
println("Initial state created.")

# ============== State Evolution ============= #
t = 150 # total time of the Evolution
r = 300
cutoff = 1e-10
maxdim = 256
save_every = 1
ts = collect(range(0.0, t; length=r+1))
h = [nothing] # h vector

print_simulation_parameters(N, cutoff, t, r)

save_complete_state_evolution(
    "mpo_evolution_N$(N)_mu$(μ)_t$(t)_r$(r)_saveevery$(save_every).h5",
    sites,
    ρ0,
    ts,
    r;
    h = h,
    per = false,
    cutoff = cutoff,
    maxdim = maxdim,
    showprogress = true,
    hermitianize = true,
    normalize_every = 10,
    save_every = save_every,
    normalize_saved = true,
    params=Dict(
        "N" => N,
        "mu" => μ,
        "t_total" => t,
        "r" => r,
        "cutoff" => cutoff,
        "maxdim" => maxdim,
        "save_every" => save_every
    )
)

println("Done.")    

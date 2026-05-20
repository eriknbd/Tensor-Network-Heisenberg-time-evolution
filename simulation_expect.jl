using Pkg
Pkg.activate()

include("module/new_att.jl")

# =============== Initial Mixed State ============== #
N = 1400
μ = 0.001
ρ0, sites = initial_rho_mu(N, μ; normalize=true, conserve_qns=true)
println("Initial state created.")

# ============== State Evolution ============= #
t = 600 # total time of the Evolution
r = t*10
cutoff = 1e-10
maxdim = 400
expect_every = 10
ts = collect(range(0.0, t; length=r+1))
h = [nothing] # h vector
J = 1
current_kind = :both
current_bonds = nothing

print_simulation_parameters(N, cutoff, t, r)

avg_result = avg_state_evo(
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
    expect_every = expect_every,
    J = J,
    measure_currents = true,
    current_kind = current_kind,
    current_bonds = current_bonds,
)

save_avg_state_evo(
    "avg_state_evo_N$(N)_mu$(μ)_t$(t)_r$(r)_expectevery$(expect_every).h5",
    avg_result;
    params = Dict(
        "mu" => μ,
        "t_total" => t,
        "r" => r,
        "J" => J,
        "cutoff" => cutoff,
        "maxdim" => maxdim,
        "expect_every" => expect_every,
        "normalize_every" => 10,
        "current_kind" => string(current_kind)
    )
)

println("Done.")    

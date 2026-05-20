using Pkg
Pkg.activate(".")

include("module/new_PKZ_f.jl")

# =============== Initial Mixed State ============== #
N = 1600
μ = 0.001
ρ0, sites = initial_rho_mu(N, μ; normalize=true, conserve_qns=true)
println("Initial state created.")

# ============== State Evolution ============= #
t = 200
r = t * 10
maxdim = 512 * 2
expect_every = 10
cutoff = 1e-12
normalize_every = 10
ts = collect(range(0.0, t; length = r + 1))
h = [nothing]
J = 1
current_kind = :both

print_simulation_parameters(N, cutoff, t, r)

avg_result = complete_expectation_evo(
    sites,
    ρ0,
    ts,
    r;
    h = h,
    J = J,
    cutoff = cutoff,
    maxdim = maxdim,
    avg_every = expect_every,
    hermitianize = true,
    normalize_every = normalize_every,
    showprogress = true,
    current_kind = current_kind,
)

save_complete_expectation_evo(
    "kpz_expect_N$(N)_mu$(μ)_t$(t)_r$(r)_expectevery$(expect_every).h5",
    avg_result;
    params = Dict(
        "mu" => μ,
        "t_total" => t,
        "r" => r,
        "J" => J,
        "cutoff" => cutoff,
        "maxdim" => maxdim,
        "avg_every" => expect_every,
        "normalize_every" => normalize_every,
        "current_kind" => string(current_kind),
    ),
)

println("Done.")
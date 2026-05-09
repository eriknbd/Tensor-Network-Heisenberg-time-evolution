using ITensors, ITensorMPS
using Random
using Printf
using Plots
using LaTeXStrings
using ProgressMeter

# ========================================= #
# ============== MPS utilities ============ #
# ========================================= #

function _mps_distance(ψ::MPS, ϕ::MPS)
    ψϕ = inner(ψ, ϕ)

    # d = sqrt( ⟨ψ∣ψ⟩ + ⟨ϕ∣ϕ⟩ - 2ℜ⟨ψ∣ϕ⟩ )
    # Suponiendo estados normalizados: d = sqrt( 2 - 2ℜ⟨ψ∣ϕ⟩ ) 
    d2 = real(2 - 2*real(ψϕ))

    return sqrt(max(d2, 0.0))
end

function mps_distance(ψv, ϕv)
    n = size(ψv)[1]
    return [_mps_distance(ψv[i], ϕv[i]) for i in 1:n]
end

function Entropy_Bipartition(psi::MPS, b::Int; base = 2, tol = 1e-20)
    ψ = deepcopy(psi)    
    orthogonalize!(ψ, b)
    s_b = siteind(ψ, b)

    left_inds = b == 1 ? 
        (s_b,) : 
        (linkind(ψ, b - 1), s_b)

    _, S, _ = svd(ψ[b], left_inds)
    logfun = base == 2 ? log2 : log
    
    SvN = 0.0
    for n in 1:dim(S, 1)
        p = abs2(S[n, n])  # Safe, fast squaring
        if p > tol
            SvN -= p * logfun(p)
        end
    end
    
    return SvN
end

function Schmidt_Rank(psi::MPS, b; tol::Real = 1e-8)  
  psi = orthogonalize(psi, b)
  _, S, _ = svd(psi[b], (linkinds(psi, b-1)..., siteinds(psi, b)...))
  rank = 0
  for n = 1:dim(S, 1)
    s = S[n,n]
    if s > tol
      rank += 1
    end
  end
  return rank
end

_normalize!(psi:: MPS) = normalize!(psi)
function _normalize!(ρ::MPO)
    Z = tr(ρ)
    iszero(Z) && error("No se puede normalizar un MPO con traza cero.")
    ρ /= Z
    return ρ
end

# ========================================= #
# ============= MPO functions ============= #
# ========================================= #

# Inital States

const MU_REF = Ref(0.0)

# Operador local exp(+μ Sz)
ITensors.op(::OpName"ExpPlusMuSz", ::SiteType"S=1/2") =
  [exp(MU_REF[] / 2)  0.0
   0.0                exp(-MU_REF[] / 2)]

# Operador local exp(-μ Sz)
ITensors.op(::OpName"ExpMinusMuSz", ::SiteType"S=1/2") =
  [exp(-MU_REF[] / 2)  0.0
   0.0                 exp(MU_REF[] / 2)]

function initial_rho_mu(L::Int, μ::Real; normalize=true, conserve_qns=true)
    @assert iseven(L)

    # 1) Sitios físicos
    sites = siteinds("S=1/2", L; conserve_qns=conserve_qns)

    # 2) Fijamos el valor de μ que leerán los operadores definidos arriba
    MU_REF[] = float(μ)

    # 3) Primera mitad: exp(+μ Sz), segunda mitad: exp(-μ Sz)
    ops = [j <= L ÷ 2 ? "ExpPlusMuSz" : "ExpMinusMuSz" for j in 1:L]

    # 4) Construimos el MPO producto
    ρ = MPO(sites, ops)

    # 5) Normalización opcional: Tr(ρ)=1
    if normalize
        Z = (2 * cosh(μ / 2))^L
        ρ = ρ / Z
    end

    return ρ, sites
end

# Utilities

function _mpo_distance_frobenius(ρ::MPO, σ::MPO)
    Δ = ρ - σ
    return norm(Δ) # ∥ρ−σ∥_2 ​= √{Tr[(ρ−σ)†(ρ−σ)]}
end

# ⟨Oj​⟩ = Tr(ρOj​)/Tr(ρ)​
function local_expect_mpo(ρ::MPO, opname::String = "Sz"; normalize::Bool = true, ket_primelev::Int = 0, bra_primelev::Int = 1)
    N = length(ρ)

    # Physical indices of the MPO
    s  = [siteind(ρ, j; plev = ket_primelev) for j in 1:N]
    sp = [siteind(ρ, j; plev = bra_primelev) for j in 1:N]

    any(isnothing, s)  && error("Could not find ket site indices with prime level =$ket_primelev")
    any(isnothing, sp) && error("Could not find bra site indices with prime level=$bra_primelev")

    # Trace tensors for each site: contract local bra/ket legs with identity
    Ttr = Vector{ITensor}(undef, N)
    for j in 1:N
        Ttr[j] = ρ[j] * delta(dag(s[j]), dag(sp[j]))  # test tr(ρ[j])
    end

    # Right: R[N] = Ttr[N], R[N-1] = Ttr[N-1] * Ttr[N], R[1] = Ttr[1] * Ttr[2] * ... * Ttr[N] = Tr(ρ)
    R = Vector{ITensor}(undef, N + 1)
    R[N + 1] = ITensor(1.0)
    for j in N:-1:1
        R[j] = Ttr[j] * R[j + 1]
    end

    # Tr(ρ)
    Z = scalar(R[1])

    vals = Vector{ComplexF64}(undef, N)
    L = ITensor(1.0)

    for j in 1:N    # vals[j] = Ttr[1] * ... * Ttr[j-1] * (ρ[j] * Oj) * R[j+1]
        Oj = op(opname, s[j])
        vals[j] = scalar(L * ρ[j] * Oj * R[j + 1])
        L *= Ttr[j]
    end

    return normalize ? vals ./ Z : vals
end

function Entropy_Bipartition_Op_MPO(rho::MPO, b; base = 2, tol = 1e-20, normalize::Bool = false)

    ρ = deepcopy(rho)
    if normalize
        _normalize!(ρ)
    end
    orthogonalize!(ρ, b)

    left_inds = b == 1 ?
        Tuple(siteinds(ρ, b)) :
        tuple(linkind(ρ, b - 1), siteinds(ρ, b)...)

    _, S, _ = svd(ρ[b], left_inds)

    logfun = base == 2 ? log2 : log
    SvN = 0.0
    for n in 1:dim(S, 1)
        p = real(S[n, n]^2)
        if p > tol
            SvN -= p * logfun(p)
        end
    end
    return SvN
end

function partial_trace(rho::MPO, trsites::UnitRange{Int})
    N = length(rho)
    i, j = first(trsites), last(trsites)
    @assert 1 <= i <= j <= N

    # Build the traced block tensor E by tracing the physical legs of each MPO tensor in the range i:j.
    E = ITensor(1.0)
    for k in i:j
        s  = siteind(rho, k; plev=0)
        sp = siteind(rho, k; plev=1)
        E *= rho[k] * delta(dag(s), dag(sp))
    end


    if i == 1 && j == N
        return scalar(E)
    end

    # Output MPO on the remaining sites
    rhoA = MPO(N - length(trsites))
    n = 1

    for k in 1:(i - 2)
        rhoA[n] = rho[k]
        n += 1
    end

    if i > 1
        # left boundary tensor * traced block
        rhoA[n] = rho[i - 1] * E
        n += 1

        # right block
        for k in (j + 1):N
            rhoA[n] = rho[k]
            n += 1
        end
    else
        # absorb into first right tensor
        rhoA[n] = E * rho[j + 1]
        n += 1

        for k in (j + 2):N
            rhoA[n] = rho[k]
            n += 1
        end
    end

    return rhoA
end

partial_trace(rho::MPO, i::Int, j::Int) = partial_trace(rho, i:j)

function second_Renyi_entropy(rho::MPO, c::Int; base = 2)
    N = length(rho)
    @assert 1 <= c < N

    rhoA = partial_trace(rho, (c+1), N)

    TrA = tr(rhoA)
    
    logTr_rhoA2 = loginner(swapprime(dag(rhoA), 0 => 1), rhoA)
    
    if base == 2
        return -(logTr_rhoA2 - 2log(TrA)) / log(2)
    else
        return -(logTr_rhoA2 - 2log(TrA))
    end
end

function hermitian_mpo(rho::MPO)
    return swapprime(dag(rho),0,1)
end

function hermiticity_error(rho::MPO)
    rho_h = hermitian_mpo(rho)
    return norm(rho - rho_h) / norm(rho)
end

function hermitianize(rho::MPO; cutoff=1e-10, normalize::Bool = false)
    rhoH = hermitian_mpo(rho)
    rho_hermitian = 0.5 * add(rho, rhoH; cutoff=cutoff)
    if normalize
        _normalize!(rho_hermitian)
    end
    return rho_hermitian
end

function hermitianize!(rho::MPO; cutoff=1e-10, normalize::Bool = false)
    rho .= hermitianize(rho; cutoff=cutoff, normalize = normalize)
    return rho
end

# ========================================= #
# ================= Gates ================= #
# ========================================= #

function even_odd_Hamiltonian(hj, sites)
    N = length(sites)
    # Split bonds into odd and even sets:
    odd_bonds  = 1:2:N
    even_bonds = 2:2:N
    H_even = [hj(j, sites) for j in even_bonds]
    H_odd  = [hj(j, sites) for j in  odd_bonds]
    H_even, H_odd   # Tuple{Vector{ITensor}, Vector{ITensor}}
end

function time_operator(H)
  if eltype(H) == ITensor
    U_op1(tau) = exp.(-im * (tau) * H)
    return  U_op1

  elseif eltype(H) == Vector{ITensor}
    Us = []
    for Hi in H
      U_op2(tau) = exp.(-im * (tau) * Hi)
      push!(Us, U_op2)
    end
    return Tuple(Us)
  end
end  # Tuple{anonymous function{Vector{ITensor}}, anonymous function{Vector{ITensor}}}

function time_operators_even_odd(hj, sites)
    H_even, H_odd = even_odd_Hamiltonian(hj, sites)
    # TEBD gate layers for 2nd-order Suzuki–Trotter:
    U_even(tau) = exp.(-im*(tau) * H_even)
    U_odd(tau)  = exp.(-im*(tau) * H_odd )
    return U_even, U_odd  # Tuple{anonymous function{Vector{ITensor}}, anonymous function{Vector{ITensor}}}
end

PF1_gate(U_a, U_b, τ::Real) = vcat(U_a(τ),   U_b(τ))
PF2_gate(U_a, U_b, τ::Real) = vcat(U_a(τ/2), U_b(τ), U_a(τ/2))
PF3_gate(U_a, U_b, τ::Real) = vcat(
    U_a((7/24)*τ),  U_b((2/3)*τ),
    U_a((3/4)*τ),   U_b((-2/3)*τ),
    U_a((-1/24)*τ), U_b(τ),
)

function PF_gate(hj, sites, τ::Real; order::Int = 1)
    U_even, U_odd = time_operators_even_odd(hj, sites)
    
    if     order == 1
        return PF1_gate(U_even, U_odd, τ)
    elseif order == 2
        return PF2_gate(U_even, U_odd, τ)
    elseif order == 3
        return PF3_gate(U_even, U_odd, τ)
    else
        throw(ArgumentError("Order must be 1, 2, or 3."))
  end
end


function _bond_endpoint(r::Int, N::Int; per::Bool = false)
    if 1 <= r < N
        return r + 1
    elseif r == N && per
        return 1
    end
    throw(ArgumentError("bond index r=$r is outside the open chain 1:$(N - 1)"))
end


function discrete_odd_spin_current_density(r::Int, sites, J::Real; per::Bool = false)
    N = length(sites)
    rp = _bond_endpoint(r, N; per = per)
    sr, srp = sites[r], sites[rp]

    jr = (op("S-", sr) * op("S+", srp) - op("S+", sr) * op("S-", srp)) / (2im)
    dz = op("Id", sr) * op("Sz", srp) - op("Sz", sr) * op("Id", srp)

    return 2sin(J) * jr - 0.5sin(J / 2)^2 * dz
end

function discrete_odd_spin_current_density_mpo(r::Int, sites, J::Real; per::Bool = false)
    N = length(sites)
    rp = _bond_endpoint(r, N; per = per)

    os = OpSum()
    os += sin(J) / im, "S-", r, "S+", rp
    os += -sin(J) / im, "S+", r, "S-", rp
    os += 0.5sin(J / 2)^2, "Sz", r
    os += -0.5sin(J / 2)^2, "Sz", rp

    return MPO(os, sites)
end


function discrete_spin_current_density_operators(
    sites,
    J::Real,
    Ue_gates;
    bonds = nothing,
    per::Bool = false,
    cutoff = 1e-10,
    maxdim = nothing,
)
    N = length(sites)
    selected_bonds = bonds === nothing ? (per ? collect(1:N) : collect(1:(N - 1))) : collect(bonds)

    j_odd = [
        discrete_odd_spin_current_density_mpo(r, sites, J; per = per)
        for r in selected_bonds
    ]

    Ue_dag = swapprime.(dag.(Ue_gates),0,1)
    j_even = [
        apply(Ue_dag, jr; cutoff = cutoff, maxdim = maxdim, apply_dag = true)
        for jr in j_odd
    ]

    return (odd = j_odd, even = j_even, bonds = selected_bonds)
end


function step(psi::MPS, gate; cutoff=1e-10, maxdim=nothing, hermitianize::Bool = false, normalize::Bool = true)
    psi = apply(gate, psi; cutoff=cutoff, maxdim=maxdim)
    if normalize
        _normalize!(psi)
    end
    return psi
end

function step(rho::MPO, gate; cutoff=1e-10, maxdim=nothing, hermitianize::Bool = false, normalize::Bool = true)
    rho = apply(gate, rho; cutoff=cutoff, maxdim=maxdim, apply_dag=true)
    if hermitianize
        hermitianize!(rho, normalize = normalize)
    elseif normalize
        _normalize!(rho)
    end
    return rho
end


# one step: U_even(τ/2) → U_odd(τ) → U_even(τ/2)
function step_r2!(ψ, τ, U_even, U_odd; cutoff=1e-10, maxdim=nothing)
  ψ = apply(U_even(τ/2), ψ; cutoff=cutoff, maxdim=maxdim)
  ψ = apply(U_odd(τ),    ψ; cutoff=cutoff, maxdim=maxdim)
  ψ = apply(U_even(τ/2), ψ; cutoff=cutoff, maxdim=maxdim)
  _normalize!(ψ)
  return ψ
end

function step_r2!(ρ, τ, U_even, U_odd; cutoff=1e-10, maxdim=nothing)
  ρ = apply(U_even(τ/2), ρ; cutoff=cutoff, maxdim=maxdim, apply_dag=true)
  ρ = apply(U_odd(τ),    ρ; cutoff=cutoff, maxdim=maxdim, apply_dag=true)
  ρ = apply(U_even(τ/2), ρ; cutoff=cutoff, maxdim=maxdim, apply_dag=true)
  _normalize!(ρ)
  return ρ
end

# ========================================= #
# ============== Hamiltonians ============= #
# ========================================= #


function RLD_vec(W::Real, N::Int; seed::Int=123)
    Random.seed!(seed)
    return (2W) .* rand(N) .- W   # h[i] ∈ [-W, W]
end

function heis_hj(j::Int, s; J::Real = 1.0, Δ::Real = 1.0, per:: Bool = false) # 2-site Heisenberg term (j,j+1); S·S = SzSz + 1/2(S+S- + S-S+)
  @assert 1 ≤ j ≤ length(s) "heis_hj: j must be in 1:N"
  if j < length(s)
    return J*(Δ * op("Sz", s[j]) * op("Sz", s[j+1]) + 
      0.5  * op("S+", s[j]) * op("S-", s[j+1]) + 
      0.5  * op("S-", s[j]) * op("S+", s[j+1]) )
  end

  if per == true         # j+1 = 1
    return J*(Δ * op("Sz", s[j]) * op("Sz", s[1]) + 
      0.5  * op("S+", s[j]) * op("S-", s[1]) + 
      0.5  * op("S-", s[j]) * op("S+", s[1]) )
  end

  0 * op("Id", s[j])
end


function heis_rf(j::Int, s; h::AbstractVector = h_example, J::Real=1.0, Δ::Real=1.0, per:: Bool = false)
    N = length(s)
    @assert 1 ≤ j ≤ N "heis_rf_bond: j must be in 1:N"
    if j < N
        return heis_hj(j, s; J = J, Δ = Δ, per = per) + h[j] * (op("Sz", s[j]) * op("Id", s[j+1])) # add on-site terms
    end
    if per == true         # j+1 = 1
        return heis_hj(j, s; J = J, Δ = Δ, per = per) + h[j] * (op("Sz", s[j]) * op("Id", s[1]))
    end
    h[j] * op("Sz", s[j])
end

function heis_rf_for_h(h::AbstractVector; J::Real=1.0, Δ::Real=1.0, per::Bool=false)
    return (j::Int, s) -> heis_rf(j, s; h=h, J=J, Δ=Δ, per=per)
end

function heis_hj_no_h(; J::Real=1.0, Δ::Real=1.0, per::Bool=false)
    return (j::Int, s) -> heis_hj(j, s; J=J, Δ=Δ, per=per)
end


# ========================================= #
# ===== Differences for a changing r ====== #
# ========================================= #

### Distancia entre estados a diferentes r's

_distance(ρ::MPO, σ::MPO) = _mpo_distance_frobenius(ρ, σ)
_distance(ψ::MPS, ϕ::MPS) = _mps_distance(ψ, ϕ)
function states_distance(ψv, ϕv)
    n = size(ψv)[1]
    return [_distance(ψv[i], ϕv[i]) for i in 1:n]
end

function _expect(psi::MPS, opname::String = "Sz") 
    N = length(psi)
    
    return expect(psi, opname; sites = 1:N)
end
_expect(rho::MPO, opname::String = "Sz") = local_expect_mpo(rho, opname)

_Entropy_Bipartition(psi::MPS, b::Int; base = 2, tol = 1e-20) = Entropy_Bipartition(psi, b; base = base, tol = tol)
_Entropy_Bipartition(rho::MPO, b; base = 2)                   = second_Renyi_entropy(rho, b; base = base)

### Evolución estados

function state_evo(
    s,
    psi,
    t_Vec,
    r;
    h::Union{Nothing,AbstractVector}=nothing,
    per::Bool=false,
    cutoff::Real=1e-10,
    showprogress::Bool=true,
    hermitianize::Bool=false,
    store_states::Bool=true,
    normalize_every::Int=1,
)

    normalize_every < 1 && throw(ArgumentError("normalize_every must be >= 1"))

    tau =  (t_Vec[end] - t_Vec[1]) / r

    if h === nothing || all(isnothing, h)
        f_hamiltonian = heis_hj_no_h(; per = per)
    else
        f_hamiltonian = heis_rf_for_h(h; per = per) # each h[i] ∈ [-W, W] ts
    end
    
    gate = PF_gate(f_hamiltonian, s, tau; order = 1)

    nt = length(t_Vec)
    N = length(s)
    c = div(N, 2)

    psi_vec = store_states ? Vector{typeof(psi)}(undef, nt) : Vector{typeof(psi)}()
    Sz_all = Matrix{Float64}(undef, nt, N)
    S_Bi = Vector{Float64}(undef, nt)
    psi_t = deepcopy(psi)

    p = showprogress ? Progress(nt; desc="State evo (r=$r)", dt=0.2) : nothing

    for i in 1:nt
        if store_states
            psi_vec[i] = copy(psi_t)
        end

        Sz_all[i, :] = real.(_expect(psi_t, "Sz"))

        S = _Entropy_Bipartition(psi_t, c)
        if abs(imag(S)) > 1e-8 * abs(real(S))
            @warn "Entropy has a non-negligible imaginary part" S
        end
        S_Bi[i] = real(S)

        if i < nt
            do_normalize = (normalize_every == 1) || (i % normalize_every == 0) || (i == nt - 1)
            do_hermitianize = hermitianize && do_normalize
            psi_t = step(
                psi_t,
                gate;
                cutoff=cutoff,
                hermitianize=do_hermitianize,
                normalize=do_normalize,
            )
        end

        showprogress && next!(p)
    end

    return (psi_vec, Sz_all, S_Bi)
end


function complete_state_evo(
    s,
    psi,
    t_Vec,
    r;
    h::Union{Nothing,AbstractVector}=nothing,
    per::Bool=false,
    cutoff::Real=1e-10,
    showprogress::Bool=true,
    hermitianize::Bool=false,
    normalize_every::Int=1,
)

    normalize_every < 1 && throw(ArgumentError("normalize_every must be >= 1"))

    tau =  (t_Vec[end] - t_Vec[1]) / r

    if h === nothing || all(isnothing, h)
        f_hamiltonian = heis_hj_no_h(; per = per)
    else
        f_hamiltonian = heis_rf_for_h(h; per = per)
    end

    gate = PF_gate(f_hamiltonian, s, tau; order = 1)

    nt = length(t_Vec)
    psi_vec = Vector{typeof(psi)}(undef, nt)
    psi_t = deepcopy(psi)

    p = showprogress ? Progress(nt; desc="Full state evo (r=$r)", dt=0.2) : nothing

    for i in 1:nt
        psi_vec[i] = copy(psi_t)

        if i < nt
            do_normalize = (normalize_every == 1) || (i % normalize_every == 0) || (i == nt - 1)
            do_hermitianize = hermitianize && do_normalize
            psi_t = step(
                psi_t,
                gate;
                cutoff=cutoff,
                hermitianize=do_hermitianize,
                normalize=do_normalize,
            )
        end

        showprogress && next!(p)
    end

    return psi_vec
end


function Diff_trotter_r(
    s,
    psi,
    t,
    rv::Vector;
    h::Union{Nothing,AbstractVector}=nothing,
    per::Bool=false,
    cutoff::Real=1e-10,
    doprint::Bool=true,
    showprogress::Bool=false,
    hermitianize::Bool=false,
    store_states::Bool=true,
    normalize_every::Int=1,
)
    N = length(s) 

    TOut = Tuple{Vector{typeof(psi)}, Matrix{Float64}, Vector{Float64}}
    All_Full_evos = Vector{TOut}(undef, length(rv))

    for (ir, r) in enumerate(rv)
        tau = t/r
        ts = collect(range(0.0, t; length=r+1))

        if doprint == true 
            print_simulation_parameters(N, tau, cutoff, t, r)
        end
        All_Full_evos[ir] = state_evo(
            s,
            psi,
            ts,
            r;
            h=h,
            per=per,
            showprogress=showprogress,
            cutoff=cutoff,
            hermitianize=hermitianize,
            store_states=store_states,
            normalize_every=normalize_every
        )
    end
    return All_Full_evos
end


# ========================================= #
# =============== Plotting ================ #
# ========================================= #


function print_simulation_parameters(N, tau, cutoff, t, r)
    println("=============== Simulation Parameters ===============")
    @printf("   %-6s = %-12d  # Number of sites\n", "N", N)
    @printf("   %-6s = %-12d  # Number of Trotter steps\n", "r", r)
    @printf("   %-6s = %-12g  # Trotter step\n", "τ", tau)
    @printf("   %-6s = %-12g  # Total simulation time\n", "t", t)
    @printf("   %-6s = %-12.1e # Singular value cutoff\n", "cutoff", cutoff)
    println("=====================================================")
end

function plot_spin_dynamics(ts, Sz_all, S_Bi, n, N, tau; gifname = "spins_bars.gif", fps = 30)
    # Mantener tus defaults exactos
    default(size = (1200, 300), left_margin = 8Plots.mm, bottom_margin = 6Plots.mm)

    # --- Lógica de tiempo real ---
    # Calculamos cada cuántos pasos (stride) debemos tomar un frame
    # para que la velocidad coincida con los FPS elegidos.
    # Stride = 1 / (tau * fps)
    every = max(1, round(Int, 1 / (tau * fps)))

    sz_maximum = maximum(Sz_all)*1.15

    # --- Animación ---
    anim = @animate for i in 1:every:size(Sz_all, 1)
        bar(1:N, Sz_all[i, :];
            ylim = (-sz_maximum, sz_maximum),
            xlabel = "site j",
            ylabel = "⟨Sᶻ⟩",
            legend = false,
            title = "t = $(round(ts[i]; digits = 2))",
            bar_width = 1,
            linewidth = clamp(100 / N, 0.0, 1.0),
        )
    end
    
    # Generar el GIF con los FPS fijos (ej. 30)
    Sz_animation = gif(anim, gifname, fps = fps)

    # --- Gráficas estáticas (sin cambios de tamaño) ---
    Szc_plot = plot(ts, Sz_all[:, n];
        xlabel = "t",
        ylabel = "⟨Sᶻ_$n⟩",
        legend = false,
        title = "Site j = $n",
    )

    SvN_plot = plot(ts, S_Bi;
        xlabel = "t",
        ylabel = "Entanglement entropy",
        legend = false,
        title = "Center bipartition",
    )

    combined = plot(Szc_plot, SvN_plot; layout = (1, 2))

    return combined, Sz_animation
end


function plot_diff(All_evos, t; plot_diffs::Bool = true)
    L = length(All_evos)
    N = size(All_evos[1][2], 2)
    c = div(N, 2)
    has_states = all(ev -> !isempty(ev[1]), All_evos)

    tvv = [
        collect(range(0, t, length = size(All_evos[i][2], 1))) for i in 1:(L)
    ]

    idv = Vector{Vector{Int}}(undef, L)
    for i in 1:(L-1)
        id   = round.(Int, range(1, size(All_evos[L][2], 1),  length=size(All_evos[i][2], 1)))
        idv[i] = id
    end
    idv[L] = collect(1:size(All_evos[L][2], 1))

    MPSv_all = [evos[1] for evos in All_evos]
    Szv_all  = [evos[2] for evos in All_evos]
    Sc_all   = [evos[3] for evos in All_evos]

    dist_MPS = has_states ? Vector{Vector{Float64}}(undef, L-1) : Vector{Vector{Float64}}()
    diff_Szv = Vector{Matrix{Float64}}(undef, L-1)
    diff_Sc  = Vector{Vector{Float64}}(undef, L-1)

    for i in 1:(L-1)
        if has_states
            dist_MPS[i] = states_distance(MPSv_all[i], MPSv_all[L][idv[i]])
        end

        diff_Szv[i] = Szv_all[i] - Szv_all[L][idv[i], :]

        diff_Sc[i] = Sc_all[i] - Sc_all[L][idv[i]]
    end
    diff_Szv_c = [Szv[: , c] for Szv in diff_Szv]

    p = nothing
    if plot_diffs
        
        default(; lw=2, framestyle=:box, grid=true,
                legendfontsize=8, titlefontsize=11,
                guidefontsize=10, tickfontsize=8, 
                left_margin = 6Plots.mm, right_margin = 4Plots.mm, top_margin = 2Plots.mm, bottom_margin = 4Plots.mm)

        # Labels opcionales para la leyenda (solo en la distancia para no saturar)
        labels = ["evo $i" for i in 1:(L-1)]   # si tienes rv, puedes poner ["r=$(rv[i])" for i in 1:(L-1)]
        labels = ["r = $(length(diff_Sc[i])-1)" for i in 1:(L-1)]

        # Paneles individuales
        p1 = plot(; xlabel="t", ylabel="Δ⟨Sᶻ_$c⟩", title="Site j = $c",
                legend=false, xlims=(0, t))

        p2 = plot(; xlabel="t", ylabel="ΔS", title="Center bipartition",
                legend=false, xlims=(0, t))

        p3 = plot(; xlabel="t", ylabel="∥ψ−ϕ∥", title="States distance (2 - 2ℜ⟨ψ∣ϕ⟩)^1/2",
                legend=:topright, xlims=(0, t))

        # Curvas
        for i in 1:(L-1)
            plot!(p1, tvv[i], diff_Szv_c[i])
            plot!(p2, tvv[i], diff_Sc[i])
            if has_states
                plot!(p3, tvv[i], dist_MPS[i]; label=labels[i])
            end
        end

        if !has_states
            annotate!(p3, t / 2, 0.0, text("State distance unavailable (store_states=false)", 9))
        end

        # Layout: dos arriba, una abajo
        ptop = plot(p1, p2; layout=(1,2))
        p    = plot(ptop, p3; layout=(2,1), heights=[0.55, 0.45], size=(1200, 700))

    end
    return (dist_MPS, diff_Szv, diff_Sc, p)
end


function complete_plots(N, t, tau, evos)
    ts = collect(range(0, t, step = tau))
    c  = div(N,2)

    Sz_all = evos[end][2]
    S_Bi   = evos[end][3]

    combined, anim = plot_spin_dynamics(ts, Sz_all, S_Bi, c, N, tau, gifname = "Plots/spins_evo_magnon.gif")
    savefig(combined, "Plots/combined_magnon.png")

    display(combined)
    display(anim)

    if length(evos) ≠ 1
        d_plt = plot_diff(evos, t; plot_diffs = true)[end]
        savefig(d_plt, "Plots/differences_magnon.png")
        display(d_plt)
    end
end


# ========================================= #
# ============ Final functions ============ #
# ========================================= #


function wave_state(s, f; k = nothing, cutoff = 10e-10)
    N = length(s)
    psi0 = productMPS(s, fill("Up", N))
    if k === nothing
        k = -2*pi*div(N, 4)/N
    end

    # coeficiente complejo α_j = e^{ikj} f(j)
    α(j) = exp(1im * k * j) * f(j)

    # MPO del operador O = Σ α_j S-_j
    ampo = AutoMPO()
    for j in 1:N
        add!(ampo, α(j), "S-", j)
    end
    O = MPO(ampo, s)

    # aplicar y normalizar
    psi = apply(O, psi0; cutoff=cutoff)  #, maxdim
    psi /= sqrt(real(inner(psi, psi)))
    return psi
end


# ========================================= #
# ============== not in use =============== #
# ========================================= #

### Cálculo de parámetros de un vector de estados

function state_evo_parameters(s, MPS_vec; showprogress::Bool = true)
    N = length(s)
    l = length(MPS_vec)
    c = div(N, 2)

    Sz_all = Matrix{Float64}(undef, l, N)   # row i = at time ts[i], columns = sites 1..N
    S_Bi   = Vector{Float64}(undef, l)

    p = showprogress ? Progress(l; desc="Computing parameters", dt=0.2) : nothing

    for (i, psi_t) in enumerate(MPS_vec)
        Sz_all[i, :] = real.(_expect(psi_t, "Sz")) # local expectation value of Sz at every site → ⟨ψ∣Sz​∣ψ⟩
        #println(_Entropy_Bipartition(psi_t, c))
        S = _Entropy_Bipartition(psi_t, c)
        if abs(imag(S)) > 1e-8 * abs(real(S))
            @warn "Entropy has a non-negligible imaginary part" S
        end
        S_Bi[i] = real(S)      # von Neumann entropy of the bipartition
        showprogress && next!(p)
    end

    return (MPS_vec, Sz_all, S_Bi)
end



nothing

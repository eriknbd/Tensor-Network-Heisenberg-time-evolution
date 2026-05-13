using ITensors, ITensorMPS
using Random
using Printf
using Plots
using LaTeXStrings
using ProgressMeter
using HDF5

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
# ================= Gates ================= #
# ========================================= #


function even_odd_bonds(N::Int; per::Bool = false)
    last_bond = per ? N : N - 1
    return collect(2:2:last_bond), collect(1:2:last_bond)
end

function even_odd_Hamiltonian(hj, sites; per::Bool = false)
    N = length(sites)
    even_bonds, odd_bonds = even_odd_bonds(N; per = per)
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

function time_operators_even_odd(hj, sites; per::Bool = false)
    H_even, H_odd = even_odd_Hamiltonian(hj, sites; per = per)
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

function PF_gate(hj, sites, τ::Real; order::Int = 1, per::Bool = false)
    U_even, U_odd = time_operators_even_odd(hj, sites; per = per)
    
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
        zloc = 2 * cosh(μ / 2)
        for j in 1:L
            ρ[j] /= zloc
        end
    end
    isinf(tr(ρ)) && error("Dimensions are too large to compute the trace, tr(rho) -> inf. Use normalize = true.")

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
# ============== Spin current ============= #
# ========================================= #

function _bond_endpoint(r::Int, N::Int; per::Bool = false)
    if 1 <= r < N
        return r + 1
    elseif r == N && per
        return 1
    end
    throw(ArgumentError("bond index r=$r is outside the chain"))
end

function _bond_sites(r::Int, N::Int; per::Bool = false)
    if 1 <= r < N
        return (r, r + 1)
    elseif r == N && per
        return (N, 1)
    end
    return ()
end

function _overlaps_sites(a, b)
    for x in a, y in b
        x == y && return true
    end
    return false
end

function _is_contiguous_sites(site_numbers)
    isempty(site_numbers) && return false
    first_site, last_site = first(site_numbers), last(site_numbers)
    return site_numbers == collect(first_site:last_site)
end

function _mpo_to_itensor(O::MPO)
    T = ITensor(1.0)
    for j in 1:length(O)
        T *= O[j]
    end
    return T
end

"""
    discrete_odd_spin_current_density(r, sites, J; per=false)

Return the local discrete-time odd spin-current density operator on bond
`(r, r + 1)`,

`j_odd = 2sin(J) j_r - 0.5sin(J / 2)^2 (Sz_{r + 1} - Sz_r)`.
"""
function discrete_odd_spin_current_density(r::Int, sites, J::Real; per::Bool = false)
    N = length(sites)
    rp = _bond_endpoint(r, N; per = per)
    sr, srp = sites[r], sites[rp]

    jr = (op("S-", sr) * op("S+", srp) - op("S+", sr) * op("S-", srp)) / (2im)
    dz = op("Id", sr) * op("Sz", srp) - op("Sz", sr) * op("Id", srp)

    return 2sin(J) * jr - 0.5sin(J / 2)^2 * dz
end

function _discrete_odd_spin_current_density_mpo(r::Int, sites, J::Real; per::Bool = false)
    N = length(sites)
    rp = _bond_endpoint(r, N; per = per)

    os = OpSum()
    os += sin(J) / im, "S-", r, "S+", rp
    os += -sin(J) / im, "S+", r, "S-", rp
    os += 0.5sin(J / 2)^2, "Sz", r
    os += -0.5sin(J / 2)^2, "Sz", rp

    return MPO(os, sites)
end

"""
    discrete_spin_current_density_operators(sites, J, Ue_gates; bonds=nothing,
                                            per=false, cutoff=1e-10,
                                            maxdim=nothing)
    discrete_spin_current_density_operators(sites, J, tau; bonds=nothing,
                                            per=false, hj=nothing,
                                            cutoff=1e-10, maxdim=nothing)

Build local spin-current density operators for the discrete-time model.
`Ue_gates` should be the half-step even layer, for example `U_even(tau / 2)`.
The `tau` method builds that layer using `hj`, or the clean Heisenberg
Hamiltonian with coupling `J` when `hj === nothing`.

Returns a named tuple
`(odd, even, odd_sites, even_sites, bonds)`, where
`odd[i]` is `j_odd` on `bonds[i]` and `even[i]` is
`dag(U_e) * j_odd * U_e` restricted to the even gates that overlap that bond.
"""
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

    even_bonds, _ = even_odd_bonds(N; per = per)
    length(Ue_gates) == length(even_bonds) ||
        throw(ArgumentError("Ue_gates length must match the even bond layer"))

    Ue_dag = swapprime.(dag.(Ue_gates), 0, 1)
    even_gate_sites = [_bond_sites(r, N; per = per) for r in even_bonds]

    j_odd = Vector{ITensor}(undef, length(selected_bonds))
    j_even = Vector{ITensor}(undef, length(selected_bonds))
    odd_sites = Vector{Vector{Int}}(undef, length(selected_bonds))
    even_sites = Vector{Vector{Int}}(undef, length(selected_bonds))

    for (i, r) in enumerate(selected_bonds)
        current_sites = collect(_bond_sites(r, N; per = per))
        isempty(current_sites) && throw(ArgumentError("current bond r=$r is outside the chain"))

        j_odd[i] = discrete_odd_spin_current_density(r, sites, J; per = per)
        odd_sites[i] = current_sites

        local_gates = ITensor[]
        support_sites = copy(current_sites)

        for (gate_sites, gate) in zip(even_gate_sites, Ue_dag)
            if _overlaps_sites(current_sites, gate_sites)
                push!(local_gates, gate)
                append!(support_sites, gate_sites)
            end
        end

        sort!(unique!(support_sites))
        _is_contiguous_sites(support_sites) ||
            throw(ArgumentError("current support must be contiguous; periodic wraparound currents are not supported"))

        local_sites = sites[first(support_sites):last(support_sites)]
        local_r = r - first(support_sites) + 1
        local_odd = _discrete_odd_spin_current_density_mpo(local_r, local_sites, J; per = false)
        local_even = isempty(local_gates) ?
            local_odd :
            apply(local_gates, local_odd; cutoff = cutoff, maxdim = maxdim, apply_dag = true)

        j_even[i] = _mpo_to_itensor(local_even)
        even_sites[i] = support_sites
    end

    return (odd = j_odd, even = j_even, odd_sites = odd_sites, even_sites = even_sites, bonds = selected_bonds)
end

function discrete_spin_current_density_operators(
    sites,
    J::Real,
    tau::Real;
    bonds = nothing,
    per::Bool = false,
    hj = nothing,
    cutoff = 1e-10,
    maxdim = nothing,
)
    hamiltonian = hj === nothing ? heis_hj_no_h(; J = J, per = per) : hj
    U_even, _ = time_operators_even_odd(hamiltonian, sites; per = per)
    return discrete_spin_current_density_operators(
        sites,
        J,
        U_even(tau / 2);
        bonds = bonds,
        per = per,
        cutoff = cutoff,
        maxdim = maxdim,
    )
end

"""
    expect_local_operators_mpo(rho, ops, supports; normalize=true,
                               ket_primelev=0, bra_primelev=1)

Compute `Tr(rho * O)` for local ITensor operators using one set of trace
environments for the whole MPO. `supports[i]` must list the contiguous sites
where `ops[i]` acts.
"""
function expect_local_operators_mpo(
    rho::MPO,
    ops,
    supports;
    normalize::Bool = true,
    ket_primelev::Int = 0,
    bra_primelev::Int = 1,
)
    N = length(rho)
    length(ops) == length(supports) ||
        throw(DimensionMismatch("number of local operators and supports must match"))

    s = [siteind(rho, j; plev = ket_primelev) for j in 1:N]
    sp = [siteind(rho, j; plev = bra_primelev) for j in 1:N]

    any(isnothing, s) &&
        error("Could not find ket site indices with prime level=$ket_primelev")
    any(isnothing, sp) &&
        error("Could not find bra site indices with prime level=$bra_primelev")

    Ttr = Vector{ITensor}(undef, N)
    for j in 1:N
        Ttr[j] = rho[j] * delta(dag(s[j]), dag(sp[j]))
    end

    left_env = Vector{ITensor}(undef, N + 1)
    right_env = Vector{ITensor}(undef, N + 1)
    left_env[1] = ITensor(1.0)
    right_env[N + 1] = ITensor(1.0)

    for j in 1:N
        left_env[j + 1] = left_env[j] * Ttr[j]
    end
    for j in N:-1:1
        right_env[j] = Ttr[j] * right_env[j + 1]
    end

    Z = scalar(left_env[N + 1])
    invZ = normalize ? inv(Z) : one(Z)
    vals = Vector{ComplexF64}(undef, length(ops))

    for i in eachindex(ops)
        support = collect(supports[i])
        _is_contiguous_sites(support) ||
            throw(ArgumentError("local operator support must be contiguous"))

        first_site, last_site = first(support), last(support)
        1 <= first_site <= last_site <= N ||
            throw(ArgumentError("local operator support is outside the chain"))

        T = left_env[first_site]
        for j in first_site:last_site
            T *= rho[j]
        end
        vals[i] = scalar(T * dag(ops[i]) * right_env[last_site + 1]) * invZ
    end

    return vals
end

"""
    expected_spin_currents_mpo(rho, current_ops; kind=:both, normalize=true,
                               real_values=true)
    expected_spin_currents_mpo(rho, sites, J, tau_or_Ue_gates; kwargs...)

Efficiently compute the expected odd and/or even discrete-time spin currents
for an MPO. For repeated measurements, precompute `current_ops` with
`discrete_spin_current_density_operators` and pass it to the first method.

Returns `(bonds, odd, even)`, where unrequested current arrays are `nothing`.
"""
function expected_spin_currents_mpo(
    rho::MPO,
    current_ops;
    kind::Symbol = :both,
    normalize::Bool = true,
    real_values::Bool = true,
)
    kind in (:odd, :even, :both) ||
        throw(ArgumentError("kind must be :odd, :even, or :both"))

    odd = nothing
    even = nothing

    if kind in (:odd, :both)
        odd = expect_local_operators_mpo(
            rho,
            current_ops.odd,
            current_ops.odd_sites;
            normalize = normalize,
        )
        odd = real_values ? real.(odd) : odd
    end

    if kind in (:even, :both)
        even = expect_local_operators_mpo(
            rho,
            current_ops.even,
            current_ops.even_sites;
            normalize = normalize,
        )
        even = real_values ? real.(even) : even
    end

    return (bonds = current_ops.bonds, odd = odd, even = even)
end

function expected_spin_currents_mpo(
    rho::MPO,
    sites,
    J::Real,
    tau_or_Ue_gates;
    bonds = nothing,
    per::Bool = false,
    hj = nothing,
    cutoff = 1e-10,
    maxdim = nothing,
    kind::Symbol = :both,
    normalize::Bool = true,
    real_values::Bool = true,
)
    current_ops = if tau_or_Ue_gates isa Real
        discrete_spin_current_density_operators(
            sites,
            J,
            tau_or_Ue_gates;
            bonds = bonds,
            per = per,
            hj = hj,
            cutoff = cutoff,
            maxdim = maxdim,
        )
    else
        discrete_spin_current_density_operators(
            sites,
            J,
            tau_or_Ue_gates;
            bonds = bonds,
            per = per,
            cutoff = cutoff,
            maxdim = maxdim,
        )
    end

    return expected_spin_currents_mpo(
        rho,
        current_ops;
        kind = kind,
        normalize = normalize,
        real_values = real_values,
    )
end


# ========================================= #
# ======= Complete stable evolution ======= #
# ========================================= #


function hermitianize_stable(rho::MPO; cutoff=1e-10, normalize::Bool = false)
    rhoH = hermitian_mpo(rho)
    rho_hermitian = 0.5 * add(rho, rhoH; cutoff=cutoff)
    if normalize
        normalize!(rho_hermitian)
    end
    return rho_hermitian
end

function hermitianize_stable!(rho::MPO; cutoff=1e-10, normalize::Bool = false)
    rho .= hermitianize_stable(rho; cutoff=cutoff, normalize = normalize)
    return rho
end

function step_stable(rho::MPO, gate; cutoff=1e-10, maxdim=nothing, hermitianize::Bool = false, normalize::Bool = true)
    rho = apply(gate, rho; cutoff=cutoff, maxdim=maxdim, apply_dag=true)
    if hermitianize
        hermitianize_stable!(rho; cutoff = cutoff, normalize = normalize)
    elseif normalize
        normalize!(rho)
    end
    return rho
end

function complete_state_evo(
    s,
    psi,
    t_Vec,
    r;
    h::Union{Nothing,AbstractVector}=nothing,
    per::Bool=false,
    cutoff = 1e-10,
    maxdim = nothing,
    showprogress::Bool=true,
    hermitianize::Bool=false,
    normalize_every::Int=1,
)

    normalize_every < 1 && throw(ArgumentError("normalize_every must be >= 1"))

    tau =  (t_Vec[end] - t_Vec[1]) / r

    if h === nothing || all(isnothing, h)
        f_hamiltonian = heis_hj_no_h(; per = per, J=pi/2)
    else
        f_hamiltonian = heis_rf_for_h(h; per = per)
    end

    gate = PF_gate(f_hamiltonian, s, tau; order = 1, per = per)

    nt = length(t_Vec)
    psi_vec = Vector{typeof(psi)}(undef, nt)
    psi_t = deepcopy(psi)
    normalize!(psi_t)

    p = showprogress ? Progress(nt; desc="Full state evo (r=$r)", dt=0.2) : nothing

    for i in 1:nt
        psi_vec[i] = copy(psi_t)

        if i < nt
            do_normalize = (normalize_every == 1) || (i % normalize_every == 0) || (i == nt - 1)
            do_hermitianize = hermitianize && do_normalize
            psi_t = step_stable(
                psi_t,
                gate;
                cutoff=cutoff,
                maxdim = maxdim,
                hermitianize=do_hermitianize,
                normalize=do_normalize,
            )
        end

        showprogress && next!(p)
    end

    return psi_vec
end

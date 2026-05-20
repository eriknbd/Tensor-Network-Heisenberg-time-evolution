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

function _mpo_trace_tensor(rho::MPO, j::Int; ket_primelev::Int = 0, bra_primelev::Int = 1)
    s = siteind(rho, j; plev = ket_primelev)
    sp = siteind(rho, j; plev = bra_primelev)

    isnothing(s) && error("Could not find ket site index at site $j with prime level=$ket_primelev")
    isnothing(sp) && error("Could not find bra site index at site $j with prime level=$bra_primelev")

    return rho[j] * delta(dag(s), dag(sp))
end

function _rescale_trace_env(T::ITensor)
    nrm = norm(T)
    if iszero(nrm) || !isfinite(nrm)
        return T
    end
    return T / nrm
end

# ⟨Oj​⟩ = Tr(ρOj​)/Tr(ρ)​
function local_expect_mpo(ρ::MPO, opname::String = "Sz"; normalize::Bool = true, ket_primelev::Int = 0, bra_primelev::Int = 1)
    N = length(ρ)
    s = [siteind(ρ, j; plev = ket_primelev) for j in 1:N]

    any(isnothing, s) && error("Could not find ket site indices with prime level=$ket_primelev")

    ops = [op(opname, s[j]) for j in 1:N]
    left, right, Z = _expect_env_helplocal_expect_mpo(
        ρ;
        ket_primelev = ket_primelev,
        bra_primelev = bra_primelev,
    )

    return local_expect_mpo(
        ρ,
        ops,
        left,
        right,
        Z;
        normalize = normalize,
        ket_primelev = ket_primelev,
        bra_primelev = bra_primelev,
    )
end

function _normalize!(ρ::MPO)
    Z = tr(ρ)
    iszero(Z) && error("No se puede normalizar un MPO con traza cero.")

    if !(isfinite(real(Z)) && isfinite(imag(Z)))
        error("La traza no es finita: tr(ρ) = $Z")
    end

    N = length(ρ)
    absZ = abs(Z)
    phase = Z / absZ

    # Distribute absZ over all tensors
    λ = exp(log(absZ) / N)

    for j in 1:N
        ρ[j] /= λ
    end

    ρ[1] /= phase

    return ρ
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

function _bond_endpoint(r::Int, N::Int; per::Bool=false)
    if 1 <= r < N
        return r + 1
    elseif r == N && per
        return 1
    end

    throw(ArgumentError("Bond $r is outside the $(per ? "periodic" : "open") chain with N=$N sites."))
end

function continuous_spin_current_density_mpo(r::Int, sites; per::Bool=false)
    N = length(sites)
    rp = _bond_endpoint(r, N; per=per)

    os = OpSum()

    os +=  1 / (2im), "S-", r, "S+", rp
    os += -1 / (2im), "S+", r, "S-", rp

    return MPO(os, sites)
end

function continuous_spin_current_density_mpos(
    sites;
    bonds=nothing,
    per::Bool=false,
)
    N = length(sites)

    selected_bonds = bonds === nothing ?
        (per ? collect(1:N) : collect(1:N-1)) :
        collect(bonds)

    currents = Vector{MPO}(undef, length(selected_bonds))

    for (i, r) in enumerate(selected_bonds)
        currents[i] = continuous_spin_current_density_mpo(r, sites; per=per)
    end

    return (
        ops = currents,
        bonds = selected_bonds,
    )
end

function local_current_opsum(b::Int, bp::Int; A, B)
    os = OpSum()

    # A * j_b
    os += A / (2im),  "S-", b,  "S+", bp
    os += -A / (2im), "S+", b,  "S-", bp

    # B * (Sz_bp - Sz_b)
    os += B,  "Sz", bp
    os += -B, "Sz", b

    return os
end

function discrete_time_spin_current_mpos(
    sites;
    J::Real,
    tau::Real = 1.0,
    per::Bool = false,
)
    N = length(sites)
    last_bond = per ? N : N - 1

    θ = J * tau

    # Odd-current coefficients:
    #
    # jᵒ_b = Aₒ j_b + Bₒ (Sz_{b+1} - Sz_b)
    A_odd = 2 * sin(θ)
    B_odd = -0.5 * sin(θ / 2)^2

    # Even current:
    #
    # jᵉ_b = U(τ/2)† jᵒ_b U(τ/2)
    #
    # Under the two-site Heisenberg gate with angle φ,
    #
    # j_b -> j_b cos(φ) - 1/2 (Sz_{b+1} - Sz_b) sin(φ)
    # d_b -> d_b cos(φ) + 2 j_b sin(φ),
    #
    # where d_b = Sz_{b+1} - Sz_b.
    φ = θ / 2

    A_even = A_odd * cos(φ) + 2 * B_odd * sin(φ)
    B_even = B_odd * cos(φ) - 0.5 * A_odd * sin(φ)

    currents = MPO[]
    supports = Tuple{Int,Int}[]
    bonds = Int[]
    kind = Symbol[]

    for b in 1:last_bond
        bp = b == N ? 1 : b + 1

        if isodd(b)
            os = local_current_opsum(b, bp; A = A_odd, B = B_odd)
            push!(kind, :odd)
        else
            os = local_current_opsum(b, bp; A = A_even, B = B_even)
            push!(kind, :even)
        end

        push!(currents, MPO(os, sites))
        push!(supports, (b, bp))
        push!(bonds, b)
    end

    coeffs = (
        theta = θ,
        A_odd = A_odd,
        B_odd = B_odd,
        A_even = A_even,
        B_even = B_even,
    )

    return (
        currents = currents,
        supports = supports,
        bonds = bonds,
        kind = kind,
        coeffs = coeffs,
    )
end


#return local ITensor
function local_current_itensor(sites, b::Int, bp::Int; A, B)
    Id_b  = op("Id", sites[b])
    Id_bp = op("Id", sites[bp])

    # j_b = 1/(2i) (S-_b S+_bp - S+_b S-_bp)
    O =
        A / (2im)  * op("S-", sites[b]) * op("S+", sites[bp]) -
        A / (2im)  * op("S+", sites[b]) * op("S-", sites[bp]) +
        B          * Id_b               * op("Sz", sites[bp]) -
        B          * op("Sz", sites[b]) * Id_bp

    return O
end

function discrete_time_spin_current_local_itensors(
    sites;
    J::Real,
    tau::Real = 1.0,
    per::Bool = false,
)
    N = length(sites)
    last_bond = per ? N : N - 1

    θ = J * tau

    # Odd-current coefficients:
    #
    # jᵒ_b = Aₒ j_b + Bₒ (Sz_{b+1} - Sz_b)
    A_odd = 2 * sin(θ)
    B_odd = -0.5 * sin(θ / 2)^2

    # Even current:
    #
    # jᵉ_b = U(τ/2)† jᵒ_b U(τ/2)
    #
    # Under the two-site Heisenberg gate with angle φ = θ/2:
    #
    # j_b -> j_b cos(φ) - 1/2 (Sz_{b+1} - Sz_b) sin(φ)
    # d_b -> d_b cos(φ) + 2 j_b sin(φ),
    #
    # where d_b = Sz_{b+1} - Sz_b.
    φ = θ / 2

    A_even = A_odd * cos(φ) + 2 * B_odd * sin(φ)
    B_even = B_odd * cos(φ) - 0.5 * A_odd * sin(φ)

    currents = ITensor[]
    supports = Tuple{Int,Int}[]
    bonds = Int[]
    kind = Symbol[]

    for b in 1:last_bond
        bp = b == N ? 1 : b + 1

        if isodd(b)
            O = local_current_itensor(sites, b, bp; A = A_odd, B = B_odd)
            push!(kind, :odd)
        else
            O = local_current_itensor(sites, b, bp; A = A_even, B = B_even)
            push!(kind, :even)
        end

        push!(currents, O)
        push!(supports, (b, bp))
        push!(bonds, b)
    end

    coeffs = (
        theta = θ,
        A_odd = A_odd,
        B_odd = B_odd,
        A_even = A_even,
        B_even = B_even,
    )

    return (
        currents = currents,
        supports = supports,
        bonds = bonds,
        kind = kind,
        coeffs = coeffs,
    )
end

# ========================================= #
# ============ expected values ============ #
# ========================================= #

function _expectation_steps(nt::Int, expect_every::Int)
    expect_every < 1 && throw(ArgumentError("expect_every must be >= 1"))
    steps = collect(1:expect_every:nt)
    last(steps) == nt || push!(steps, nt)
    return steps
end

function _expect_env_helplocal_expect_mpo(ρ::MPO;
                                 ket_primelev::Int = 0,
                                 bra_primelev::Int = 1,
                                 rescale::Bool = true)

    N = length(ρ)

    Ttr = [_mpo_trace_tensor(ρ, j; ket_primelev = ket_primelev, bra_primelev = bra_primelev) for j in 1:N]

    R = Vector{ITensor}(undef, N + 1)
    R[N + 1] = ITensor(1.0)
    for j in N:-1:1
        R[j] = Ttr[j] * R[j + 1]
        if rescale
            R[j] = _rescale_trace_env(R[j])
        end
    end

    L = Vector{ITensor}(undef, N + 1)
    L[1] = ITensor(1.0)
    for j in 1:N
        L[j + 1] = L[j] * Ttr[j]
        if rescale
            L[j + 1] = _rescale_trace_env(L[j + 1])
        end
    end

    # When rescale=true, this is the scaled full trace, not the raw Tr(ρ).
    # local_expect_mpo normalizes with local denominators so the scale cancels.
    return L, R, scalar(L[N + 1])
end  

# ⟨Oj​⟩ = Tr(ρOj​)/Tr(ρ)​

function mpo_block(rho::MPO, i::Int, f::Int)
    @assert i <= f "mpo_block assumes a contiguous support with i <= f"

    B = rho[i]
    for j in (i + 1):f
        B *= rho[j]
    end

    return B
end

function mpo_trace_block(rho::MPO, i::Int, f::Int; ket_primelev::Int = 0, bra_primelev::Int = 1)
    @assert i <= f "mpo_trace_block assumes a contiguous support with i <= f"

    B = _mpo_trace_tensor(rho, i; ket_primelev = ket_primelev, bra_primelev = bra_primelev)
    for j in (i + 1):f
        B *= _mpo_trace_tensor(rho, j; ket_primelev = ket_primelev, bra_primelev = bra_primelev)
    end

    return B
end

function local_expect_mpo(
    rho,
    Ovec::Vector{ITensor},
    left::Vector,
    right::Vector,
    Z;
    supports = nothing,
    normalize::Bool = true,
    ket_primelev::Int = 0,
    bra_primelev::Int = 1,
)

    N = length(rho)
    no = length(Ovec)
    vals = Vector{ComplexF64}(undef, supports === nothing ? no : length(supports))

    if supports === nothing
        no == N || throw(DimensionMismatch("supports are required unless there is one operator per site"))

        for j in 1:no
            Oj = Ovec[j]
            numerator = scalar(left[j] * rho[j] * Oj * right[j + 1])
            if normalize
                denominator = scalar(
                    left[j] *
                    _mpo_trace_tensor(rho, j; ket_primelev = ket_primelev, bra_primelev = bra_primelev) *
                    right[j + 1]
                )
                vals[j] = numerator / denominator
            else
                vals[j] = numerator
            end
        end

        return vals
    end

    length(Ovec) == length(supports) ||
        throw(DimensionMismatch("number of local operators and supports must match"))

    for j in eachindex(supports)
        support = collect(supports[j])
        isempty(support) && throw(ArgumentError("local operator support must not be empty"))

        sitei = first(support)
        sitef = last(support)

        1 <= sitei <= sitef <= N ||
            throw(ArgumentError("local operator support must be an in-bounds contiguous block"))
        support == collect(sitei:sitef) ||
            throw(ArgumentError("local operator support must be contiguous"))

        rho_loc = mpo_block(rho, sitei, sitef)
        trace_loc = mpo_trace_block(
            rho,
            sitei,
            sitef;
            ket_primelev = ket_primelev,
            bra_primelev = bra_primelev,
        )

        Oj = Ovec[j]
        numerator = scalar(left[sitei] * rho_loc * dag(Oj) * right[sitef + 1])
        if normalize
            denominator = scalar(left[sitei] * trace_loc * right[sitef + 1])
            vals[j] = numerator / denominator
        else
            vals[j] = numerator
        end
    end

    return vals
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

function _mpo_bond_dims(rho::MPO)
    N = length(rho)
    dims = Vector{Int}(undef, max(N - 1, 0))
    for j in 1:(N - 1)
        link = commonind(rho[j], rho[j + 1])
        dims[j] = isnothing(link) ? 1 : dim(link)
    end
    return dims
end

function complete_expectation_evo(
    s, rho0::MPO, t_Vec, r;
    h::Union{Nothing,AbstractVector}=nothing, J::Real = 1, per::Bool=false,
    cutoff = 1e-10, maxdim = nothing, 
    showprogress::Bool=true,
    hermitianize::Bool=false, 
    normalize_every::Int=1, 
    avg_every::Int=1,
    current_operators = nothing, 
    current_bonds = nothing,
    current_kind::Symbol = :both,
    normalize_expectations::Bool = true,
    real_values::Bool = true,
)
    normalize_every < 1 && throw(ArgumentError("normalize_every must be >= 1"))
    avg_every < 1 && throw(ArgumentError("avg_every must be >= 1"))
    current_kind in (:odd, :even, :both) ||
        throw(ArgumentError("current_kind must be :odd, :even, or :both"))

    tau = (t_Vec[end] - t_Vec[1]) / r

    f_hamiltonian = if h === nothing || all(isnothing, h)
        heis_hj_no_h(; per = per, J = J)
    else
        heis_rf_for_h(h; per = per, J = J)
    end

    gate = PF_gate(f_hamiltonian, s, tau; order = 1, per = per)

    current_ops = if current_operators === nothing
        discrete_time_spin_current_local_itensors(s; J = J, tau = tau, per = per)
    else
        current_operators
    end

    current_ops.currents isa Vector{ITensor} ||
        throw(ArgumentError("current_operators.currents must be local ITensors; use discrete_time_spin_current_local_itensors"))

    current_indices = collect(eachindex(current_ops.currents))
    if current_bonds !== nothing
        selected_bonds = Set(collect(current_bonds))
        current_indices = [i for i in current_indices if current_ops.bonds[i] in selected_bonds]
    end
    if current_kind != :both
        current_indices = [i for i in current_indices if current_ops.kind[i] == current_kind]
    end

    current_tensors = current_ops.currents[current_indices]
    current_supports = current_ops.supports[current_indices]
    current_selected_bonds = current_ops.bonds[current_indices]
    current_selected_kind = current_ops.kind[current_indices]

    nt = length(t_Vec)
    N = length(s)
    avg_steps = _expectation_steps(nt, avg_every)
    avg_ts = collect(t_Vec[avg_steps])
    navg = length(avg_steps)

    value_type = real_values ? Float64 : ComplexF64
    Sz_all = Matrix{value_type}(undef, navg, N)
    J_all = Matrix{value_type}(undef, navg, length(current_tensors))
    bond_dims_all = Matrix{Int}(undef, navg, max(N - 1, 0))

    rho_t = deepcopy(rho0)
    normalize!(rho_t)
    Sz_ops = [op("Sz", siteind(rho_t, j; plev = 0)) for j in 1:N]

    p = showprogress ? Progress(nt; desc="Expectation evo (r=$r)", dt=0.2) : nothing
    avg_count = 0

    for i in 1:nt
        do_avg = avg_count < navg && i == avg_steps[avg_count + 1]
        if do_avg
            avg_count += 1
            bond_dims_all[avg_count, :] = _mpo_bond_dims(rho_t)
            left_env, right_env, Z = _expect_env_helplocal_expect_mpo(rho_t)

            sz_vals = local_expect_mpo(
                rho_t,
                Sz_ops,
                left_env,
                right_env,
                Z;
                normalize = normalize_expectations,
            )
            Sz_all[avg_count, :] = real_values ? real.(sz_vals) : sz_vals

            current_vals = local_expect_mpo(
                rho_t,
                current_tensors,
                left_env,
                right_env,
                Z;
                supports = current_supports,
                normalize = normalize_expectations,
            )
            J_all[avg_count, :] = real_values ? real.(current_vals) : current_vals
        end

        if i < nt
            do_normalize = (normalize_every == 1) || (i % normalize_every == 0) || (i == nt - 1)
            do_hermitianize = hermitianize && do_normalize
            rho_t = step_stable(
                rho_t,
                gate;
                cutoff = cutoff,
                maxdim = maxdim,
                hermitianize = do_hermitianize,
                normalize = do_normalize,
            )
        end

        showprogress && next!(p)
    end

    currents = (
        values = J_all,
        bonds = current_selected_bonds,
        kind = current_selected_kind,
        supports = current_supports,
        coeffs = hasproperty(current_ops, :coeffs) ? current_ops.coeffs : nothing,
        bond_dims = bond_dims_all,
    )

    return Sz_all, avg_ts, currents
end

const PKZ_EVOLUTION_DIR = joinpath(pwd(), "evolution_files")

function _pkz_evolution_filepath(filename)
    return isabspath(filename) ? filename : joinpath(PKZ_EVOLUTION_DIR, filename)
end

function _write_pkz_hdf5_params(parent, params, reserved)
    for (key, val) in params
        key = string(key)
        if key in reserved
            @warn "Skipping reserved HDF5 metadata key" key
        else
            write(parent, key, val)
        end
    end
end

function _current_kind_codes(kind)
    return Int8[k == :odd ? 1 : k == :even ? 2 : 0 for k in kind]
end

function _current_kind_from_codes(codes)
    return Symbol[c == 1 ? :odd : c == 2 ? :even : :unknown for c in codes]
end

function _supports_to_matrix(supports)
    M = Matrix{Int}(undef, length(supports), 2)
    for (i, support) in enumerate(supports)
        M[i, 1] = first(support)
        M[i, 2] = last(support)
    end
    return M
end

function _supports_from_matrix(M)
    return [(M[i, 1], M[i, 2]) for i in axes(M, 1)]
end

function _write_current_coeffs(parent, coeffs)
    coeffs === nothing && return

    g = create_group(parent, "coeffs")
    for name in propertynames(coeffs)
        write(g, string(name), getproperty(coeffs, name))
    end
end

function _read_current_coeffs(parent)
    haskey(parent, "coeffs") || return nothing

    g = parent["coeffs"]
    names = Tuple(Symbol.(collect(keys(g))))
    values = Tuple(read(g, string(name)) for name in names)
    return NamedTuple{names}(values)
end

function save_complete_expectation_evo(filename, result::Tuple; params = Dict())
    length(result) == 3 ||
        throw(ArgumentError("result must be the 3-value tuple returned by complete_expectation_evo"))

    Sz_all, avg_ts, currents = result
    return save_complete_expectation_evo(
        filename,
        Sz_all,
        avg_ts;
        currents = currents,
        params = params,
    )
end

function save_complete_expectation_evo(
    filename,
    Sz_all,
    avg_ts;
    currents,
    params = Dict(),
)
    filepath = _pkz_evolution_filepath(filename)
    mkpath(dirname(filepath))

    navg, N = size(Sz_all)
    length(avg_ts) == navg ||
        throw(ArgumentError("length(avg_ts) must match size(Sz_all, 1)"))

    nbonds = length(currents.bonds)
    size(currents.values, 1) == navg ||
        throw(ArgumentError("size(currents.values, 1) must match length(avg_ts)"))
    size(currents.values, 2) == nbonds ||
        throw(ArgumentError("size(currents.values, 2) must match length(currents.bonds)"))
    length(currents.kind) == nbonds ||
        throw(ArgumentError("length(currents.kind) must match length(currents.bonds)"))
    length(currents.supports) == nbonds ||
        throw(ArgumentError("length(currents.supports) must match length(currents.bonds)"))

    has_bond_dims = hasproperty(currents, :bond_dims) && currents.bond_dims !== nothing
    if has_bond_dims
        size(currents.bond_dims, 1) == navg ||
            throw(ArgumentError("size(currents.bond_dims, 1) must match length(avg_ts)"))
        size(currents.bond_dims, 2) == max(N - 1, 0) ||
            throw(ArgumentError("size(currents.bond_dims, 2) must match N - 1"))
    end

    h5open(filepath, "w") do f
        write(f, "format", "complete_expectation_evo_v1")
        write(f, "Sz_all", Sz_all)
        write(f, "avg_ts", avg_ts)
        write(f, "navg", navg)
        write(f, "N", N)
        write(f, "has_bond_dims", has_bond_dims)
        if has_bond_dims
            write(f, "bond_dims", currents.bond_dims)
        end

        reserved = Set(["format", "Sz_all", "avg_ts", "navg", "N", "has_bond_dims", "bond_dims", "currents"])
        _write_pkz_hdf5_params(f, params, reserved)

        g = create_group(f, "currents")
        write(g, "values", currents.values)
        write(g, "bonds", currents.bonds)
        write(g, "kind_codes", _current_kind_codes(currents.kind))
        write(g, "supports", _supports_to_matrix(currents.supports))
        _write_current_coeffs(g, hasproperty(currents, :coeffs) ? currents.coeffs : nothing)
    end

    println("Saved complete_expectation_evo values to: $filepath")
    return filepath
end

function load_complete_expectation_evo(filename)
    filepath = _pkz_evolution_filepath(filename)

    h5open(filepath, "r") do f
        Sz_all = read(f, "Sz_all")
        avg_ts = read(f, "avg_ts")

        bond_dims = haskey(f, "bond_dims") ? read(f, "bond_dims") : nothing

        g = f["currents"]
        currents = (
            values = read(g, "values"),
            bonds = vec(read(g, "bonds")),
            kind = _current_kind_from_codes(vec(read(g, "kind_codes"))),
            supports = _supports_from_matrix(read(g, "supports")),
            coeffs = _read_current_coeffs(g),
            bond_dims = bond_dims,
        )

        return Sz_all, avg_ts, currents
    end
end

function print_simulation_parameters(N, cutoff, t, r)
    println("=============== Simulation Parameters ===============")
    @printf("   %-6s = %-12d  # Number of sites\n", "N", N)
    @printf("   %-6s = %-12d  # Number of Trotter steps\n", "r", r)
    @printf("   %-6s = %-12g  # Trotter step\n", "τ", t/r)
    @printf("   %-6s = %-12g  # Total simulation time\n", "t", t)
    @printf("   %-6s = %-12.1e # Singular value cutoff\n", "cutoff", cutoff)
    println("=====================================================")
end

function print_simulation_params(filename)
    filepath = _pkz_evolution_filepath(filename)

    skip_keys = Set([
        "Sz_all",
        "avg_ts",
        "currents",
    ])

    h5open(filepath, "r") do f
        println("Simulation file: $filepath")
        println()

        println("Saved parameters (metadata):")

        for key in sort(collect(keys(f)))
            key in skip_keys && continue

            try
                val = read(f, key)

                if val isa AbstractArray
                    if length(val) <= 20
                        println("  $key = $val")
                    else
                        println("  $key = $(summary(val))")
                    end
                else
                    println("  $key = $val")
                end

            catch
                # This happens for groups or unsupported objects
                println("  $key = <group or unreadable object>")
            end
        end
    end

    return nothing
end

function plot_bond_dims_heatmap(bond_dims, ts)
    Nt, Nb = size(bond_dims)
    length(ts) == Nt || throw(DimensionMismatch("length(ts) must match size(bond_dims, 1)"))

    heatmap(
        1:Nb,
        ts,
        bond_dims;
        xlabel = "bond b",
        ylabel = "time t",
        title = "Bond dimension evolution",
        colorbar_title = "χ",
        size = (1200, 600)
    )
end
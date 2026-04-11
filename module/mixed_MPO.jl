using ITensors, ITensorMPS
using Random
using Printf
using Plots
using LaTeXStrings
using ProgressMeter


# ========================================= #
#             Initial State                 #
# ========================================= #


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


# ========================================= #
#             Hamiltonians                  #
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
#               Make Gate                   #
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


function PF_gate(hj, sites, τ::Real; order::Int = 2)
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

function step!(ψ, gate; cutoff=1e-8, maxdim=nothing)
  ψ = apply(gate, ψ; cutoff=cutoff, maxdim=maxdim)
  normalize!(ψ)
  return ψ
end


# ========================================= #
#                  MPO                      #
# ========================================= #


# MPO step!
function step!(ρ::MPO, gate; cutoff=1e-8, maxdim=nothing)
  ρ = apply(gate, ρ; cutoff=cutoff, maxdim=maxdim, apply_dag=true)
  return ρ
end

# trace and local expectation values for MPO
trace_mpo(ρ::MPO, IdMPO::MPO) = real(inner(IdMPO, ρ))

function fast_trace_mpo(ρ::MPO)
    N = length(ρ)
    s  = [siteind(ρ, j; plev=0) for j in 1:N]
    sp = [siteind(ρ, j; plev=1) for j in 1:N]

    L = ITensor(1.0)
    for j in 1:N
        L *= (ρ[j] * delta(sp[j], s[j]))
    end
    return real(scalar(L))
end

function local_expect_mpo(ρ::MPO, opname::String="Sz")
    N = length(ρ)
    s  = [siteind(ρ, j; plev=0) for j in 1:N]
    sp = [siteind(ρ, j; plev=1) for j in 1:N]

    tr_tensors = [ρ[j] * delta(sp[j], s[j]) for j in 1:N]

    Lenv = Vector{ITensor}(undef, N + 1)
    Lenv[1] = ITensor(1.0)
    for j in 1:N
        Lenv[j+1] = Lenv[j] * tr_tensors[j]
    end
    trρ = real(scalar(Lenv[N+1]))

    Renv = Vector{ITensor}(undef, N + 1)
    Renv[N+1] = ITensor(1.0)
    for j in N:-1:1
        Renv[j] = tr_tensors[j] * Renv[j+1]
    end

    vals = zeros(Float64, N)
    for j in 1:N
        Oj = op(opname, s[j])
        vals[j] = real(scalar(Lenv[j] * ρ[j] * Oj * Renv[j+1])) / trρ
    end

    return vals
end


# ========================================= #
#             Do evolution                  #
# ========================================= #


# entropy overload for MPO
function Entropy_Bipartition(ρ::MPO, b)
  ρ̃ = deepcopy(ρ)
  normalize!(ρ̃)
  ρ̃ = orthogonalize(ρ̃, b)
  U,S,V = svd(ρ̃[b], (linkinds(ρ̃, b-1)..., siteinds(ρ̃, b)...))
  SvN = 0.0
  for n=1:dim(S, 1)
    p = abs2(S[n,n])
    if p > 0
      SvN -= p * log(p)
    end
  end
  return SvN
end

function Entropy_Bipartition(psi::MPS, b)
  psi = orthogonalize(psi, b)
  U,S,V = svd(psi[b], (linkinds(psi, b-1)..., siteinds(psi, b)...))
  SvN = 0.0
  for n=1:dim(S, 1)
    p = abs2(S[n,n])
    if p > 0
      SvN -= p * log(p)
    end
  end
  return SvN
end

function _mps_distance(A::Union{MPS,MPO}, B::Union{MPS,MPO})
    # d = sqrt( ⟨ψ∣ψ⟩ + ⟨ϕ∣ϕ⟩ - 2ℜ⟨ψ∣ϕ⟩ )
    d2 = real(inner(A, A) + inner(B, B) - 2 * inner(A, B))
    return sqrt(max(real(d2), 0.0))
end

function mps_distance(ψv, ϕv)
    n = length(ψv)
    return [_mps_distance(ψv[i], ϕv[i]) for i in 1:n]
end

function MPS_evo(s, psi, t_Vec, r; h::Union{Nothing, AbstractVector}=nothing, per::Bool=false,
                 cutoff::Real=1e-10, showprogress::Bool=true)

    N = length(s)
    c = div(N, 2)
    tau = (t_Vec[end] - t_Vec[1]) / r

    # Cleaned up the h check to be more type-stable
    if h === nothing || all(==(nothing), h)
        f_hamiltonian = heis_hj_no_h(; per=per)
    else
        f_hamiltonian = heis_rf_for_h(h; per=per)
    end

    gate = PF_gate(f_hamiltonian, s, tau; order=1)

    Sz_all = Matrix{Float64}(undef, r+1, N)
    S_Bi   = Vector{Float64}(undef, r+1)

    psi_t = psi
    psi_vec = typeof(psi)[]

    # Removed the IdMPO and SzMPOs allocations from here!

    p = showprogress ? Progress(length(t_Vec); desc="MPS evo (r=$r)", dt=0.2) : nothing

    for (i, t) in enumerate(t_Vec)
        push!(psi_vec, deepcopy(psi_t))

        # 1. Fast Local Measurement
        if psi_t isa MPO
            Sz_all[i, :] = local_expect_mpo(psi_t, "Sz")
        else
            Sz_all[i, :] = expect(psi_t, "Sz"; sites=1:N)
        end

        # 2. Bipartition Entropy (Operator Entanglement for MPOs)
        S_Bi[i] = Entropy_Bipartition(psi_t, c) 

        if i == r+1
            showprogress && next!(p)
            break
        end

        # 3. Time Evolution Step
        psi_t = step!(psi_t, gate; cutoff=cutoff)

        # 4. Fast Trace Normalization
        if psi_t isa MPO
            trρ = fast_trace_mpo(psi_t)
            psi_t = psi_t / trρ
        end

        showprogress && next!(p)
    end

    return [psi_vec, Sz_all, S_Bi]
end

function Diff_trotter_r(s, psi,  t,  rv::Vector; h::Vector = [nothing], per:: Bool = false,  
                        cutoff::Real = 1e-10, doprint::Bool = true, showprogress::Bool = false)
    N = length(s) 

    All_Full_evos = []

    for r in rv
        tau = t/r
        ts = collect(0.0:tau:t)
        if doprint == true 
            print_simulation_parameters(N, tau, cutoff, t, r)
        end
        evopars = MPS_evo(s, psi, ts, r; h, per = per, showprogress = showprogress)
        push!(All_Full_evos, evopars)
    end
    return All_Full_evos
end


# ========================================= #
#            plot evolution                 #
# ========================================= #


function plot_spin_dynamics(ts, Sz_all, S_Bi, n, N, tau; gifname = "spins_bars.gif", fps = 30)
    # Mantener tus defaults exactos
    default(size = (1200, 300), left_margin = 8Plots.mm, bottom_margin = 6Plots.mm)

    # --- Lógica de tiempo real ---
    # Calculamos cada cuántos pasos (stride) debemos tomar un frame
    # para que la velocidad coincida con los FPS elegidos.
    # Stride = 1 / (tau * fps)
    every = max(1, round(Int, 1 / (tau * fps)))

    sz_maximum = maximum(abs.(Sz_all))*1.15

    # --- Animación ---
    anim = @animate for i in 1:every:size(Sz_all, 1)
        bar(1:N, Sz_all[i, :];
            ylim = (-sz_maximum, sz_maximum),
            xlabel = "site j",
            ylabel = "⟨Sᶻ⟩",
            legend = false,
            title = "t = $(round(ts[i]; digits = 2))",
            bar_width = 1,
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
    N = length(All_evos[1][1][1])
    c = div(N, 2)

    tvv = [
        collect(range(0, t, length = length(All_evos[i][1]))) for i in 1:(L)
    ]

    idv = []
    for i in 1:(L-1)
        id   = round.(Int, range(1, length(All_evos[L][1]),  length=length(All_evos[i][1])))
        push!(idv, id)
    end
    push!(idv, collect(eachindex(All_evos[L][1])))

    MPSv_all = [evos[1] for evos in All_evos]
    Szv_all  = [evos[2] for evos in All_evos]
    Sc_all   = [evos[3] for evos in All_evos]

    dist_MPS = []
    diff_Szv = []
    diff_Sc  = []

    for i in 1:(L-1)
        dmps = mps_distance(MPSv_all[i], MPSv_all[L][idv[i]])
        push!(dist_MPS, dmps)

        dSz  =  Szv_all[i] - Szv_all[L][idv[i], :]
        push!(diff_Szv, dSz)

        dSc  =  Sc_all[i] - Sc_all[L][idv[i]]
        push!(diff_Sc, dSc)
    end
    diff_Szv_c = [Szv[: , c] for Szv in diff_Szv]

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
            plot!(p3, tvv[i], dist_MPS[i]; label=labels[i])
        end

        # Layout: dos arriba, una abajo
        ptop = plot(p1, p2; layout=(1,2))
        p    = plot(ptop, p3; layout=(2,1), heights=[0.55, 0.45], size=(1200, 700))

    end
    return [dist_MPS, diff_Szv, diff_Sc, p]
end

function complete_plots(N, t, tau, evos)
    ts = collect(range(0, t, step = tau))
    c  = div(N,2)

    Sz_all = evos[end][2]
    S_Bi   = evos[end][3]

    combined, anim = plot_spin_dynamics(ts, Sz_all, S_Bi, c, N, tau, gifname = "Plots/spins_evo_magnon.gif")
    savefig(combined, "Plots/combined_magnon.png")

    d_plt = plot_diff(evos, t; plot_diffs = true)[end]
    savefig(d_plt, "Plots/differences_magnon.png")

    display(combined)
    display(anim)
    display(d_plt)
end


# ==================
# extras
# =================

function print_simulation_parameters(N, tau, cutoff, t, r)
    println("=============== Simulation Parameters ===============")
    @printf("   %-6s = %-12d  # Number of sites\n", "N", N)
    @printf("   %-6s = %-12d  # Number of Trotter steps\n", "r", r)
    @printf("   %-6s = %-12g  # Trotter step\n", "τ", tau)
    @printf("   %-6s = %-12g  # Total simulation time\n", "t", t)
    @printf("   %-6s = %-12.1e # Singular value cutoff\n", "cutoff", cutoff)
    println("=====================================================")
end

nothing
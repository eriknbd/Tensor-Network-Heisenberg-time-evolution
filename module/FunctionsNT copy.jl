using ITensors, ITensorMPS
using Random
using Printf
using Plots
using LaTeXStrings
using ProgressMeter

# ========================================= #
# ============== MPS utilities ============ #
# ========================================= #

function Entropy_Bipartition(psi::MPS, b)  
  psi = orthogonalize(psi, b)
  U,S,V = svd(psi[b], (linkinds(psi, b-1)..., siteinds(psi, b)...))
  SvN = 0.0
  for n=1:dim(S, 1)
    p = S[n,n]^2
    SvN -= p * log(p)
  end
  return SvN
end

function Schmidt_Rank(psi::MPS, b; tol::Real = 1e-8)  
  psi = orthogonalize(psi, b)
  U,S,V = svd(psi[b], (linkinds(psi, b-1)..., siteinds(psi, b)...))
  rank = 0
  for n = 1:dim(S, 1)
    s = S[n,n]
    if s > tol
      rank += 1
    end
  end
  return rank
end

# ========================================= #
# =========== TEBD / PF gate utils ======== #
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

function PF_gate(hj, sites, τ::Real; order::Int = 2)
  U_even, U_odd = time_operators_even_odd(hj, sites)
  if     order == 1
    return PF1_gate(U_even, U_odd, τ::Real)
  elseif order == 2
    return PF2_gate(U_even, U_odd, τ::Real)
  elseif order == 3
    return PF3_gate(U_even, U_odd, τ::Real)
  else
    throw(ArgumentError("Order must be 1, 2, or 3."))
  end
end

function step!(ψ, gate; cutoff=1e-8, maxdim=nothing)
  ψ = apply(gate, ψ; cutoff=cutoff, maxdim=maxdim)
  normalize!(ψ)
end

# one step: U_even(τ/2) → U_odd(τ) → U_even(τ/2)
function step_r2!(ψ, τ, U_even, U_odd; cutoff=1e-8, maxdim=nothing)
  ψ = apply(U_even(τ/2), ψ; cutoff=cutoff, maxdim=maxdim)
  ψ = apply(U_odd(τ),    ψ; cutoff=cutoff, maxdim=maxdim)
  ψ = apply(U_even(τ/2), ψ; cutoff=cutoff, maxdim=maxdim)
  normalize!(ψ)
  return ψ
end

# ========================================= #
# ============= Hamiltonians ============== #
# ========================================= #

# ---- nearestneighbor Heisenberg model in 1D ---- #
function heis_hj(j::Int, s; J::Real = 1.0, Δ::Real = 1.0) # 2-site Heisenberg term (j,j+1); S·S = SzSz + 1/2(S+S- + S-S+)
  @assert 1 ≤ j ≤ length(s) "heis_hj: j must be in 1:N"
  if j < length(s)
    return J*(Δ * op("Sz", s[j]) * op("Sz", s[j+1]) + 
      0.5  * op("S+", s[j]) * op("S-", s[j+1]) + 
      0.5  * op("S-", s[j]) * op("S+", s[j+1]) )
  end
  0 * op("Id", s[j])
end

# ---- nearestneighbor Heisenberg model in 1D with random local disorder ---- #
N = 50
W = 3                 # disorder strength
Random.seed!(1)         # reproducible
h_example = (2W) .* rand(N) .- W   # each h[i] ∈ [-W, W]

function heis_rf(j::Int, s; h::AbstractVector = h_example, J::Real=1.0, Δ::Real=1.0)
  N = length(s)
  @assert 1 ≤ j ≤ N "heis_rf_bond: j must be in 1:N"
  if j < N
    return heis_hj(j, s; J, Δ) + h[j] * (op("Sz", s[j]) * op("Id", s[j+1])) # add on-site terms
  end
  h[j] * op("Sz", s[j])
end


function RLD_vec(W::Real, N::Int; seed::Int=123, random::Bool=true)
    if random
        Random.seed!(seed)
        return (2W) .* rand(N) .- W   # h[i] ∈ [-W, W]
    else
        return fill(W, N)             # h[i] = W para todo i
    end
end


# ---- factory for fixed disorder vector h ----
function heis_rf_for_h(h::AbstractVector; J::Real=1.0, Δ::Real=1.0)
  return (j::Int, s) -> heis_rf(j, s; h=h, J=J, Δ=Δ)
end


# ========================================= #
# ================= Plots ================= #
# ========================================= #

function plot_spin_dynamics(ts, Sz_all, S_Bi, n, N, tau; gifname = "spins_bars.gif", fps = 30)
    # Mantener tus defaults exactos
    default(size = (1200, 300), left_margin = 8Plots.mm, bottom_margin = 6Plots.mm)

    # --- Lógica de tiempo real ---
    # Calculamos cada cuántos pasos (stride) debemos tomar un frame
    # para que la velocidad coincida con los FPS elegidos.
    # Stride = 1 / (tau * fps)
    every = max(1, round(Int, 1 / (tau * fps)))

    # --- Animación ---
    anim = @animate for i in 1:every:size(Sz_all, 1)
        bar(1:N, Sz_all[i, :];
            ylim = (-0.55, 0.55),
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


function print_simulation_parameters(N, tau, cutoff, t, r)
    println("=============== Simulation Parameters ===============")
    @printf("   %-6s = %-12d  # Number of sites\n", "N", N)
    @printf("   %-6s = %-12d  # Number of Trotter steps\n", "r", r)
    @printf("   %-6s = %-12g  # Trotter step\n", "τ", tau)
    @printf("   %-6s = %-12g  # Total simulation time\n", "t", t)
    @printf("   %-6s = %-12.1e # Singular value cutoff\n", "cutoff", cutoff)
    println("=====================================================")
end




# ========================================= #
# ====== Cotas de error de trotter ======== #
# ========================================= #


function OpSum_heis_hj(j::Int; J::Real = 1.0, Δ::Real = 1.0, N = 0)
    op = OpSum()
    op += Δ,  "Sz", j, "Sz", j + 1
    op += 0.5,"S+", j, "S-", j + 1
    op += 0.5,"S-", j, "S+", j + 1
    return op * J
end


# ---- nearestneighbor Heisenberg model in 1D with random local disorder ---- #
using Random

W = 1               # disorder strength
Random.seed!(123)         # seed
h_fun1(x) = (2W) .* rand(x) .- W   # h[i] ∈ [-W, W]
h = h_fun1(500)

function OpSum_heis_rf(j::Int; h = h, J::Real=1.0, Δ::Real=1.0, N = 50)
  if j < (N-1)
    op = OpSum_heis_hj(j; J = J, Δ = Δ)
    op += h[j] , "Sz", j
    return op
  end
  op = OpSum_heis_hj(j; J = J, Δ = Δ)
  op += h[j] , "Sz", j
  op += h[j+1] , "Sz", j+1
end


function OpSum_even_odd_Hamiltonian(hj, sites)
  N = length(sites)
  # Split bonds into odd and even sets:
  odd_bonds  = 1:2:(N-1)
  even_bonds = 2:2:(N-1)
  op_odd  = OpSum()
  op_even = OpSum()
  for j in even_bonds
    op_even += hj(j, N = N)
  end
  for j in odd_bonds
    op_odd += hj(j, N = N)
  end 

  H_even = MPO(op_even, sites)
  H_odd = MPO(op_odd, sites)

  H_even, H_odd   # Tuple{Vector{ITensor}, Vector{ITensor}}
end



comm(H1, H2) =  apply(H1, H2) - apply(H2, H1)

# ==================================================
# ==================================================

function _faster_Trotter_error_bound_U1_old(C, t, r)

    return C * (t^ 2) / (2r)

end


function faster_Trotter_error_fixed_tau_old(comm_norm, tvec::AbstractVector, τ;)

    [let r = t/τ
        _faster_Trotter_error_bound_U1_old(comm_norm, t, r)
     end for t in tvec]

end

function faster_Trotter_error_fixed_r_old(comm_norm, tvec::AbstractVector, r)

    [_faster_Trotter_error_bound_U1_old(comm_norm, t, r) for t in tvec]

end

function faster_Trotter_error_fixed_t_old(comm_norm, t, rvec) #en r no en tau

    [_faster_Trotter_error_bound_U1_old(comm_norm, t, r) for r in rvec]

end

# ==================================================
# ==================================================

# C1 = min(‖H1‖, ‖H2‖)
function C1(H1, H2; normfun=norm)
    min(normfun(H1), normfun(H2))
end

# C2 = 1/2 ‖[H1, H2]‖
function C2(H1, H2; normfun=norm)
    0.5 * normfun(comm(H1, H2))
end

# S = { ‖[H1,[H1,H2]]‖ , ‖[H2,[H2,H1]]‖ }
function Sset(H1, H2; normfun=norm)
    s1 = normfun(comm(H1, comm(H1, H2)))
    s2 = normfun(comm(H2, comm(H2, H1)))
    return s1, s2
end

# C3 = 1/12 ( min S + 1/2 max S )
function C3(H1, H2; normfun=norm)
    s1, s2 = Sset(H1, H2; normfun)
    mn = min(s1, s2)
    mx = max(s1, s2)
    (1/12) * (mn + 0.5*mx)
end



function _faster_Trotter_error_bound_U1(c1, c2, c3, I_norm, t, r)
    e1 = c2 * (t^2) / r
    e2 = c1 * (t / r) + c3 * (t^3) / (r^2)
    e3 = 2 * I_norm

    return min(e1, e2, e3)
end


function trotter_constants(H1, H2, sites; normfun = norm)
    Id = MPO(sites, "Id")
    I_norm = normfun(Id)

    return C1(H1, H2; normfun), C2(H1, H2; normfun), C3(H1, H2; normfun) , I_norm
end


function faster_Trotter_error_fixed_tau(Cs, tvec::AbstractVector, τ)

    [let r = t/τ
        _faster_Trotter_error_bound_U1(Cs[1], Cs[2], Cs[3], Cs[4], t, r)
     end for t in tvec]

end


function faster_Trotter_error_fixed_r(Cs, tvec::AbstractVector, r)

    [_faster_Trotter_error_bound_U1(Cs[1], Cs[2], Cs[3], Cs[4], t, r) for t in tvec]

end


function faster_Trotter_error_fixed_t(Cs, t::Real, rvec::AbstractVector) #en r no en tau

    [_faster_Trotter_error_bound_U1(Cs[1], Cs[2], Cs[3], Cs[4], t, r) for r in rvec]

end


function mpo_opnorm(M::MPO; nsweeps::Int=5, cutoff::Float64=1e-8, linkdims::Int=10)
    # Build -M†M as an MPO
    M = dense(M)
    Mdag  = swapprime(dag(M), 0 => 1)
    MdagM = apply(Mdag, M)
    MdagM *= -1

    # Get physical sites of the MPO
    sites = firstsiteinds(MdagM)

    # Initial random MPS
    psi0 = randomMPS(sites; linkdims=linkdims)

    # DMRG on -M†M
    energy, _ = dmrg(MdagM, psi0; nsweeps=nsweeps, cutoff=cutoff, outputlevel = 0)

    # Operator norm = sqrt(max eigenvalue of M†M)
    return sqrt(abs(energy))
end


function default_plots()
    theme(:bright)
    default(
        framestyle = :box, dpi = 200, size = (720, 480),
        legend = :bottomleft, guidefont = font(12), tickfont = font(10),
        legendfont = font(10), palette = :tab10, grid = :both, minorgrid = true,
        gridalpha = 0.7, minorgridalpha = 0.05, gridlinewidth = 1.2, minorgridlinewidth = 0.6,
    )
end


function º_old_trotter_error_plots(vtime_tau, err_tau, err_tau_o, vtime_r, err_r, err_r_o; title = "Trotter Error" )

    # ======== plot for constant tau ======== #
    plt_tau = plot(
        vtime_tau, err_tau;
        xscale = :log10, yscale = :log10,
        lw = 2.5, label = "new", color = "red",
    )

    plot!(plt_tau, 
        vtime_tau, err_tau_o; 
        lw = 2.5, label = "old", color = "black", 
        ls = :dash,
    )

    xlabel!(plt_tau, L"t")
    ylabel!(plt_tau, L"\mathrm{Trotter\ error\ upper\ bound}")

    # Add constant tau label
    x1, x2 = xlims(plt_tau); y1, y2 = ylims(plt_tau)
    gx(a,b,θ) = a * (b/a)^θ
    px = gx(x1, x2, 0.08)
    py = gx(y1, y2, 0.92)
    annotate!(plt_tau, px, py, Plots.text(L"\tau = 10^{-3}", 12, :black, :left))

    # ======== plot for constant r ======== #

    plt_r = plot(
        vtime_r, err_r;
        xscale = :log10, yscale = :log10,
        lw = 2.5, label = "new", color = "red",
    )

    plot!(plt_r, vtime_r, err_r_o; lw = 2.5, ls = :dash, label = "old", color = "black")

    xlabel!(plt_r, L"t")

    # Add constant r label
    x1, x2 = xlims(plt_r); y1, y2 = ylims(plt_r)
    gx(a,b,θ) = a * (b/a)^θ
    px = gx(x1, x2, 0.08)
    py = gx(y1, y2, 0.92)
    annotate!(plt_r, px, py, Plots.text(L"r = 10^{4}", 12, :black, :left))

    
    combined = plot(plt_tau, plt_r; layout = (1, 2), size = (480*2.2, 480),     
        plot_title = title, 
        plot_titlefont = font(16),
        left_margin = 6Plots.mm,  
        bottom_margin = 6Plots.mm
    )
    return combined
end



function trotter_error_plots(
    vtime_tau, err_tau, err_tau_o,
    vtime_r,   err_r,   err_r_o,
    rvec,      errs_t,  errs_old_t;
    title    = "Trotter Error",
    tau_text = L"\tau = Cte",
    r_text   = L"r = Cte",
    t_text   = L"t = Cte",
)

    # helper para coordenadas "bonitas" en log-log
    gx(a,b,θ) = a * (b/a)^θ

    # ======== plot for constant tau ======== #
    plt_tau = plot(
        vtime_tau, err_tau;
        xscale = :log10, yscale = :log10,
        lw = 2.5, label = "new", color = "red",
    )
    plot!(plt_tau,
        vtime_tau, err_tau_o;
        lw = 2.5, label = "old", color = "black", ls = :dash,
    )
    xlabel!(plt_tau, L"t")
    ylabel!(plt_tau, L"\mathrm{Trotter\ error\ upper\ bound}")

    x1, x2 = xlims(plt_tau); y1, y2 = ylims(plt_tau)
    annotate!(plt_tau,
        gx(x1, x2, 0.08), gx(y1, y2, 0.92),
        Plots.text(tau_text, 12, :black, :left)
    )

    # ======== plot for constant r ======== #
    plt_r = plot(
        vtime_r, err_r;
        xscale = :log10, yscale = :log10,
        lw = 2.5, label = "new", color = "red",
    )
    plot!(plt_r,
        vtime_r, err_r_o;
        lw = 2.5, ls = :dash, label = "old", color = "black",
    )
    xlabel!(plt_r, L"t")

    x1, x2 = xlims(plt_r); y1, y2 = ylims(plt_r)
    annotate!(plt_r,
        gx(x1, x2, 0.08), gx(y1, y2, 0.92),
        Plots.text(r_text, 12, :black, :left)
    )

    # ======== plot for constant t (error vs r) ======== #
    plt_t = plot(
        rvec, errs_t;
        xscale = :log10, yscale = :log10,
        lw = 2.5, label = "new", color = "red",
    )
    plot!(plt_t,
        rvec, errs_old_t;
        lw = 2.5, label = "old", color = "black", ls = :dash,
    )
    xlabel!(plt_t, L"r")

    x1, x2 = xlims(plt_t); y1, y2 = ylims(plt_t)
    annotate!(plt_t,
        gx(x1, x2, 0.15), gx(y1, y2, 0.92),
        Plots.text(t_text, 12, :black, :left)
    )

    combined = plot(
        plt_tau, plt_r, plt_t;
        layout = (1, 3),
        size = (480*3.3*1.2, 480*1.2),
        plot_title = title,
        plot_titlefont = font(16),
        left_margin = 9Plots.mm,
        bottom_margin = 9Plots.mm,
    )
    return combined
end





# ========================================= #
# ===== Differences for a changing r ====== #
# ========================================= #

### Funciones para calcular Diferencias en la evolucion a diferentes r


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


# ======================================================
# ======================================================


function MPS_evo(s, psi, t_Vec, r; W::Real = 0, randomh::Bool = true, seed::Int = 123, cutoff::Real = 1e-10)
    N = length(s)
    c = div(N, 2)
    tau =  (t_Vec[end] - t_Vec[1]) / r
    

    disorder_vecotor =  RLD_vec(W, N, seed = seed, random = randomh)
    f_hamiltonian = heis_rf_for_h(disorder_vecotor) # each h[i] ∈ [-W, W] ts

    gate = PF_gate(f_hamiltonian, s, tau; order = 1)

    # --- storage ---
    Sz_all = Matrix{Float64}(undef, r+1, N)   # row i = at time ts[i], columns = sites 1..N
    S_Bi   = Vector{Float64}(undef, r+1)

    psi_t = psi

    psi_vec = MPS[]

    # Compute <Sz> and SvN at each time step
    # then apply the gates to go to the next time
    @showprogress for (i, t) in enumerate(t_Vec)

        push!(psi_vec, deepcopy(psi_t))

        Sz_all[i, :] = expect(psi_t, "Sz"; sites=1:N) # local expectation value of Sz at every site → ⟨ψ∣Sz​∣ψ⟩
        S_Bi[i] = Entropy_Bipartition(psi_t, c)      # von Neumann entropy of the bipartition
        if i == r+1
            break
        end
        psi_t = step!(psi_t, gate; cutoff=cutoff)
    end

    return [psi_vec, Sz_all, S_Bi] # Vector of MPSs (steps), every site ⟨ψ∣Sz​∣ψ⟩, von Neumann entropy of the bipartition.
end



function Diff_trotter_r(s, psi,  t,  rv::Vector; W = 0, randomh::Bool = true,  
                        cutoff::Real = 1e-10,   seed::Int = 123, doprint::Bool = true,)
    N = length(s) 

    All_Full_evos = []

    @showprogress color=:cyan for r in rv
        tau = t/r
        ts = collect(0.0:tau:t)
        if doprint == true 
            print_simulation_parameters(N, tau, cutoff, t, r)
        end
        evopars = MPS_evo(s, psi, ts, r; W = W, randomh = randomh, seed = seed)
        push!(All_Full_evos, evopars)
    end
    return All_Full_evos
end

# ======================================================
# ======================================================


function plot_diff(All_evos, t; plot_diffs::Bool = true)
    L    = length(All_evos)

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




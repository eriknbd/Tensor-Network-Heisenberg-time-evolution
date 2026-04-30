using ITensors, ITensorMPS
using Random
using Printf
using Plots
using LaTeXStrings
using ProgressMeter


# ========================================= #
#             Initial State                 #
# ========================================= #


function purified_domainwall(L::Int, μ;
                             conserve_qns::Bool=true,
                             T::Type{<:Number}=ComplexF64)

  # Make distinct physical + ancilla site indices
  sitesP = siteinds("S=1/2", L; conserve_qns=conserve_qns)
  sitesA = siteinds("S=1/2", L; conserve_qns=conserve_qns)
  sitesP = addtags.(sitesP, "P")
  sitesA = addtags.(sitesA, "A")

  # Interleave: (P1,A1,P2,A2,...)
  sitesPA = Vector{Index}(undef, 2L)
  for i in 1:L
    sitesPA[2i-1] = sitesP[i]
    sitesPA[2i]   = sitesA[i]
  end

  # Bonds: dim=2 between (P_  i, A_i), dim=1 between (A_i, P_{i+1})
  linkdims = [isodd(b) ? 2 : 1 for b in 1:(2L-1)]

  psi = MPS(T, sitesPA; linkdims=linkdims)

  Zlocal = 2 * cosh(μ/2)
  invsqrtZ = inv(sqrt(Zlocal))

  for i in 1:L
    sP = sitesP[i]
    sA = sitesA[i]

    η = (i <= (L ÷ 2)) ? +1 : -1
    a = exp(η*μ/4) * invsqrtZ
    b = exp(-η*μ/4) * invsqrtZ

    # Build the 2-site state tensor on (sP, sA).
    # Convention for "S=1/2" is typically: 1="Up", 2="Dn".
    ψPA = ITensor(T, sP, sA)
    ψPA[sP=>1, sA=>2] = a   # |Up_P, Dn_A>
    ψPA[sP=>2, sA=>1] = b   # |Dn_P, Up_A>

    # Split into neighboring MPS tensors at sites (2i-1, 2i)
    U, S, V = svd(ψPA, sP)      # U(sP,ℓ), S(ℓ,ℓ'), V(ℓ',sA)
    B = S * V                   # B(ℓ,sA)

    psi[2i-1] = U               # tensor for P_i
    psi[2i]   = B               # tensor for A_i
  end

  return psi, sitesP, sitesA, sitesPA
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

function _is_interleaved_purified_layout(psi::MPS, sitesP)
    N = length(sitesP)
    length(psi) == 2N || return false
    for i in 1:N
        hasind(psi[2i - 1], sitesP[i]) || return false
    end
    return true
end

function even_odd_Hamiltonian_interleaved(hj, sitesP, sitesPA)
    N = length(sitesP)
    odd_bonds = 1:2:(N - 1)
    even_bonds = 2:2:(N - 1)
    H_even = [hj(j, sitesP) * op("Id", sitesPA[2j]) for j in even_bonds]
    H_odd = [hj(j, sitesP) * op("Id", sitesPA[2j]) for j in odd_bonds]
    return H_even, H_odd
end

function time_operators_even_odd_interleaved(hj, sitesP, sitesPA)
    H_even, H_odd = even_odd_Hamiltonian_interleaved(hj, sitesP, sitesPA)
    U_even(tau) = exp.(-im * tau * H_even)
    U_odd(tau) = exp.(-im * tau * H_odd)
    return U_even, U_odd
end

function PF_gate_interleaved(hj, sitesP, sitesPA, τ::Real; order::Int = 2)
    U_even, U_odd = time_operators_even_odd_interleaved(hj, sitesP, sitesPA)
    if order == 1
        return PF1_gate(U_even, U_odd, τ)
    elseif order == 2
        return PF2_gate(U_even, U_odd, τ)
    elseif order == 3
        return PF3_gate(U_even, U_odd, τ)
    else
        throw(ArgumentError("Order must be 1, 2, or 3."))
    end
end

function step!(ψ::MPS, gate; cutoff=1e-10, normalize::Bool=true)
    ψ = apply(gate, ψ; cutoff=cutoff)
    if normalize
        normalize!(ψ)
    end
    return ψ
end

# ========================================= #
#             Do evolution                  #
# ========================================= #


function Entropy_Bipartition(psi::MPS, b::Int; base=2, tol=1e-20)
    orthogonalize!(psi, b)
    left_inds = b == 1 ?
        Tuple(siteinds(psi, b)) :
        tuple(linkind(psi, b - 1), siteinds(psi, b)...)
    _, S, _ = svd(psi[b], left_inds)
    logfun = base == 2 ? log2 : log
    SvN = 0.0
    for n in 1:dim(S, 1)
        p = abs2(S[n, n])
        if p > tol
            SvN -= p * logfun(p)
        end
    end
    return SvN
end

function _mps_distance(ψ::MPS, ϕ::MPS)
    ψϕ = inner(ψ, ϕ)

    # d = sqrt( ⟨ψ∣ψ⟩ + ⟨ϕ∣ϕ⟩ - 2ℜ⟨ψ∣ϕ⟩ )
    # Suponiendo estados normalizados: d = sqrt( 2 - 2ℜ⟨ψ∣ϕ⟩ ) 
    d2 = real(2 - 2*real(ψϕ))

    return sqrt(max(d2, 0.0))
end

function mps_distance(ψv, ϕv)
    n = length(ψv)
    return [_mps_distance(ψv[i], ϕv[i]) for i in 1:n]
end

function MPS_evo(
    s,
    psi,
    t_Vec,
    r;
    h::Union{Nothing,AbstractVector}=nothing,
    per::Bool=false,
    cutoff::Real=1e-10,
    showprogress::Bool=true,
    store_states::Bool=false,
    snapshot_every::Int=0,
    sz_every::Int=1,
    entropy_every::Int=1,
    normalize_every::Int=1,
    trotter_order::Int=1,
)
    snapshot_every < 0 && throw(ArgumentError("snapshot_every must be >= 0"))
    sz_every < 1 && throw(ArgumentError("sz_every must be >= 1"))
    entropy_every < 1 && throw(ArgumentError("entropy_every must be >= 1"))
    normalize_every < 1 && throw(ArgumentError("normalize_every must be >= 1"))

    N = length(s)
    c = div(N, 2)
    nt = length(t_Vec)
    tau = (t_Vec[end] - t_Vec[1]) / r

    if h === nothing || all(isnothing, h)
        f_hamiltonian = heis_hj_no_h(; per = per)
    else
        f_hamiltonian = heis_rf_for_h(h; per = per)
    end

    use_local_gate_engine = !per && _is_interleaved_purified_layout(psi, s)
    gate = if use_local_gate_engine
        sitesPA = [siteind(psi, n) for n in 1:length(psi)]
        PF_gate_interleaved(f_hamiltonian, s, sitesPA, tau; order = trotter_order)
    else
        PF_gate(f_hamiltonian, s, tau; order = trotter_order)
    end

    if length(psi) == 2N
        phys_pos = 1:2:(2N - 1)
        cut_pos = 2c
    else
        phys_pos = 1:N
        cut_pos = c
    end

    Sz_all = Matrix{Float64}(undef, nt, N)
    S_Bi = Vector{Float64}(undef, nt)
    fill!(Sz_all, NaN)
    fill!(S_Bi, NaN)

    psi_t = deepcopy(psi)
    psi_vec = MPS[]
    stored_steps = Int[]

    p = showprogress ? Progress(nt; desc="MPS evo (r=$r)", dt=0.2) : nothing

    for i in 1:nt
        if store_states && snapshot_every > 0
            do_snapshot = (i == 1) || (i == nt) || ((i - 1) % snapshot_every == 0)
            if do_snapshot
                push!(psi_vec, deepcopy(psi_t))
                push!(stored_steps, i)
            end
        end

        do_sz = (i == 1) || (i == nt) || (sz_every == 1) || (i % sz_every == 0)
        if do_sz
            Sz_all[i, :] = real.(expect(psi_t, "Sz"; sites = phys_pos))
        end

        do_entropy = (i == 1) || (i == nt) || (entropy_every == 1) || (i % entropy_every == 0)
        if do_entropy
            S_Bi[i] = Entropy_Bipartition(psi_t, cut_pos)
        end

        if i < nt
            do_normalize = (normalize_every == 1) || (i % normalize_every == 0) || (i == nt - 1)
            psi_t = step!(psi_t, gate; cutoff = cutoff, normalize = do_normalize)
        end
        showprogress && next!(p)
    end

    return (psi_vec, Sz_all, S_Bi, stored_steps)
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
    store_states::Bool=false,
    snapshot_every::Int=0,
    sz_every::Int=1,
    entropy_every::Int=1,
    normalize_every::Int=1,
    trotter_order::Int=1,
)
    N = length(s)

    TOut = Tuple{Vector{MPS}, Matrix{Float64}, Vector{Float64}, Vector{Int}}
    All_Full_evos = Vector{TOut}(undef, length(rv))

    for (ir, r) in enumerate(rv)
        tau = t / r
        ts = collect(range(0.0, t; length = r + 1))
        if doprint
            print_simulation_parameters(N, tau, cutoff, t, r)
        end
        All_Full_evos[ir] = MPS_evo(
            s,
            psi,
            ts,
            r;
            h=h,
            per=per,
            cutoff=cutoff,
            showprogress=showprogress,
            store_states=store_states,
            snapshot_every=snapshot_every,
            sz_every=sz_every,
            entropy_every=entropy_every,
            normalize_every=normalize_every,
            trotter_order=trotter_order,
        )
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

    valid_sz = Sz_all[.!isnan.(Sz_all)]
    sz_maximum = isempty(valid_sz) ? 1.0 : maximum(abs.(valid_sz)) * 1.15
    valid_rows = findall(i -> all(!isnan, Sz_all[i, :]), 1:size(Sz_all, 1))
    isempty(valid_rows) && error("No valid Sz samples to plot. Increase measurement frequency (sz_every).")
    plot_rows = valid_rows[1:every:end]

    # --- Animación ---
    anim = @animate for i in plot_rows
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
    sz_mask = .!isnan.(Sz_all[:, n])
    Szc_plot = plot(ts[sz_mask], Sz_all[sz_mask, n];
        xlabel = "t",
        ylabel = "⟨Sᶻ_$n⟩",
        legend = false,
        title = "Site j = $n",
    )

    entropy_mask = .!isnan.(S_Bi)
    SvN_plot = plot(ts[entropy_mask], S_Bi[entropy_mask];
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
        collect(range(0, t, length = size(All_evos[i][2], 1))) for i in 1:L
    ]

    idv = Vector{Vector{Int}}(undef, L)
    for i in 1:(L-1)
        idv[i] = round.(Int, range(1, size(All_evos[L][2], 1), length=size(All_evos[i][2], 1)))
    end
    idv[L] = collect(1:size(All_evos[L][2], 1))

    MPSv_all = [evos[1] for evos in All_evos]
    Szv_all  = [evos[2] for evos in All_evos]
    Sc_all   = [evos[3] for evos in All_evos]
    stored_steps_all = [length(evos) >= 4 ? evos[4] : collect(1:length(evos[1])) for evos in All_evos]

    dist_MPS = has_states ? Vector{Vector{Float64}}(undef, L-1) : Vector{Vector{Float64}}()
    dist_tv  = has_states ? Vector{Vector{Float64}}(undef, L-1) : Vector{Vector{Float64}}()
    diff_Szv = Vector{Matrix{Float64}}(undef, L-1)
    diff_Sc  = Vector{Vector{Float64}}(undef, L-1)
    ref_state_by_step = Dict{Int,MPS}()

    if has_states
        for (k, step) in enumerate(stored_steps_all[L])
            ref_state_by_step[step] = MPSv_all[L][k]
        end
    end

    for i in 1:(L-1)
        if has_states
            psi_i = MPS[]
            psi_ref = MPS[]
            times_i = Float64[]
            for (k, step_i) in enumerate(stored_steps_all[i])
                mapped_ref_step = idv[i][step_i]
                if haskey(ref_state_by_step, mapped_ref_step)
                    push!(psi_i, MPSv_all[i][k])
                    push!(psi_ref, ref_state_by_step[mapped_ref_step])
                    push!(times_i, tvv[i][step_i])
                end
            end
            if isempty(psi_i)
                dist_MPS[i] = Float64[]
                dist_tv[i] = Float64[]
            else
                dist_MPS[i] = mps_distance(psi_i, psi_ref)
                dist_tv[i] = times_i
            end
        end

        diff_Szv[i] = Szv_all[i] - Szv_all[L][idv[i], :]

        diff_Sc[i] = Sc_all[i] - Sc_all[L][idv[i]]
    end
    diff_Szv_c = [Szv[:, c] for Szv in diff_Szv]

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

        for i in 1:(L-1)
            plot!(p1, tvv[i], diff_Szv_c[i])
            plot!(p2, tvv[i], diff_Sc[i])
            if has_states && !isempty(dist_MPS[i])
                plot!(p3, dist_tv[i], dist_MPS[i]; label=labels[i])
            end
        end

        if !has_states || all(isempty, dist_MPS)
            annotate!(p3, t / 2, 0.0, text("State distance unavailable (no matching snapshots)", 9))
        end

        ptop = plot(p1, p2; layout=(1,2))
        p    = plot(ptop, p3; layout=(2,1), heights=[0.55, 0.45], size=(1200, 700))

    end
    return (dist_MPS, diff_Szv, diff_Sc, p)
end

function complete_plots(N, t, tau, evos; save_at = "Plots/complete_plot")
    ts = collect(range(0, t, length = size(evos[end][2], 1)))
    c  = div(N,2)

    Sz_all = evos[end][2]
    S_Bi   = evos[end][3]


    combined, anim = plot_spin_dynamics(ts, Sz_all, S_Bi, c, N, tau, gifname = save_at * "evo.gif")
    savefig(combined, save_at * "combined.png")

    d_plt = nothing
    if length(evos) != 1
        d_plt = plot_diff(evos, t; plot_diffs = true)[end]
        savefig(d_plt, save_at * "differences.png")
    end

    display(combined)
    display(anim)
    if d_plt !== nothing
        display(d_plt)
    end
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

using JuMP
using SCS
using LinearAlgebra
using Clarabel
using CSV
using DataFrames

# Solver
const SOLVER = SCS.Optimizer

# =============================================================================
#  UTILITY FUNCTIONS
# =============================================================================

# Creates a random matrix of dimensions n x n
function hermitian_matrix(n::Int)
    A_real = randn(n, n); A_real = A_real + A_real'   # Symmetric real part
    A_imag = randn(n, n); A_imag = A_imag - A_imag'   # Antisymmetric imaginary part
    return A_real + im * A_imag
end

# Creates a random density operator of dimensions n x n
function rand_state(n::Int)
    A = randn(ComplexF64, n, n)                       # Random matrix n x n
    ρ  = A' * A                                       # Make it hermitian A^† A
    return ρ / tr(ρ)                                  # Normalize it's trace
end

# Given a set of basis vectors, return their projectors
function projectors(basis::Vector)
    return [ket * ket' for ket in basis]
end

# Calculate expected values for each measurement
function expected_values(ρ::AbstractMatrix, M::Vector)
    return [real(tr(Mi * ρ)) for Mi in M]
end

# Create a random POVM with o elements of dimension d
function random_POVM(o::Int, d::Int)
    M = [begin A = randn(ComplexF64, d, d); A * A' end for _ in 1:o]  # Generate positive semidefinite operators

    # Modify the measurement operators such that they sum the identity
    S  = sum(M)
    U, s, _ = svd(S)
    S_inv_sqrt = U * Diagonal(1 ./ sqrt.(s)) * U'
    return [S_inv_sqrt * Mi * S_inv_sqrt for Mi in M]
end

# The generalized σₓ for qudits of dimension d
function X_operator(d::Int)
    X = zeros(ComplexF64, d, d)
    for j in 1:d 
        X[mod1(j+1, d), j] = 1.0
    end
    return X
end

# Discrete Fourier Transform of dimension d
function F_operator(d::Int)
    F = zeros(ComplexF64, d, d)
    ω = exp(2π * im / d)
    for j in 1:d
        for k in 1:d
            F[j, k] = ω^((j-1)*(k-1))
        end
    end

    F = F / sqrt(d)
end

# Tunning Measurements
function tunning_M(d::Int, α::Float64)
    F = F_operator(d)
    X = X_operator(d)

    # Usar schur en lugar de eigen: garantiza Q unitaria
    SF = schur(F)
    SX = schur(X)

    λ_F = SF.values   # autovalores en la forma de Schur diagonal
    Q_F = SF.vectors  # unitaria garantizada

    λ_X = SX.values
    Q_X = SX.vectors

    F_pow = Q_F * Diagonal(λ_F .^ (1 - α)) * Q_F'
    X_pow = Q_X * Diagonal(λ_X .^ (-α / 2)) * Q_X'

    return F_pow * X_pow
end

# Probability P(A_a = B_b + k) = ∑_{j=0}^{d-1} Tr(ρ Π_{j|a} ⊗ Π_{j+k mod d|b})
function prob_eq_mod(ρ, Proj_Aa::Vector, Proj_Bb::Vector, k::Int, d::Int)
    P = 0.0
    for j in 1:d
        idx_B = mod1(j+k, d)
        kron_ProjAB = kron(Proj_Aa[j], Proj_Bb[idx_B])
        P += real(tr(ρ * kron_ProjAB))
    end
    return P
end

# CGLMP expression
function CGLMP_exp(ρ, Proj_A::Vector, Proj_B::Vector, d::Int)
    @assert length(Proj_A) == 2            # There must be two sets of measurements for A
    @assert length(Proj_B) == 2            # There must be two sets of measurements for B
    @assert all(length.(Proj_A) .== d)     # Each measurement must have d elements of measurement for A
    @assert all(length.(Proj_B) .== d)     # Each measurement must have d elements of measurement for B

    Id = 0.0
    for k in 0:(div(d,2) - 1)
        weight = (1 - 2*k/(d-1))

        # Positive terms
        Positive = prob_eq_mod(ρ, Proj_A[1], Proj_B[1], k, d)
        Positive += prob_eq_mod(ρ, Proj_A[2], Proj_B[1], -k-1, d)
        Positive += prob_eq_mod(ρ, Proj_A[2], Proj_B[2], k, d)
        Positive += prob_eq_mod(ρ, Proj_A[1], Proj_B[2], -k, d)

        # Negative terms
        Negative = prob_eq_mod(ρ, Proj_A[1], Proj_B[1], -k-1, d)
        Negative += prob_eq_mod(ρ, Proj_A[2], Proj_B[1], k, d)
        Negative += prob_eq_mod(ρ, Proj_A[2], Proj_B[2], -k-1, d)
        Negative += prob_eq_mod(ρ, Proj_A[1], Proj_B[2], k+1, d)

        Id += weight * (Positive - Negative)
    end

    return Id
end

# Partial trace of σ, indicating the dimensions of each subsystem in dims as vector and in traced_sys the index/indexes of subsystems to trace
function partial_trace(σ, dims::Vector{Int}, traced_sys)
    traced_sys = isa(traced_sys, Int) ? [traced_sys] : collect(traced_sys)
    n = length(dims)
    traced_dims = [dims[ts + 1] for ts in traced_sys]

    result = nothing
    for idx in Iterators.product([0:d-1 for d in traced_dims]...)
        idx_arr = collect(idx)
        K  = ones(ComplexF64, 1, 1)
        ti = 0
        for i in 0:n-1
            if i in traced_sys
                e = zeros(ComplexF64, dims[i+1], 1)
                e[idx_arr[ti+1] + 1, 1] = 1.0
                K  = kron(K, e')
                ti += 1
            else
                K = kron(K, Matrix{ComplexF64}(I, dims[i+1], dims[i+1]))
            end
        end
        term   = K * σ * K'
        result = isnothing(result) ? term : result + term
    end
    return result
end

# Given a state ρ_{A B₁ B₂ … Bₖ}, trace over all subsystems B₂,…,Bₖ, returning ρ_{A B₁}
function partial_trace_Bsystems(ρ, dA::Int, dB::Int, k::Int)
    ρ_red = ρ
    dims  = [dA; fill(dB, k)]
    for _ in 1:k-1
        ρ_red = partial_trace(ρ_red, dims, length(dims) - 1)
        pop!(dims)
    end
    return ρ_red
end

# Return the partial transpose over the subsystem A
function partial_transpose_A(M, dA::Int, dB::Int)
    d = dA * dB
    return [M[(col ÷ dB) * dB + (row % dB) + 1,
              (row ÷ dB) * dB + (col % dB) + 1]
            for row in 0:d-1, col in 0:d-1]
end

# 
function _mixed_radix_decode(n::Int, dims::Vector{Int})
    idx = zeros(Int, length(dims))
    for i in length(dims):-1:1
        idx[i] = n % dims[i]
        n      = n ÷ dims[i]
    end
    return idx
end

# 
function _mixed_radix_encode(idx::Vector{Int}, dims::Vector{Int})
    n = 0
    for i in eachindex(dims)
        n = n * dims[i] + idx[i]
    end
    return n
end

# Partial transpose over the subsystem k par arbitrary dimensions
function partial_transpose_sys(M, dims::Vector{Int}, k::Int)
    D = prod(dims)
    return [begin
        r_multi    = _mixed_radix_decode(row, dims)
        c_multi    = _mixed_radix_decode(col, dims)
        r_new      = copy(r_multi); r_new[k+1] = c_multi[k+1]
        c_new      = copy(c_multi); c_new[k+1] = r_multi[k+1]
        M[_mixed_radix_encode(r_new, dims)+1,
          _mixed_radix_encode(c_new, dims)+1]
    end for row in 0:D-1, col in 0:D-1]
end

# Permutation operator between subsystems i and j of a system A⊗B₁⊗B₂⊗…⊗Bₖ with dimensions [dA, dB,…,dB]
function permutation_operator(dA::Int, dB::Int, k::Int, i::Int, j::Int)
    dims = [dA; fill(dB, k)]
    D    = dA * dB^k
    P    = zeros(ComplexF64, D, D)
    for idx in 0:D-1
        multi           = _mixed_radix_decode(idx, dims)
        multi_perm      = copy(multi)
        multi_perm[i+1], multi_perm[j+1] = multi_perm[j+1], multi_perm[i+1]
        idx_perm        = _mixed_radix_encode(multi_perm, dims)
        P[idx_perm+1, idx+1] = 1.0
    end
    return P
end

# =============================================================================
#  SEMIDEFINITE PROGRAMMING APPLIED TO QUANTUM INFORMATION
# =============================================================================

# Find the maximum eigenvalue of a hermitian operator via SDP
function max_eigenvalue_sdp(H::AbstractMatrix)
    n = size(H, 1)
    @assert isapprox(H, H', atol=1e-10) "H must be Hermitian"

    model = Model(SOLVER); set_silent(model)
    @variable(model, ρ[1:n, 1:n] in HermitianPSDCone())
    @constraint(model, real(tr(ρ)) == 1)
    @objective(model, Max, real(tr(H * ρ)))
    optimize!(model)
    return objective_value(model), value.(ρ)
end

# Trace norm : ∥H∥₁ = ∑ᵢ |λᵢ| via SDP
function trace_norm_sdp(H::AbstractMatrix)
    n = size(H, 1)
    @assert isapprox(H, H', atol=1e-10) "H must be Hermitian"

    model = Model(SOLVER); set_silent(model)
    @variable(model, X[1:n, 1:n] in HermitianPSDCone())
    @constraint(model, H + X in HermitianPSDCone())
    @constraint(model, X - H in HermitianPSDCone())
    @objective(model, Min, real(tr(X)))
    optimize!(model)
    return objective_value(model), value.(X)
end

# Operator norm : ∥H∥∞ = max |λᵢ| via SDP    
function operator_norm_sdp(H::AbstractMatrix)
    n = size(H, 1)
    @assert isapprox(H, H', atol=1e-10) "H muste be Hermitian"

    model = Model(SOLVER); set_silent(model)
    I_n = Matrix{ComplexF64}(I, n, n)
    @variable(model, x)
    @constraint(model, H + x * I_n in HermitianPSDCone())
    @constraint(model, x * I_n - H in HermitianPSDCone())
    @objective(model, Min, x)
    optimize!(model)
    return objective_value(model), value(x)
end

# Quantum State Estimation via Trace distance : Given measurement operators and their experimental expected values, do a tomography
function trace_distance_sdp(σ::AbstractMatrix, M::Vector, m::Vector)
    n = size(σ, 1)

    model = Model(SOLVER); set_silent(model)
    @variable(model, ρ[1:n, 1:n] in HermitianPSDCone())
    @variable(model, X[1:n, 1:n] in HermitianPSDCone())  # X ≥ 0 por construcción

    @constraint(model, real(tr(ρ)) == 1)
    @constraint(model, X - (ρ - σ) in HermitianPSDCone())
    @constraint(model, X + (ρ - σ) in HermitianPSDCone())
    for i in eachindex(M)
        @constraint(model, real(tr(M[i] * ρ)) == real(m[i]))
    end
    @objective(model, Min, 0.5 * real(tr(X)))
    optimize!(model)
    return objective_value(model), value.(ρ)
end

# Quantum State Estimation via Fidelity : Given measurement operators and their experimental expected values, do a tomography
function fidelity_sdp(σ::AbstractMatrix, M::Vector, m::Vector)
    n = size(σ, 1)

    model = Model(SOLVER); set_silent(model)
    @variable(model, ρ[1:n, 1:n] in HermitianPSDCone())
    @variable(model, Y[1:n, 1:n], Hermitian)   # off-diagonal real block
    @variable(model, Z[1:n, 1:n], Hermitian)   # off-diagonal imaginary block

    # Block matrix
    block = [ρ              (Y .+ im .* Z);
             (Y .- im .* Z)  σ            ]
    @constraint(model, block in HermitianPSDCone())
    @constraint(model, real(tr(ρ)) == 1)
    for i in eachindex(M)
        @constraint(model, real(tr(M[i] * ρ)) == real(m[i]))
    end
    @objective(model, Max, real(tr(Y)))
    optimize!(model)
    return objective_value(model)^2, value.(ρ)
end

# Quantum Marginal Problem : Given the marginal density operators of a tripartite system, find the global state
function marginal_problem(ρ_XYZ, ρ_XY, ρ_XZ, ρ_YZ)
    dXY, dXZ, dYZ = size(ρ_XY,1), size(ρ_XZ,1), size(ρ_YZ,1)
    dX   = Int(round(sqrt(dXY * dXZ / dYZ)))
    dY   = Int(round(sqrt(dXY * dYZ / dXZ)))
    dZ   = Int(round(sqrt(dXZ * dYZ / dXY)))
    dims = [dX, dY, dZ]
    dXYZ = dX * dY * dZ

    model = Model(SOLVER); set_silent(model)
    @variable(model, σ[1:dXYZ, 1:dXYZ] in HermitianPSDCone())
    @variable(model, Γ[1:dXYZ, 1:dXYZ], Hermitian)

    @constraint(model, Γ - (σ - ρ_XYZ) in HermitianPSDCone())
    @constraint(model, Γ + (σ - ρ_XYZ) in HermitianPSDCone())
    @constraint(model, partial_trace(σ, dims, 2) .== ρ_XY)
    @constraint(model, partial_trace(σ, dims, 1) .== ρ_XZ)
    @constraint(model, partial_trace(σ, dims, 0) .== ρ_YZ)
    @constraint(model, real(tr(σ)) == 1)
    @objective(model, Min, 0.5 * real(tr(Γ)))
    optimize!(model)
    return objective_value(model), value.(σ)
end

# Quantum Measurement Estimation (Primal) : Find a POVM that reproduces the statistics. ρs vector of states and Probs matrix (num_a x num_x) with values p(a|x)
function QME(ρs::Vector, Probs::Matrix)
    num_a, num_x = size(Probs)
    d = size(ρs[1], 1)
    @assert d == num_a && num_x == length(ρs)

    model = Model(SOLVER); set_silent(model)
    M_vars = [@variable(model, [1:d, 1:d] in HermitianPSDCone()) for _ in 1:num_a]
    @variable(model, δ >= 0)

    @constraint(model, sum(M_vars) .== Matrix{ComplexF64}(I, d, d))
    for a in 1:num_a, x in 1:num_x
        p_ax = real(tr(M_vars[a] * ρs[x]))
        @constraint(model, p_ax >= Probs[a,x] - δ)
        @constraint(model, p_ax <= Probs[a,x] + δ)
    end
    @objective(model, Min, δ)
    optimize!(model)
    return [value.(M_vars[a]) for a in 1:num_a], value(δ)
end

# Quantum Measurement Estimation (Dual) :
function QME_dual(ρs::Vector, Probs::Matrix)
    num_a, num_x = size(Probs)
    d = size(ρs[1], 1)
    @assert d == num_a && num_x == length(ρs)

    model = Model(SOLVER); set_silent(model)
    @variable(model, v[1:num_a, 1:num_x] >= 0)
    @variable(model, u[1:num_a, 1:num_x] >= 0)
    @variable(model, Y[1:d, 1:d], Hermitian)

    for a in 1:num_a
        mat = [Y[i,j] + sum((v[a,x] - u[a,x]) * ρs[x][i,j] for x in 1:num_x)
               for i in 1:d, j in 1:d]
        @constraint(model, -mat in HermitianPSDCone())
    end
    @constraint(model, sum(u + v) == 1)

    @objective(model, Max,
        sum((v[a,x] - u[a,x]) * Probs[a,x] for a in 1:num_a, x in 1:num_x) +
        real(tr(Y))
    )
    optimize!(model)
    return value.(u), value.(v), value.(Y)
end

# Minimum-Error Quantum State Discrimination : Find  POVMs that discriminates the input states better
function ME_QSD(ρs::Vector, probs::Vector)
    @assert length(ρs) == length(probs)
    N, d = length(ρs), size(ρs[1], 1)

    model = Model(SOLVER); set_silent(model)
    M_vars = [@variable(model, [1:d, 1:d] in HermitianPSDCone()) for _ in 1:N]
    @constraint(model, sum(M_vars) .== Matrix{ComplexF64}(I, d, d))
    @objective(model, Max,
        sum(probs[i] * real(tr(M_vars[i] * ρs[i])) for i in 1:N)
    )
    optimize!(model)
    return objective_value(model), [value.(M_vars[i]) for i in 1:N]
end

# Generalized Robustness of Measurement Informativeness : Quantifier of the information that a certain POVM provides
function GRMI(Measurements::Vector)
    o = length(Measurements)
    d = size(Measurements[1], 1)

    model = Model(SOLVER); set_silent(model)
    Y_vars = [@variable(model, [1:d, 1:d] in HermitianPSDCone()) for _ in 1:o]
    @variable(model, z[1:o] >= 0)

    for i in 1:o
        @constraint(model, Measurements[i] + Y_vars[i] .==
                           z[i] * Matrix{ComplexF64}(I, d, d))
    end
    @objective(model, Min, sum(z))
    optimize!(model)
    return objective_value(model) - 1
end

# Calculate the negativity of entanglement and find an entanglement witness
function entanglement_witness(ρ::AbstractMatrix, dA::Int, dB::Int)
    d = dA * dB
    @assert size(ρ) == (d, d)

    model = Model(SOLVER); set_silent(model)
    @variable(model, W[1:d, 1:d], Hermitian)

    # Partial transpose over A
    W_TA = partial_transpose_A(W, dA, dB)
    I_d  = Matrix{ComplexF64}(I, d, d)

    @constraint(model, I_d - W_TA in HermitianPSDCone())
    @constraint(model, W_TA       in HermitianPSDCone())
    @objective(model, Max, -real(tr(W * ρ)))
    optimize!(model)
    return objective_value(model), value.(W)
end

# Random Robustness of Entanglement relaxed by PPT criterion
function RRE_PPT(ρ_AB::AbstractMatrix, dA::Int, dB::Int)
    d = dA * dB
    @assert size(ρ_AB) == (d, d)

    model = Model(SOLVER); set_silent(model)
    @variable(model, r >= 0)

    X    = ρ_AB + r * Matrix{ComplexF64}(I, d, d) / (dA * dB)
    X_TA = partial_transpose_A(X, dA, dB)
    @constraint(model, Hermitian(X_TA) in HermitianPSDCone())
    @objective(model, Min, r)
    optimize!(model)
    return objective_value(model)
end

# Random Robustess of Entanglement relaxed by k-symmetric extension over B
function RRE_k(ρ_AB::AbstractMatrix, dA::Int, dB::Int, k::Int)
    d = dA * dB^k

    model = Model(SOLVER); set_silent(model)
    @variable(model, r >= 0)
    @variable(model, ρ_ext[1:d, 1:d] in HermitianPSDCone())

    ρ_AB1    = partial_trace_Bsystems(ρ_ext, dA, dB, k)
    ρ_noised = ρ_AB + r * Matrix{ComplexF64}(I, dA*dB, dA*dB) / (dA * dB)

    @constraint(model, ρ_noised .== ρ_AB1)
    @constraint(model, real(tr(ρ_ext)) == 1 + r)

    for j in 2:k
        P = permutation_operator(dA, dB, k, 1, j)
        @constraint(model, ρ_ext .== P * ρ_ext * P')
    end

    @objective(model, Min, r)
    optimize!(model)
    r_val = value(r)
    return r_val, value.(ρ_ext) / (1 + r_val)
end

# Random Robustess of Entanglement relaxed by PPT criterion and k-symmetric extension over B
function RRE_PPT_k(ρ_AB::AbstractMatrix, dA::Int, dB::Int, k::Int)
    d    = dA * dB^k
    dims = [dA; fill(dB, k)]

    model = Model(SOLVER); set_silent(model)
    @variable(model, r >= 0)
    @variable(model, ρ_ext[1:d, 1:d] in HermitianPSDCone())

    ρ_AB1    = partial_trace_Bsystems(ρ_ext, dA, dB, k)
    ρ_noised = ρ_AB + r * Matrix{ComplexF64}(I, dA*dB, dA*dB) / (dA * dB)

    @constraint(model, ρ_noised .== ρ_AB1)
    @constraint(model, real(tr(ρ_ext)) == 1 + r)

    for j in 2:k
        P = permutation_operator(dA, dB, k, 1, j)
        @constraint(model, ρ_ext .== P * ρ_ext * P')
    end

    # Partial transpose over B₁
    ρ_ext_TB1 = partial_transpose_sys(ρ_ext, dims, 1)
    @constraint(model, Hermitian(ρ_ext_TB1) in HermitianPSDCone())

    @objective(model, Min, r)
    optimize!(model)
    r_val = value(r)
    return r_val, value.(ρ_ext) / (1 + r_val)
end

# Random Robustness of Measurement Incompatibility
function RRMI(M_list::Vector{<:Vector})
    m         = length(M_list)
    o         = length(M_list[1])
    d         = size(M_list[1][1], 1)
    num_parent = o^m

    model = Model(SOLVER); set_silent(model)
    N_vars = [@variable(model, [1:d, 1:d] in HermitianPSDCone()) for _ in 1:num_parent]
    @variable(model, r >= 0)

    # outcome(k, x): outcome of the x instrument for the k outcome set
    outcome(k, x) = (k ÷ o^(m - x - 1)) % o

    for x in 0:m-1, a in 0:o-1
        sum_terms = sum(N_vars[k+1] for k in 0:num_parent-1 if outcome(k, x) == a)
        @constraint(model,
            M_list[x+1][a+1] + r * Matrix{ComplexF64}(I, d, d) * tr(M_list[x+1][a+1]) / o .== sum_terms)
    end
    @constraint(model,
        sum(N_vars) .== (1 + r) * Matrix{ComplexF64}(I, d, d))

    @objective(model, Min, r)
    optimize!(model)

    return value(r)
end

# Noise Robustness
function NoiseRobustness(M_list::Vector{<:Vector})
    m          = length(M_list)
    o          = length(M_list[1])
    d          = size(M_list[1][1], 1)
    num_parent = o^m

    model = Model(SOLVER); set_silent(model)
    G_vars = [@variable(model, [1:d, 1:d] in HermitianPSDCone()) for _ in 1:num_parent]
    
    @variable(model, n >= 0)                    
    @constraint(model, n <= 1)                 

    outcome(k, x) = (k ÷ o^(m - x - 1)) % o
    @constraint(model, sum(G_vars) .== Matrix{ComplexF64}(I, d, d))

    for x in 0:m-1, a in 0:o-1
        sum_terms = sum(G_vars[k+1] for k in 0:num_parent-1 if outcome(k, x) == a)
        @constraint(model,
            n * M_list[x+1][a+1] + (1 - n) * tr(M_list[x+1][a+1]) * 
            Matrix{ComplexF64}(I, d, d) / d .== sum_terms)
    end

    @objective(model, Max, n)
    optimize!(model)

    return value(n)
end

# Dual Noise Robustness
function NoiseRobustness_dual(M_list::Vector{<:Vector})
    m = length(M_list)
    o = length(M_list[1])
    d = size(M_list[1][1], 1)
    num_parent = o^m

    model = Model(SOLVER); set_silent(model)

    X_vars = [[(@variable(model, [1:d, 1:d], Hermitian)) for a in 1:o] for x in 1:m]

    outcome(k, x) = (k ÷ o^(m - x - 1)) % o

    lhs = 1 + sum(real(tr(X_vars[x][a] * M_list[x][a])) for x in 1:m, a in 1:o)
    rhs = (1/d) * sum(real(tr(M_list[x][a])) * real(tr(X_vars[x][a])) for x in 1:m, a in 1:o)
    @constraint(model, lhs >= rhs)

    for k in 0:num_parent-1
        sum_terms = sum(X_vars[x+1][outcome(k, x)+1] for x in 0:m-1)
        @constraint(model, sum_terms in HermitianPSDCone())
    end

    @objective(model, Min, lhs)
    optimize!(model)

    status = termination_status(model)
    if status ∉ (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        @warn "NoiseRobustness_dual: solver status $status"
    end

    return objective_value(model)
end

#= # Random Robustness of Measurement Incompatibility (Stable implementation)
function RRMI(M_list::Vector{<:Vector})
    m          = length(M_list)
    o          = length(M_list[1])
    d          = size(M_list[1][1], 1)
    num_parent = o^m

    M_clean = map(M_list) do M
        S    = Hermitian(sum(M))
        corr = (S - I(d)) / length(M)
        [Hermitian(m_ax - corr) for m_ax in M]
    end

    model = Model(Clarabel.Optimizer)
    set_silent(model)
    set_attribute(model, "tol_gap_abs", 1e-9)
    set_attribute(model, "tol_gap_rel", 1e-9)
    set_attribute(model, "tol_feas",    1e-9)
    set_attribute(model, "max_iter",    200000)

    @variable(model, r >= 0)

    sym_idx  = [(i,j) for i in 1:d for j in i:d]
    asym_idx = [(i,j) for i in 1:d for j in i+1:d]
    n_sym    = length(sym_idx)
    n_asym   = length(asym_idx)

    av = [@variable(model, [1:n_sym])  for _ in 1:num_parent]
    bv = [@variable(model, [1:n_asym]) for _ in 1:num_parent]

    for λ in 1:num_parent
        R = [AffExpr(0) for _ in 1:2d, _ in 1:2d]

        for (k,(i,j)) in enumerate(sym_idx)
            if i == j
                R[i,    i   ] = av[λ][k]
                R[d+i,  d+i ] = av[λ][k]
            else
                R[i,  j  ] = av[λ][k];   R[j,  i  ] = av[λ][k]
                R[d+i,d+j] = av[λ][k];   R[d+j,d+i] = av[λ][k]
            end
        end

        for (k,(i,j)) in enumerate(asym_idx)
            R[i,   d+j] = -bv[λ][k];   R[j,   d+i] =  bv[λ][k]
            R[d+i, j  ] =  bv[λ][k];   R[d+j, i  ] = -bv[λ][k]
        end

        @constraint(model, R in PSDCone())
    end

    outcome(λ0, x) = (λ0 ÷ o^(m - x - 1)) % o

    for (k,(i,j)) in enumerate(sym_idx)
        s = sum(av[λ][k] for λ in 1:num_parent)
        if i == j
            @constraint(model, s == 1 + r)
        else
            @constraint(model, s == 0)
        end
    end
    for k in 1:n_asym
        @constraint(model, sum(bv[λ][k] for λ in 1:num_parent) == 0)
    end

    for x in 0:m-1, a_out in 0:o-1
        M_ax  = M_clean[x+1][a_out+1]
        ReM   = real.(Matrix(M_ax))
        ImM   = imag.(Matrix(M_ax))
        group = [λ for λ in 1:num_parent if outcome(λ-1, x) == a_out]

        for (k,(i,j)) in enumerate(sym_idx)
            s = sum(av[λ][k] for λ in group)
            if i == j
                @constraint(model, s == ReM[i,i] + r/o)
            else
                @constraint(model, s == ReM[i,j])
            end
        end
        for (k,(i,j)) in enumerate(asym_idx)
            @constraint(model, sum(bv[λ][k] for λ in group) == ImM[i,j])
        end
    end

    @objective(model, Min, r)
    optimize!(model)

    status = termination_status(model)
    if status ∉ (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL)
        @warn "RRMI: solver status $status"
    end
    return value(r)
end =#


# Auxiliary function for JuMP to calculate P(A_a = B_b + k) = ∑_{j=0}^{d-1} Tr(ρ Π_{j|a} ⊗ Π_{j+k mod d|b})
function prob_juMP_expr(ρ, Proj_Aa::Vector, Proj_Bb::Vector, k::Int, d::Int)
    n = d * d
    K = zeros(ComplexF64, n, n)
    for j in 1:d
        idx_B = mod1(j + k, d)
        K += kron(Proj_Aa[j], Proj_Bb[idx_B])
    end
    
    return real(tr(ρ * K))
end

# Semidefinite program to maximize the CGLMP expression with respect the quantum state for a fixed measurement
function CGLMP_opt(Proj_A::Vector, Proj_B::Vector, d::Int)
    model = Model(SOLVER); set_silent(model)
    
    n = d * d
    @variable(model, ρ[1:n, 1:n] in HermitianPSDCone())
    @constraint(model, real(tr(ρ)) == 1)
    
    # CGLMP expression directly with JuMP variables
    Id = 0.0
    for k in 0:(div(d,2) - 1)
        weight = 1 - 2*k/(d-1)
        
        # Positive terms
        pos_expr = prob_juMP_expr(ρ, Proj_A[1], Proj_B[1], k, d)
        pos_expr += prob_juMP_expr(ρ, Proj_A[2], Proj_B[1], -k-1, d)
        pos_expr += prob_juMP_expr(ρ, Proj_A[2], Proj_B[2], k, d)
        pos_expr += prob_juMP_expr(ρ, Proj_A[1], Proj_B[2], -k, d)
        
        # Negative terms
        neg_expr = prob_juMP_expr(ρ, Proj_A[1], Proj_B[1], -k-1, d)
        neg_expr += prob_juMP_expr(ρ, Proj_A[2], Proj_B[1], k, d)
        neg_expr += prob_juMP_expr(ρ, Proj_A[2], Proj_B[2], -k-1, d)
        neg_expr += prob_juMP_expr(ρ, Proj_A[1], Proj_B[2], k+1, d)
        
        Id += weight * (pos_expr - neg_expr)
    end
    
    @objective(model, Max, Id)
    optimize!(model)
    
    return objective_value(model), value.(ρ)
end

# Optimize the CGLMP inequality fixing a tunning measurement and save data
function CGLMP_analysis(d::Int)
    Alphas = LinRange(0, 1, 100)
    n = length(Alphas)
    Id_array   = zeros(Float64, n)
    Entanglement = zeros(Float64, n)
    NoiseDual  = zeros(Float64, n)

    Threads.@threads for i in 1:n
        α = Alphas[i]
        M = tunning_M(d, α)

        Proj_A1 = [begin
            A1j = zeros(ComplexF64, d, d)
            A1j[j, j] = 1.0
            A1j
        end for j in 1:d]

        Proj_A2 = [begin
            ket = zeros(ComplexF64, d)
            ket[j] = 1.0
            ψ = M * ket
            ψ = ψ / norm(ψ)
            ψ * ψ'
        end for j in 1:d]

        Proj_A = [Proj_A1, Proj_A2]
        Proj_B = [Proj_A1, Proj_A2]

        Id_max, ρ = CGLMP_opt(Proj_A, Proj_B, d)

        Id_array[i]    = Id_max
        r = RRE_PPT(ρ, d, d)
        Entanglement[i] = r / (1 + r)
        NoiseDual[i]   = NoiseRobustness_dual(Proj_A)

        println("Terminated process for α = $(round(α, digits=4))")
    end

    # Create the directory in case it doesn't exist
    mkpath("resultsCGLMP")

    # Save as CSV
    df = DataFrame(
        alpha         = collect(Alphas),
        CGLMP         = Id_array,
        Entanglement  = Entanglement,
        NoiseDual = NoiseDual,
    )

    path = "resultsCGLMP/cglmp_d$(d).csv"
    CSV.write(path, df)

    return df
end


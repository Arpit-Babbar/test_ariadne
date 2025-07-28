using Trixi
using ForwardDiff

using FiniteDiff, StaticArrays
using BenchmarkTools
using LinearAlgebra
using Ariadne

# @assert !Trixi._PREFERENCE_POLYESTER
@assert !Trixi._PREFERENCE_LOOPVECTORIZATION

# using SparsityDetection
using SparseArrays
# using SparseDiffTools

trixi_include(joinpath(Trixi.examples_dir(), "tree_2d_dgsem", "elixir_navierstokes_lid_driven_cavity.jl"), initial_refinement_level = 4, sol = nothing)

F!(du, u, p) = Trixi.rhs!(du, u, p, 0.0)


uin = copy(ode.u0)
uout = similar(uin)
rand_J = rand(length(uin), length(uin))

J = Ariadne.JacobianOperator(F!, uout, uin, semi)
J_fast = Ariadne.JacobianOperator(F!, uout, uin, semi, assume_p_const = true)
@time Ariadne.mul!(uout, J, uin)
@time mul!(uout, rand_J, uin)

du = copy(uout)
Trixi.rhs!(du, uin, ode.p, 0.0)
res = copy(du)
Δt = 0.1
function axby!(res, a, x, b, y)
    res .= a .* x .+ b .* y
end
@btime axby!($res, 1.0, $uin, $Δt, $du)
kc = Ariadne.KrylovConstructor(res)
workspace = Ariadne.krylov_workspace(:gmres, kc)
struct LMOperator{JOp}
   J::JOp
    dt::Float64
end
Base.size(M::LMOperator) = size(M.J)
Base.eltype(M::LMOperator) = eltype(M.J)
Base.length(M::LMOperator) = length(M.J)

import LinearAlgebra: mul!
function mul!(out::AbstractVector, M::LMOperator, v::AbstractVector)
    # out = (I/dt - J(f,x,p)) * v
    mul!(out, M.J, v)
    @. out = v - out * M.dt
    return nothing
end


M = LMOperator(J, Δt)
Ariadne.krylov_solve!(workspace, M, res)


@time mul!(uout, J, uin);
@time mul!(uout, J_fast, uin);
@time Ariadne.krylov_solve!(workspace, M, res);
@btime Ariadne.krylov_solve!($workspace, $M, $res);

M_fast = LMOperator(J_fast, Δt);
Trixi.rhs!(du, uin, ode.p, 0.0);
res_fast = copy(du);
kc_fast = Ariadne.KrylovConstructor(res_fast);
workspace_fast = Ariadne.krylov_workspace(:gmres, kc_fast);
@time Ariadne.krylov_solve!(workspace_fast, M_fast, res_fast);
@btime Ariadne.krylov_solve!($workspace_fast, $M_fast, $res_fast);

# @btime mul!($uout, $J, $uin);
# @btime Ariadne.krylov_solve!($workspace, $M, $res);

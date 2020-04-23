using Test
using Random
using Flux: onehotbatch, Dense, softmax, Chain, ADAM, params, train!, mse, gradient, Descent

abstract type ExprAST end

struct EInt <: ExprAST
    val :: Int
end

struct ENeg <: ExprAST
    expr :: ExprAST
end

struct EAdd <: ExprAST
    left :: ExprAST
    right :: ExprAST
end

interpret(expr::EInt) = expr.val
interpret(expr::ENeg) = - interpret(expr.val)
interpret(expr::EAdd) = interpret(expr.left) + interpret(expr.right)

@test interpret(EAdd(EInt(10), EInt(20))) == 30

num = 10
data = round.(Int, (rand(Float64, 10, 2) * 10))
inputraw = mapslices(((x,y),) -> (x,y), data, dims = [2])
inputexpressions = mapslices(((x,y),) -> EAdd(EInt(x), EInt(y)), data, dims = [2])
outputraw = float.(mapslices(((x,y),) -> x + y, data, dims = [2]))

@test interpret.(inputexpressions) == outputraw

features((l,r)) = float.((l+r, [l,r]))
features(a::AbstractArray) = hcat(features.(a)...)

X = (features(inputraw))
y = outputraw'
m = Chain(Dense(2, 10), Dense(10, 1), softmax)

function loss(out, inp)
    return mse(m(inp), out)
end

#loss(x, y) = (m(x) - y)^2
opt = Descent(.1)

# helper monitor function
function monitor(e)
    l = 0
    for x in X
        l += mse(x[1], x[2])
    end

    println("epoch $(lpad(e, 4)): loss = $(round(l; digits=4))")
    #@show (m∘float).([1 2; 3 4; 100 100])
end


# training
for e in 0:1000
    train!(loss, params(m), X, opt)
    if e % 50 == 0; monitor(e) end
end



#println(a)
#println("test")
#println(a[1,:])

using Revise
using Test
using Random
using Flux: onehotbatch, Dense, softmax, Chain, ADAM, params, train!, mse, gradient, Descent, Momentum
using Plots

function tr()
    num = 100

    data = round.(Int, (rand(Float64, num, 2) * 10))

    inputraw = mapslices(((x,y),) -> (x,y), data, dims = [2])
    outputraw = float.(mapslices(((x,y),) -> x + y, data, dims = [2]))

    features((l,r)) = float.(( [l,r],l*r))
    features(a::AbstractArray) = hcat(features.(a)...)

    X = (features(inputraw))
    y = outputraw'
    m = Chain(Dense(2, 1))
    data = []
    function lossb(inp, out)
        mm = m(inp)
        outt = [out]
        a = (mm .- outt).^2
        return sum(a)
    end

    #loss(x, y) = (m(x) - y)^2
    opt = Descent(.01)
    @show opt

    # helper monitor function
    function monitor(e)
        l = 0
        for x in X
            l += lossb(x[1], x[2])
        end
        @show m([3.0, 3.0])
        @show gradient(lossb, [3.0, 3.0], 9.0)
        push!(data, l)

        println("epoch $(lpad(e, 6)): loss = $(round(l; digits=4))")
    end

    ps = params(m)

    # training
    for e in 0:3000
        train!(lossb, ps, X, opt)
        if e % 50 == 0; monitor(e) end
    end
    plot(data)
end
tr()

### testing reflect method
using Test
# testing reflection on batchsize 0
d = 10
X0 = fill(0.0f0,d)  
X1 = X0 + randn(d)
X11 = _reflect(X0,X1,-1,1)
@test prod(X11 .< 1) && prod(X11 .> -1) 

# testing reflection on batchsize 
batch_size = 10000
y0 = repeat(X0[:],1,batch_size)
y1 = y0 + 2 .* randn(size(y0))
y0 = hcat(X0,y0)
y1 = hcat(X1,y1)
n = similar(y1)
y11 = _reflect_GPU2(y0,y1,-1,1,d,batch_size+1,n)
@test prod(y11 .< 1) && prod(y11 .> -1) 
@test prod(y11[:,1] .≈ X11[:,1]) 




# testing reflection on batchsize 
batch_size = 100000
y0 = repeat(X0[:],1,batch_size)
y1 = repeat(X0[:],1,batch_size)
n = similar(y1)
dWall = zeros(Float32, d, batch_size, d )
randn!(dWall)
dW = @view dWall[:,:,1]
for i in 1:21
    y0 .= y1
    y1 .= y0 .+ μ(y0,Dict(),0.1f0) .* dt .+ σ(y0,Dict(),0.1f0) .* sqrt(0.1f0) .* dW
    y1 .= _reflect_GPU2(y0,y1,-1f0,1f0,d,batch_size,n)
end
count(isnan.(y1))
### testing reflect method
using Test
# testing reflection on batchsize 0
d = 10
X0 = fill(0.0f0,d)  
X1 = X0 + randn(d)
X11 = _reflect(X0,X1,0,1)
@test prod(X11 .< 1) && prod(X11 .> 0) 

# testing reflection on batchsize 
batch_size = 1000
y0 = repeat(X0[:],1,batch_size)
y1 = y0 + randn(size(y0))
y0 = hcat(X0,y0)
y1 = hcat(X1,y1)
n = similar(y1)
y11 = _reflect_GPU2(y0,y1,0,1,d,batch_size+1,n)
@test prod(y11 .< 1) && prod(y11 .> 0) 
@test prod(y11[:,1] .≈ X11[:,1]) 
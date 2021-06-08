### testing reflect method
using Test
# testing reflection on batchsize 0
X0 = fill(0.0f0,10)  
X1 = X0 + randn(d)
X11 = _reflect(X0,X1,0,1)
@test prod(X11 .< 1) && prod(X11 .> 0) 

# testing reflection on batchsize 
y0 = repeat(X0[:],1,batch_size)
y1 = y0 + randn(size(y0))
y11 = _reflect(y0,y1,0,1)
@test prod(y11 .< 1) && prod(y11 .> 0) 
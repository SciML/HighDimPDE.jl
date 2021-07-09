# DeepSplitting
- [ ] export the time steps
- [ ] On should try out to keep the same parameters for the next time step for the neural network
- [ ] Try to find an architecture that fits well g(x) for d = 5 or may be more, d = 10. for this, put tspan[2] = 1f-1

## Samedi 03/06
- [x] check when you extract x in DeepSplitting regarding changes in PIDEProblem, now x can be whether u_domain or x
- [x] the errors raised in DeepSplitting.jl l81 can be displace in constructor of PIDEProblem

# Consistency
- [ ] change _reflect method to accomodate for s, e::Array of same size as a, b

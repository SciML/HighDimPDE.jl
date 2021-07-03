# DeepSplitting
- [ ] Make sure that you propose several options
    - [x] when u_domain is not nothing, then you return the nn
    - [x] When u_domain is nothing, you only return the value assessed at X0
> u_domain is the domain over which the samples for non local term should be drawn
> make sure that this is coherent
> maybe you want to use your mc_sample method to sample initial points as well

    - [ ] when neumann is specified, you reflect at neumann[1], neumann[2]

- [x] make sure the x0 argument is consistent

- [ ] Try to calculate problem for Hamel, possibly modifying the sampling (uniform)

## Samedi 03/06
- [ ] check when you extract x in DeepSplitting regarding changes in PIDEProblem, now x can be whether u_domain or x
- [ ] the errors raised in DeepSplitting.jl l81 can be displace in constructor of PIDEProblem

# Consistency
- [ ] change _reflect method to accomodate for s, e::Array of same size as a, b
# DeepSplitting
- [ ] export the time steps
- [ ] On should try out to keep the same parameters for the next time step for the neural network
- [ ] Try to find an architecture that fits well g(x) for d = 5 or may be more, d = 10. for this, put tspan[2] = 1f-1
- [ ] use fastchain instead of chain, from diffeqflux
    - [ ] this demands quite a big change, since `Flux.params(nn::FastChain)` returns nothing (`FastDense` layers do not store parameters)
- [ ] integrate `NNPDENS` and `NNPDEHan`
- [ ] Propose both TerminalPDEProblem and InitialPDEProblem

# Consistency
- [ ] change _reflect method to accomodate for s, e::Array of same size as a, b
- [ ] merge chiara's proposition
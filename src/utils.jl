using DiffEqBase
abstract type NeuralPDEAlgorithm <: DiffEqBase.AbstractODEAlgorithm end
using Flux, Zygote, LinearAlgebra
using ProgressMeter: @showprogress
using CUDA
#!/usr/bin/env julia
include("harness_receipts.jl")
using .HarnessReceipts

function main(arguments=ARGS)
    length(arguments) in (2,3) || error("usage: full_test_gate.jl PACKAGE CODES_ROOT [OUTPUT_DIRECTORY]")
    owner,codes=arguments[1:2]
    output=length(arguments)==3 ? abspath(arguments[3]) : joinpath(@__DIR__,"results","development-harness","tests")
    path=run_full_package_test(owner,codes;output,threads=4)
    println("Full package test passed; receipt: ",path)
end

abspath(PROGRAM_FILE)==abspath(@__FILE__) && main()

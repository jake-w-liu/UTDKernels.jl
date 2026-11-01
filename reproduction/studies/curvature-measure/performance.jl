# Optional warm performance controls; scientific datasets remain generate.jl.
using UTDKernels, DiffUTD, StaticArrays, LinearAlgebra, Statistics, Printf

function measure(label,f;repetitions=500)
    f()
    samples=Float64[]
    for batch=1:7
        elapsed=@elapsed begin
            for repetition=1:repetitions
                f()
            end
        end
        push!(samples,elapsed/repetitions)
    end
    bytes=@allocated f()
    @printf("%s,%.9e,%.9e,%.9e,%d\n",label,median(samples),minimum(samples),maximum(samples),bytes)
end

function main()
    axis=SVector(0.,0.,1.);q=SVector(0.,0.,0.);turn=.13
    edge=StraightEdge{Float64}(q,axis,SVector(cos(turn),sin(turn),0.),SVector(1.,0.,0.),Inf,Wedge(pi+turn))
    observer=SVector(3.,0.,0.)
    soft=CylindricalLineSource(observer,axis,8.;polarization=:soft)
    hard=CylindricalLineSource(observer,axis,8.;polarization=:hard)
    function unshared()
        return (cylindrical_line_components(soft,observer,edge;coincident=true,infinite_axis=true),
                cylindrical_line_components(hard,observer,edge;coincident=true,infinite_axis=true))
    end
    function shared()
        geometry=DiffUTD._cylindrical_line_geometry(soft,observer,edge;coincident=true)
        return (DiffUTD._cylindrical_line_transport(soft,geometry),
                DiffUTD._cylindrical_line_transport(hard,geometry))
    end
    separate=unshared();together=shared()
    for (a,b) in zip(separate,together)
        isapprox(a.full.electric,b.full.electric;rtol=2e-14,atol=2e-14) || error("shared electric transport changed")
        isapprox(a.full.magnetic,b.full.magnetic;rtol=2e-14,atol=2e-14) || error("shared magnetic transport changed")
    end
    println("# Julia=",VERSION," threads=",Threads.nthreads()," cpu=",first(Sys.cpu_info()).model)
    println("# Warmed once; seven batches of 500 calls; source geometry and all returned channels identical")
    println("case,seconds_median,seconds_min,seconds_max,allocated_bytes")
    measure("separate_geometry_two_polarizations",unshared)
    measure("shared_geometry_two_polarizations",shared)
end

if abspath(PROGRAM_FILE)==@__FILE__
    main()
end

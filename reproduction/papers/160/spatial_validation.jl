# Full-population spatial reductions. This receipt is separate from the 17
# manuscript CSVs and does not relabel L=Inf as a finite-edge far field.
using UTDKernels,DiffUTD,StaticArrays,LinearAlgebra,Random,TOML
include(joinpath(@__DIR__,"hp_oracle.jl"))
using .PaperReproductionPaths: output_directory

function numeric_csv(path)
    lines=filter(line->!isempty(line)&&!startswith(line,"#"),readlines(path))
    header=split(first(lines),',')
    rows=[parse.(Float64,split(line,',')) for line in lines[2:end]]
    return header,rows
end

function spatial_context(g,h)
    q=SVector(0.,0.,0.);axis=SVector(0.,0.,1.)
    direction(t)=SVector(cos(t),sin(t),0.)
    edge=StraightEdge{Float64}(q,axis,-direction(g.alpha),SVector(1.,0.,0.),Inf,Wedge(g.alpha))
    wave=PlaneWave(-direction(h),axis,1.,g.k)
    return (;edge,wave,axis,source=direction(h),outgoing=direction(g.phi))
end

function validate_population(data;far_field=false)
    filename=far_field ? "ensemble_farfield.csv" : "ensemble.csv"
    _,rows=numeric_csv(joinpath(data,filename))
    expected=far_field ? 400 : 800
    length(rows)==expected || error("incomplete physical population in $filename")
    rng=Xoshiro(far_field ? 20260823 : 20260816)
    count=0;max_encoding=0.;max_nominal=0.;max_field_coefficient=0.;max_field_phase=0.;max_dyad_covariance=0.
    a=normalize(SVector(.3,-.7,.2));t=.73
    ax=SMatrix{3,3}(0.,a[3],-a[2],-a[3],0.,a[1],a[2],-a[1],0.)
    R=cos(t)*I+(1-cos(t))*(a*a')+sin(t)*ax
    shift=SVector(.37,-.26,.81);heldout=0
    while count<expected
        sample,status,_=sample_geometry(rng;far_field)
        status===:accepted || continue
        g,h,tm,xm=sample;count+=1;row=rows[count]
        (row[2]==g.alpha && row[3]==g.phi && row[5]==h && row[4]==g.k*g.L) ||
            error("replayed population disagrees with $filename row $count")
        context=spatial_context(g,h)
        angles=RayAngles(g.phi,h)
        evaluation=grazing_ray_coefficients(context.edge,context.source,context.outgoing,g.k,g.L;
            method=:fixed,soft_only=true,local_angles=angles,allow_interior=true,
            allow_infinite_L=true,transition_margin=tm,x_margin=xm)
        stored=complex(row[15],row[16])
        nominal=abs(evaluation.Ds-stored)/max(abs(stored),1e-300)
        max_nominal=max(max_nominal,nominal)
        nominal<=3e-14 || error("3D dyad changed stored coefficient at $filename row $count")
        max_encoding=max(max_encoding,maximum(abs,evaluation.encoding_error))
        if far_field
            isinf(evaluation.transition_distance) && !hasproperty(evaluation,:distance) ||
                error("infinite-distance local coefficient was mislabeled as finite propagation")
        else
            result=grazing_ray_field(context.wave,g.L*context.outgoing,context.edge;
                method=:fixed,soft_only=true,local_angles=angles,allow_interior=true,
                transition_margin=tm,x_margin=xm)
            # Score the actual represented physical transition distance,
            # separately from the nominal L in the coefficient population.
            physical=Geom(g.alpha,g.phi,g.k,result.coefficients.transition_distance)
            reference=soft_hp(physical,h;dps=60)
            phase=cis(-g.k*result.distance)
            recovered=dot(context.axis,result.field.electric)*sqrt(result.distance)/phase
            coefficient_error=rel(recovered,reference)
            max_field_coefficient=max(max_field_coefficient,coefficient_error)
            coefficient_error<1e-10 || error("physical field coefficient reduction failed in row $count")
            # Propagation is assessed separately against exact binary64 inputs.
            phase_error=setprecision(BigFloat,256) do
                exact=cis(-BigFloat(g.k)*BigFloat(result.distance))
                Float64(abs(Complex{BigFloat}(phase)-exact))
            end
            max_field_phase=max(max_field_phase,phase_error)
            phase_error<=4eps(abs(g.k*result.distance))+4eps(Float64) ||
                error("outgoing phase exceeded its floating-point argument budget")
        end
        if count%31==0
            transformed=StraightEdge{Float64}(shift,R*context.edge.tangent,
                R*context.edge.face_n,R*context.edge.face_o,Inf,context.edge.wedge)
            rotated=grazing_ray_coefficients(transformed,R*context.source,R*context.outgoing,g.k,g.L;
                method=:fixed,soft_only=true,local_angles=angles,allow_interior=true,
                allow_infinite_L=true,transition_margin=tm,x_margin=xm)
            covariance_error=norm(rotated.soft_vector-R*evaluation.soft_vector)/max(norm(evaluation.soft_vector),1e-300)
            max_dyad_covariance=max(max_dyad_covariance,covariance_error)
            covariance_error<3e-12 || error("rotated local soft dyad failed covariance")
            heldout+=1
        end
    end
    return Dict("cases"=>count,"cartesian_encoding_error_max_rad"=>max_encoding,
        "stored_coefficient_reduction_relative_max"=>max_nominal,
        "finite_field_dephased_coefficient_error_max"=>max_field_coefficient,
        "propagation_phase_error_max"=>max_field_phase,"heldout_rotations"=>heldout,
        "soft_dyad_rotation_relative_max"=>max_dyad_covariance,
        "scope"=>far_field ? "local F-to-one dyad; no finite-distance/far-field equivalence inferred" :
                             "physical plane-wave L=s at normal incidence; independent coefficient reference at actual represented L")
end

function main()
    paper=output_directory(160);data=joinpath(paper,"data")
    receipt=Dict("julia"=>string(VERSION),"utdkernels"=>string(pkgversion(UTDKernels)),
        "diffutd"=>string(pkgversion(DiffUTD)),"finite"=>validate_population(data),
        "infinite"=>validate_population(data;far_field=true))
    destination=joinpath(paper,"reproduction-logs","spatial-validation.toml")
    mkpath(dirname(destination))
    open(destination,"w") do io;TOML.print(io,receipt);end
    println(receipt)
end

if abspath(PROGRAM_FILE)==@__FILE__
    main()
end

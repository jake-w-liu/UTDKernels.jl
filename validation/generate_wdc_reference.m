function generate_wdc_reference()
% GENERATE_WDC_REFERENCE  Generate reference diffraction coefficient data
% from Balanis Chapter 13 WDC.m for validation of UTDKernels.jl.
%
% Convention:
%   WDC(R, phi_deg, phip_deg, beta0_deg, n)
%     R      = distance parameter in wavelengths (kL = 2*pi*R)
%     phi    = observation angle (degrees)
%     phip   = incidence angle (degrees)
%     beta0  = oblique incidence angle (degrees), 90 = normal
%     n      = wedge factor (exterior angle / pi)
%
% Output: CSV with columns
%   n, R, phi_deg, phip_deg, Re_Ds, Im_Ds, Re_Dh, Im_Dh, Re_Di, Im_Di, Re_Dr, Im_Dr

outdir = fullfile(fileparts(mfilename('fullpath')), 'data');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

BTD = 90.0;  % Normal incidence to the edge

% =====================================================================
% Test categories
% =====================================================================

% --- Category 1: Comprehensive sweep ---
wedge_ns   = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0];
R_values   = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0];
phip_all   = [15, 30, 45, 60, 90, 120, 150];

nrows = 0;
fid = fopen(fullfile(outdir, 'wdc_reference.csv'), 'w');
fprintf(fid, 'n,R,phi_deg,phip_deg,Re_Ds,Im_Ds,Re_Dh,Im_Dh,Re_Di,Im_Di,Re_Dr,Im_Dr\n');

for ni = 1:length(wedge_ns)
    n = wedge_ns(ni);
    max_angle = n * 180.0;  % maximum valid angle (degrees)

    for ri = 1:length(R_values)
        R = R_values(ri);

        for pi_ = 1:length(phip_all)
            phip = phip_all(pi_);
            if phip >= max_angle - 1.0
                continue;  % skip invalid phip for this wedge
            end

            % Coarse sweep (every 2 degrees)
            phis = 1:2:(max_angle - 1);

            % Add near-boundary points (ISB and RSB)
            isb = 180 + phip;
            rsb = 180 - phip;
            offsets = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0];
            for oi = 1:length(offsets)
                off = offsets(oi);
                for boundary = [isb, rsb]
                    pt1 = boundary - off;
                    pt2 = boundary + off;
                    if pt1 > 0.1 && pt1 < max_angle - 0.1
                        phis = [phis, pt1];
                    end
                    if pt2 > 0.1 && pt2 < max_angle - 0.1
                        phis = [phis, pt2];
                    end
                end
            end

            % Add near-face points (phi near 0 and near max_angle)
            face_pts = [0.1, 0.25, 0.5, 1.0, 2.0, ...
                        max_angle - 2.0, max_angle - 1.0, ...
                        max_angle - 0.5, max_angle - 0.25, max_angle - 0.1];
            for fi = 1:length(face_pts)
                pt = face_pts(fi);
                if pt > 0.05 && pt < max_angle - 0.05
                    phis = [phis, pt];
                end
            end

            phis = unique(sort(phis));

            for phi = phis
                [Ds, Dh, Di, Dr] = WDC_nopause(R, phi, phip, BTD, n);
                fprintf(fid, '%.6f,%.6f,%.10f,%.10f,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e\n', ...
                    n, R, phi, phip, ...
                    real(Ds), imag(Ds), real(Dh), imag(Dh), ...
                    real(Di), imag(Di), real(Dr), imag(Dr));
                nrows = nrows + 1;
            end
        end
    end
end

% --- Category 2: Fine ISB/RSB crossing for half-plane ---
fprintf('Category 2: fine ISB/RSB crossing...\n');
n = 2.0;
for R = [0.1, 1.0, 10.0]
    for phip = [30, 45, 60]
        isb = 180 + phip;
        rsb = 180 - phip;
        fine_offsets = [-10, -5, -2, -1, -0.5, -0.2, -0.1, -0.05, ...
                         0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10];
        for boundary = [isb, rsb]
            for oi = 1:length(fine_offsets)
                phi = boundary + fine_offsets(oi);
                if phi > 0.05 && phi < 359.95
                    [Ds, Dh, Di, Dr] = WDC_nopause(R, phi, phip, BTD, n);
                    fprintf(fid, '%.6f,%.6f,%.10f,%.10f,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e\n', ...
                        n, R, phi, phip, ...
                        real(Ds), imag(Ds), real(Dh), imag(Dh), ...
                        real(Di), imag(Di), real(Dr), imag(Dr));
                    nrows = nrows + 1;
                end
            end
        end
    end
end

% --- Category 3: Grazing incidence (phi' small) ---
fprintf('Category 3: grazing incidence...\n');
n = 2.0;
for R = [0.1, 1.0, 10.0]
    for phip = [0.1, 0.5, 1.0, 5.0]
        for phi = [1, 5, 10, 30, 60, 90, 120, 150, 170, 175, 179, ...
                   181, 185, 190, 210, 240, 270, 300, 330, 350, 355, 359]
            if phi > 0.05 && phi < 359.95
                [Ds, Dh, Di, Dr] = WDC_nopause(R, phi, phip, BTD, n);
                fprintf(fid, '%.6f,%.6f,%.10f,%.10f,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e\n', ...
                    n, R, phi, phip, ...
                    real(Ds), imag(Ds), real(Dh), imag(Dh), ...
                    real(Di), imag(Di), real(Dr), imag(Dr));
                nrows = nrows + 1;
            end
        end
    end
end

% --- Category 4: Large kL (GTD limit) ---
fprintf('Category 4: large kL...\n');
for n = [1.5, 2.0]
    max_angle = n * 180;
    R = 1000.0;  % kL = 2000*pi
    for phip = [30, 60]
        for phi = 5:5:(max_angle - 5)
            [Ds, Dh, Di, Dr] = WDC_nopause(R, phi, phip, BTD, n);
            fprintf(fid, '%.6f,%.6f,%.10f,%.10f,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e\n', ...
                n, R, phi, phip, ...
                real(Ds), imag(Ds), real(Dh), imag(Dh), ...
                real(Di), imag(Di), real(Dr), imag(Dr));
            nrows = nrows + 1;
        end
    end
end

% --- Category 5: Reciprocity pairs ---
fprintf('Category 5: reciprocity pairs...\n');
for n = [1.5, 2.0]
    max_angle = n * 180;
    for R = [0.5, 1.0, 5.0]
        pairs = {[30, 60], [45, 90], [60, 120], [20, 100], [45, 150]};
        for pi_ = 1:length(pairs)
            pair = pairs{pi_};
            phi_a = pair(1); phi_b = pair(2);
            if phi_a < max_angle - 0.1 && phi_b < max_angle - 0.1
                % D(phi_a, phi_b) should equal D(phi_b, phi_a)
                [Ds1, Dh1, Di1, Dr1] = WDC_nopause(R, phi_a, phi_b, BTD, n);
                fprintf(fid, '%.6f,%.6f,%.10f,%.10f,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e\n', ...
                    n, R, phi_a, phi_b, ...
                    real(Ds1), imag(Ds1), real(Dh1), imag(Dh1), ...
                    real(Di1), imag(Di1), real(Dr1), imag(Dr1));
                [Ds2, Dh2, Di2, Dr2] = WDC_nopause(R, phi_b, phi_a, BTD, n);
                fprintf(fid, '%.6f,%.6f,%.10f,%.10f,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e\n', ...
                    n, R, phi_b, phi_a, ...
                    real(Ds2), imag(Ds2), real(Dh2), imag(Dh2), ...
                    real(Di2), imag(Di2), real(Dr2), imag(Dr2));
                nrows = nrows + 2;
            end
        end
    end
end

fclose(fid);
fprintf('Done. Generated %d test cases.\n', nrows);
end

% =========================================================================
% WDC without the interactive pause (from Balanis Chapter 13)
% =========================================================================
function [CDCS,CDCH,CDCI,CDCR] = WDC_nopause(R,PHID,PHIPD,BTD,FN)
D    = (0.0+1j*0.0)*zeros(1,4);
FT   = (0.0+1j*0.0)*zeros(1,4);
CT   = -0.05626977 + 1j*0.05626977;
UTPI = 0.15915494;
SML  = 0.001;
TPI  = 2*pi;
DTOR = pi/180.0;
PHR  = PHID*DTOR;
PHPR = PHIPD*DTOR;
if (abs(FN-0.5) < SML)
    CDCS = 0.0; CDCH = 0.0; CDCI = 0.0; CDCR = 0.0;
    return
end
SBO   = sin(BTD*DTOR);
DKL   = TPI*R;
UFN   = 1.0/FN;
BETAR = PHR - PHPR;
TERM  = CT/(FN*SBO);
SGN   = 1.0;
for i=1:4
    DN = UFN*(0.5*SGN + UTPI*BETAR);
    if DN < 0.0
        N = fix(DN - 0.5);
    else
        N = fix(DN + 0.5);
    end
    ANG = TPI*FN*N - BETAR;
    A   = 2.0*(cos(0.5*ANG))^2;
    X   = DKL*A;            X(abs(X) < 1.0e-7) = 0.0;
    Y   = pi + SGN*BETAR;   Y(abs(Y) < 1.0e-7) = 0.0;
    if (abs(X) < 1.0E-10)
        FT(i) = 0.0;
        if (abs(cos(0.5*Y*UFN)) >= 1.0E-3)
            if (Y < 0.0)
                FT(i) = -(1.7725 + 1j*1.7725)*abs(sqrt(DKL));
            else
                FT(i) = (1.7725 + 1j*1.7725)*abs(sqrt(DKL));
            end
            FT(i) = (FT(i) - 1j*2.0*DKL*(pi - SGN*ANG))*FN;
        end
    else
        FT(i) = FTF(X)/tan(0.5*Y*UFN);
    end
    SGN  = -SGN;
    D(i) = TERM*FT(i);
    if (SGN < 0.0)
        continue;
    end
    BETAR = PHR + PHPR;
end
CDCS = (D(1) + D(2)) - (D(3) + D(4));
CDCH = (D(1) + D(2)) + (D(3) + D(4));
CDCI = (D(1) + D(2));
CDCR = (D(3) + D(4));
end

% =========================================================================
% Fresnel transition function (from Balanis Chapter 13)
% =========================================================================
function [out] = FTF(XF)
XX  = [0.3, 0.5, 0.7, 1.0, 1.5, 2.3, 4.0, 5.5];
FX  = [(0.5729+1j*0.2677),(0.6768+1j*0.2682),(0.7439+1j*0.2549), ...
       (0.8095+1j*0.2322),(0.8730+1j*0.1982),(0.9240+1j*0.1577), ...
       (0.9658+1j*0.1073),(0.9797+1j*0.0828)];
FXX = [(0.0+1j*0.0),(0.5195+1j*0.0025),(0.3355-1j*0.0665), ...
       (0.2187-1j*0.0757),(0.127-1j*0.068),(0.0638-1j*0.0506), ...
       (0.0246-1j*0.0296),(0.0093-1j*0.0163)];
X = abs(XF);
if (X > 5.5)
    out = 1.0 + (75.0/(16.0*X*X)-0.75)/X/X + 1j*(0.5 - 15.0/(8.0*X*X))/X;
else
    if (X > 0.3)
        n = 2;
        while (X >= XX(n) && n <= 7)
            n = n + 1;
        end
        out = FXX(n)*(X-XX(n)) + FX(n);
    else
        out = ((1.253+1j*1.253)*sqrt(X) - (0.0+1j*2.0)*X - 0.6667*X*X) ...
              *exp(1j*X);
    end
end
if (XF < 0.0)
    out = conj(out);
end
end

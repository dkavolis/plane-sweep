clear;
Nimages = 5;       % total number of images to use in the algorithm
Nreference = 13;    % reference image number
Winsize = 5;        % length of window used for NCC
th = 0.5;           % NCC threshold

% image locations and formats
imloc = 'D:\Software\living_room_traj2_loop\scene_00_';
% imloc = 'D:\Software\office_room_traj2_loop\scene_';
% imloc = 'D:\Software\PlaneSweep\PlaneSweep\im';
imfmt = '.png';
imtxt = '.txt';
imdpt = '.depth';

% reference image file strings
reference = strcat(imloc, sprintf('%04d',Nreference), imtxt);
referenceim = strcat(imloc, sprintf('%04d',Nreference), imfmt);
refdpt = strcat(imloc, sprintf('%04d',Nreference), imdpt);

znear = 0.1;          % minimum depth
zfar = 5;          % maximum depth
nsteps = 250;       % number of planes used

% reference image properties
K = getcamK(reference);
[Rref, tref] = computeRT(reference);

% K = [0.709874 1-0.977786 0.493648;
%     0 0.945744 0.514782;
%     0 0 1];
% 
% K(1,:) = K(1,:)*640;
% K(2,:) = K(2,:)*480;
% 
% C(:,:,1) = [0.993701 0.110304 -0.0197854 0.280643;
% 0.0815973 -0.833193 -0.546929 -0.255355;
% -0.0768135 0.541869 -0.836945 0.810979];
% 
% C(:,:,2) = [0.993479 0.112002 -0.0213286 0.287891;
% 0.0822353 -0.83349 -0.54638 -0.255839;
% -0.0789729 0.541063 -0.837266 0.808608];
% 
% C(:,:,3) = [0.993199 0.114383 -0.0217434 0.295475;
% 0.0840021 -0.833274 -0.546442 -0.25538;
% -0.0806218 0.540899 -0.837215 0.805906];
% 
% C(:,:,4) = [0.992928 0.116793 -0.0213061 0.301659;
% 0.086304 -0.833328 -0.546001 -0.254563;
% -0.081524 0.5403 -0.837514 0.804653];
% 
% C(:,:,5) = [0.992643 0.119107 -0.0217442 0.309666;
% 0.0880017 -0.833101 -0.546075 -0.254134;
% -0.0831565 0.540144 -0.837454 0.802222];
% 
% C(:,:,6) = [0.992429 0.121049 -0.0208028 0.314892;
% 0.0901911 -0.833197 -0.545571 -0.253009;
% -0.0833736 0.539564 -0.837806 0.801559];
% 
% C(:,:,7) = [0.992226 0.122575 -0.0215154 0.32067;
% 0.0911582 -0.833552 -0.544869 -0.254142;
% -0.0847215 0.538672 -0.838245 0.799812];
% 
% C(:,:,8) = [0.992003 0.124427 -0.0211509 0.325942;
% 0.0930933 -0.834508 -0.543074 -0.254865;
% -0.0852237 0.536762 -0.839418 0.799037];
% 
% C(:,:,9) = [0.991867 0.125492 -0.021234 0.332029;
% 0.0938678 -0.833933 -0.543824 -0.252767;
% -0.0859533 0.537408 -0.838931 0.797979];
% 
% C(:,:,10) = [0.991515 0.128087 -0.0221943 0.33934;
% 0.095507 -0.833589 -0.544067 -0.250995;
% -0.0881887 0.53733 -0.838748 0.796756];
% 
% Rref = C(:, 1:3, Nreference+1);
% tref = C(:, 4, Nreference+1);

% read reference image
IMref = imread(referenceim);
ref = 255 * im2double(rgb2gray(IMref));

% calculate mean and std for each window in reference image
g = gpuArray(ones(Winsize,1) ./ Winsize);
ref = gpuArray(ref);
meanref = colfilter(colfilter(ref,g).',g).';
sqrref = ref .* ref;
meansqrref = colfilter(colfilter(sqrref,g).',g).';
varref = meansqrref - meanref .* meanref;
positive = bsxfun(@gt, varref, 0);
stdref = sqrt(positive .* varref);

% preallocate memory for sensor images
Rsens = zeros(3,3,Nimages-1);
tsens = zeros(3,1,Nimages-1);
Rrel = zeros(3,3,Nimages-1);
trel = zeros(3,1,Nimages-1);

sizeref = size(ref);
sens = zeros(sizeref(1),sizeref(2),Nimages-1);
depthmap = sens;

% read sensor images
for u = 1:Nimages-1
    nm = Nreference+((-1)^u)*u;
    sensor = strcat(imloc, sprintf('%04d',nm), imtxt);
    sensorim = strcat(imloc, sprintf('%04d',nm), imfmt);
    [Rsens(:,:,u), tsens(:,:,u)] = computeRT(sensor);
%     n = Nreference+u;
%     sensorim = strcat(imloc, sprintf('%01d',n), imfmt);
%     Rsens(:,:,u) = C(:,1:3,n+1);
%     tsens(:,:,u) = C(:,4,n+1);
%     Rrel(:,:,u) = Rsens(:,:,u) / Rref;
%     trel(:,:,u) = (tsens(:,:,u) - Rrel(:,:,u) * tref);
    Rrel(:,:,u) = Rsens(:,:,u)' * Rref;
    trel(:,:,u) = Rsens(:,:,u)' * (tref - tsens(:,:,u));
    IMsens = imread(sensorim);
    sens(:,:,u)= 255 * im2double(rgb2gray(IMsens));
end

dstep = (zfar - znear) / (nsteps - 1);

depths = znear:dstep:zfar;

depthmap = gpuArray(depthmap);
bestncc = depthmap;

% calculate depthmaps for each sensor image
parfor v = 1:Nimages-1
    [depthmap(:,:,v), bestncc(:,:,v)] = calcDepth(depths, K, Rrel(:,:,v), trel(:,:,v), ref, sens(:,:,v), Winsize, meanref, stdref);
end

% average depth results
dpm = gpuArray(zeros(sizeref(1), sizeref(2)));
N = dpm;

for v = 1:Nimages-1
    over = bsxfun(@gt, bestncc(:,:,v), th);
    dpm = dpm + over .* depthmap(:,:,v);
    N = N + over;
end

dpm = bsxfun(@rdivide, dpm, N);
dpm(isnan(dpm)) = zfar;
dpm = gather(dpm);

% plot calculated depth as gray image
figure(1);
imagesc(dpm); colormap(gray);

%plot calculated depth as 3d surface
figure(2);
h = surf(dpm);
set(h, 'LineStyle', 'none');

denoised = denoiseTVL1(dpm, 0.3, 100, zfar, znear);
figure(3);
imagesc(denoised); colormap(gray);

figure(1)

% figure(1);
[x, y, z] = compute3Dpositions(reference, strcat(imloc, sprintf('%04d',Nreference), imdpt));
sd = zeros(sizeref(1), sizeref(2));
for y = 1:10:sizeref(1)
    for x = 1:10:sizeref(2)
        sd(y,x) = z(y,x);
    end
end
% f = surf(x, y, z);
% set(f, 'LineStyle', 'none');
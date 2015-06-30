% clear;
Nimages = 10;       % total number of images to use in the algorithm
Nreference = 10;    % reference image number
Winsize = 5;        % length of window used for NCC


% image locations and formats
imloc = 'D:\Software\living_room_traj2_loop\scene_00_';
imfmt = '.png';
imtxt = '.txt';
imdpt = '.depth';

znear = 1;          % minimum depth
zfar =100;          % maximum depth
nsteps = 200;       % number of planes used

% reference image file strings
reference = strcat(imloc, sprintf('%04d',Nreference), imtxt);
referenceim = strcat(imloc, sprintf('%04d',Nreference), imfmt);

% reference image properties
K = getcamK(reference);
[Rref, tref] = computeRT(reference);

% read and normalize reference image
IMref = imread(referenceim);
ref = im2double(rgb2gray(IMref));
stdref = std2(ref);
meanref = mean2(ref);
nref = (ref - meanref) ./ stdref;

% preallocate memory for sensor images
Rsens = zeros(3,3,Nimages-1);
tsens = zeros(3,1,Nimages-1);
Rrel = zeros(3,3,Nimages-1);
trel = zeros(3,1,Nimages-1);

sizeref = size(nref);
sens = zeros(sizeref(1),sizeref(2),Nimages-1);
nsens = sens;
depthmap = sens;
stdsens = zeros(Nimages-1);
meansens = zeros(Nimages-1);

% read sensor images
parfor u = 1:Nimages-1
    sensor = strcat(imloc, sprintf('%04d',Nreference+u), imtxt);
    sensorim = strcat(imloc, sprintf('%04d',Nreference+u), imfmt);
    [Rsens(:,:,u), tsens(:,:,u)] = computeRT(sensor);
    Rrel(:,:,u) = Rsens(:,:,u) \ Rref;
    trel(:,:,u) = tsens(:,:,u) - Rrel(:,:,u)*tref;
    IMsens = imread(sensorim);
    sens(:,:,u)=im2double(rgb2gray(IMsens));
    stdsens(u) = std2(sens(:,:,u));
    meansens(u) = mean2(sens(:,:,u));
    nsens(:,:,u) = (sens(:,:,u) - meansens(u)) ./ stdsens(u); %normalized sensor image
end

dstep = (zfar - znear) / nsteps;

depths = znear:dstep:zfar;

depthmap = gpuArray(depthmap);

% calculate depthmaps for each sensor image
parfor v = 1:Nimages-1
    depthmap(:,:,v) = calcDepth(depths, K, Rrel(:,:,v), trel(:,:,v), nref, nsens(:,:,v), Winsize);
end

% average depth results
dpm = gpuArray(zeros(sizeref(1), sizeref(2)));

for v = 1:Nimages-1
    dpm = bsxfun(@plus, dpm, depthmap(:,:,v));
end

dpm = gather(bsxfun(@rdivide, dpm, Nimages-1));

% plot calculated depth as gray image
figure(1);
ds = uint8(dpm .* (255 / zfar));
imagesc(ds,[0 255]); colormap(gray);

%plot calculated depth as 3d surface
figure(2);
h = surf(dpm);
set(h, 'LineStyle', 'none');

% figure(1);
% [x, y, z] = compute3Dpositions(reference, strcat(imloc, sprintf('%04d',Nreference), imdpt));
% f = surf(x, y, z);
% set(f, 'LineStyle', 'none');
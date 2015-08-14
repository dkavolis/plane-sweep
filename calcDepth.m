function [depthmap, bestncc] = calcDepth(depths, K, Rrel, trel, reference, sensor, Winsize, meanref, stdref)
threshold = 0.5;             % std product threshold to determine homogenous regions
n = gpuArray([0 0 1]');

depthmap = gpuArray(ones(size(reference)));
depthmap = bsxfun(@times, depthmap, depths(1));
bestncc = gpuArray(zeros(size(reference)));

[ydim, xdim] = size(reference);

% image x and y pixel coordinates
X1 = gpuArray(repmat(1:xdim,ydim,1));
Y1 = gpuArray(repmat((1:ydim)',1,xdim));

% summation kernel
g = gpuArray(ones(Winsize,1) ./ Winsize);

ns = gpuArray(sensor);
nr = gpuArray(reference);

for d = depths
    H = (K * (Rrel + trel * n' ./ d) / K); % homography
    H = H ./ H(3,3);

    % calculate new pixel positions in sensor image
    x2 = bsxfun(@plus, bsxfun(@plus, bsxfun(@times, H(1,1), X1), bsxfun(@times, H(1,2), Y1)), H(1,3));
    y2 = bsxfun(@plus, bsxfun(@plus, bsxfun(@times, H(2,2), Y1), bsxfun(@times, H(2,1), X1)), H(2,3));
    w  = bsxfun(@plus, bsxfun(@plus, bsxfun(@times, H(3,1), X1), bsxfun(@times, H(3,2), Y1)), H(3,3));
    x2 = bsxfun(@rdivide, x2, w);
    y2 = bsxfun(@rdivide, y2, w);

    % interpolate pixel values in transformed image
    warped = interp2(X1, Y1, ns, x2, y2, 'linear', 0);

    % calculate mean and std for each window in warped image
    meanwarped = colfilter(colfilter(warped,g).',g).';
    sqrwarped = warped .* warped;
    meansqrwarped = colfilter(colfilter(sqrwarped,g).',g).';
    var = meansqrwarped - meanwarped .* meanwarped;
    positive = bsxfun(@gt, var, 0);
    
    %only apply sqrt to positive variances, set negative ones to below
    %threshold so they are ignored
    stdwarped = sqrt(positive .* var) + (1 - positive) * threshold / 10;

    % calculate NCC for each window which is given by
    % NCC = (mean of products - product of means) / product of standard deviations
    I1I2 = warped .* nr;                                    % element-wise product
    mean1mean2 = meanref .* meanwarped;                     % product of means in a window
    sI1I2 = colfilter(colfilter(I1I2,g).',g).';             % mean of products in a window
    stdprod = stdref .* stdwarped;
    ncc = (sI1I2 - mean1mean2) ./ stdprod;                  % NCC
    
    % find non-homogenous regions and use ncc only for them
    abovethreshref = bsxfun(@gt, stdref, threshold);
    abovethreshwarped = bsxfun(@gt, stdwarped, threshold);
    abovethresh = bsxfun(@and, abovethreshref, abovethreshwarped);
    ncc = abovethresh .* ncc;
    
    % find better NCC values and update depthmap and best NCC
    greater = bsxfun(@gt, ncc, bestncc);
    less = bsxfun(@minus, 1, greater);
    depthmap = bsxfun(@plus, bsxfun(@times, greater, d), bsxfun(@times, depthmap, less));
    bestncc = bsxfun(@plus, bsxfun(@times, greater, ncc), bsxfun(@times, less, bestncc));

end
end
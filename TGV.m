function depth = TGV(depthmap, sdepth, img, iters, alpha0, alpha1, tau, sigma, beta, gamma, theta)

[s1, s2] = size(depthmap);
depthmap = gpuArray(depthmap);
sdepth = gpuArray(sdepth);
img = gpuArray(img);

% normalize input images:
maxd = max(max(depthmap));
maxsd = max(max(sdepth));
maximg = max(max(img));
maxd = max(maxd, maxsd);

depthmap = depthmap / maxd;
if maxd > 0 
    sdepth = sdepth / maxd;
end
if maximg > 0
    img = img / maximg;
end

% initialize dual and primal variables
px = gpuArray(zeros(s1,s2));
py = gpuArray(zeros(s1,s2));
qx = gpuArray(zeros(s1,s2));
qy = gpuArray(zeros(s1,s2));
qz = gpuArray(zeros(s1,s2));
qw = gpuArray(zeros(s1,s2));
u = gpuArray(depthmap);
ubar = gpuArray(depthmap);
Ds = gpuArray(sdepth);
vx = gpuArray(zeros(s1,s2));
vy = gpuArray(zeros(s1,s2));
vxbar = gpuArray(zeros(s1,s2));
vybar = gpuArray(zeros(s1,s2));

% forward differences gradient kernel
grad = [1 -1 0]';
grad = gpuArray(grad);

% backward differences divergence kernel
div = [0 1 -1]';
div = gpuArray(div);
w0 = double(Ds > 0);

l = 2;
x = ones(2*l+1,1);
y = (-l:l)';
sigmah = 100;
h = exp(- (y .* y) / 2 / sigmah / sigmah);
[row, col, dm] = find(Ds);
[sd1, sd2] = size(dm);
if sd1 ~= 0
    r = Ds(Ds(1,:)>0);
    [s1, s2] = size(r);
    dm = reshape(dm, sd1 / s2, s2);
    row = reshape(row, sd1 / s2, s2);
    col = reshape(col, sd1 / s2, s2);
    x0 = gpuArray(repmat(1:640,480,1));
    y0 = gpuArray(repmat((1:480)',1,640));
%     Ds = conv2(conv2(Ds', x, 'same')', x, 'same');
    Ds = interp2(col, row, dm, x0, y0, 'linear', 1);
    w = conv2(conv2(w0', h, 'same')', h, 'same');
    w(row(end,end)+1:end, :) = 0;
    w(:, col(end,end)+1:end) = 0;
else
    w = w0;
end
% w = w0;

% calculate anisotropic diffusion tensor
dx = colfilter(img', grad)';
dy = colfilter(img, grad);
d = sqrt(dx .* dx + dy .* dy);
dx = dx ./ d;
dy = dy ./ d;
k = exp(-beta * (d .^ gamma));
T1 = k .* dx .* dx + dy .* dy;
T4 = k .* dy .* dy + dx .* dx;
T2 = (k - 1) .* dx .* dy;

% correct NaNs
T1(isnan(T1)) = 1;
T4(isnan(T4)) = 1;
T2(isnan(T2)) = 0;
T3 = T2;

for i=1:iters
    
    % update P
    dx = colfilter(ubar', grad)';
    dy = colfilter(ubar, grad);
    px = px + alpha1 * sigma * (T1 .* dx + T2 .* dy);
    py = py + alpha1 * sigma * (T3 .* dx + T4 .* dy);
    d = max(1, sqrt(px .* px + py .* py));
    px = px ./ d;
    py = py ./ d;
    
    % update Q
    dxvx = colfilter(vx', grad)';
    dyvx = colfilter(vx, grad);
    dxvy = colfilter(vy', grad)';
    dyvy = colfilter(vy, grad);
    qx = qx + alpha0 * sigma * dxvx;
    qy = qy + alpha0 * sigma * dyvy;
    qz = qz + alpha0 * sigma * (dyvx + dxvy) / 2;
    qw = qw + alpha0 * sigma * (dyvx + dxvy) / 2;
    d = max(1, sqrt(qx .* qx + qy .* qy + qz .* qz + qw .* qw));
    qx = qx ./ d;
    qy = qy ./ d;
    qz = qz ./ d;
    qw = qw ./ d;
    
    % update u and v
    dxpx = colfilter(px', div)';
    dypx = colfilter(px, div);
    dxpy = colfilter(py', div)';
    dypy = colfilter(py, div);
    dxqx = colfilter(qx', div)';
    dyqz = colfilter(qz, div);
    dxqz = colfilter(qz', div)';
    dyqy = colfilter(qy, div);
    u = (u + tau * (alpha1 * (T1 .* dxpx + T2 .* dxpy + T3 .* dypx + T4 .* dypy) + w .* Ds)) ./ (1 + tau * w);
    vx = vx + tau * (alpha1 * (T1 .* px + T2 .* py) + alpha0 * (dxqx + dyqz));
    vy = vy + tau * (alpha1 * (T3 .* px + T4 .* py) + alpha0 * (dxqz + dyqy));
    
    ubar = u + theta * (u - ubar);
    vxbar = vx + theta * (vx - vxbar);
    vybar = vy + theta * (vy - vybar);
    
    sigmah = 0.975 * sigmah;
    h = exp(- (y .* y) / 2 / sigmah / sigmah);
    w = conv2(conv2(w0', h, 'same')', h, 'same');
    w(row(end,end)+1:end, :) = 0;
    w(:, col(end,end)+1:end) = 0;
end

depth = gather(u * maxd);

end

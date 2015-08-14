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

w = double(Ds > 0);
Ds = w .* Ds;

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
end

depth = gather(u * maxd);

end

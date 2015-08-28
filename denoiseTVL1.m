function image = denoiseTVL1(in, img, lambda, niters, zfar, znear)
    L2 = 8;
    tau = 0.02;
    sigma = 1 / L2 / tau;
    theta = 1;
    
    px = gpuArray(zeros(size(in)));
    py = px;
    r = px;
    X = gpuArray(in);
    X(isnan(X)) = zfar;
    X = (X - znear) / (zfar - znear);
    raw = -sigma * X;
    
    % forward differences gradient kernel
    grad = [1 -1 0]';
    grad = gpuArray(grad);
    
    % backward differences divergence kernel
    div = [0 1 -1]';
    div = gpuArray(div);
    
%     img = gpuArray(img / 255);
%     g = exp(- 20 * sqrt(colfilter(img', grad)' .^ 2 + colfilter(img, grad) .^ 2) .^ 0.5);
    
    % calculate anisotropic diffusion tensor
    img = gpuArray(img / 255);
    dx = colfilter(img', grad)';
    dy = colfilter(img, grad);
    d = sqrt(dx .* dx + dy .* dy);
    dx = dx ./ d;
    dy = dy ./ d;
    k = exp(-20 * (d .^ 0.5));
    T1 = k .* dx .* dx + dy .* dy;
    T4 = k .* dy .* dy + dx .* dx;
    T2 = (k - 1) .* dx .* dy;
    
    % correct NaNs
    T1(isnan(T1)) = 1;
    T4(isnan(T4)) = 1;
    T2(isnan(T2)) = 0;
    T3 = T2;
    
    for i=1:niters
       if i == 1 currsigma = sigma + 1;
       else currsigma = sigma;
       end
           
       dx = colfilter(X', grad)';
       dy = colfilter(X, grad);
       
%        px = px + currsigma * dx .* g;
%        py = py + currsigma * dy .* g;
       px = px + currsigma * (T1 .* dx + T2 .* dy);
       py = py + currsigma * (T3 .* dx + T4 .* dy);
       
       l = max(sqrt(px .* px + py .* py), 1);
       px = px ./ l;
       py = py ./ l;
       
       r = r + raw + sigma * X;
       r = min(r, lambda);
       r = max(r, - lambda);
       
       xn = X + tau * (colfilter(px', div)' + colfilter(py, div)) - tau * r;
       
       X = xn + theta * (xn - X);
    end
    
    X = (zfar - znear) * X + znear;
    image = gather(X);
end
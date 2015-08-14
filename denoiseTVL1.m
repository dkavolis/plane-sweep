function img = denoiseTVL1(in, lambda, niters, zfar, znear)
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
    
    for i=1:niters
       if i == 1 currsigma = sigma + 1;
       else currsigma = sigma;
       end
           
       dx = colfilter(X', grad)';
       dy = colfilter(X, grad);
       
       px = px + currsigma * dx;
       py = py + currsigma * dy;
       
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
    img = gather(X);
end
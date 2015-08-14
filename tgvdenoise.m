iters = 1000;
alpha0 = 1;
alpha1 = 0.05;
tau = 0.02;
sigma = 80;
beta = 35;
gamma = 0.8;
theta = 1;

tgv = TGV(dpm, sd, ref, iters, alpha0, alpha1, tau, sigma, beta, gamma, theta);
imagesc(tgv);
colormap(gray);
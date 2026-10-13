% demo_p1

% options
options.maxit   = 100;
options.tol     = 1e-4;
options.display = true;

% functions
f               = @(x) 4*x.^2 - x.^4;
g               = @(x) 8*x - 4*x.^3;

xl              = -1;
xr              = 1/2;

[xa]            = ausection(f,xl,xr,options);
[xb,~]          = bisection(g,xl,xr,options);
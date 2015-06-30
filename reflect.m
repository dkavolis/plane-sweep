function y = reflect(x,minx,maxx)
% function y = reflect(x,minx,maxx)
% Reflect the values in matrix x about the scalar values minx and maxx.
% Hence a vector x containing a long linearly increasing series
% is converted into a waveform which ramps linearly up and down
% between minx and maxx.
% If x contains integers and minx and maxx are (integers + 0.5),
% the ramps will have repeated max and min samples.
%
% Nick Kingsbury, Cambridge University, January 1999.

y = x;

% Reflect y in maxx.
t = find(y > maxx);
y(t) = 2*maxx - y(t);

% Reflect y in minx.
t1 = find(y < minx);
t2 = 0;
while ~isempty(t1) | ~isempty(t2), % Repeat until no more values out of range.
   if ~isempty(t1), y(t1) = 2*minx - y(t1); end
   t2 = find(y > maxx);
   if ~isempty(t2), y(t2) = 2*maxx - y(t2); end
   t1 = find(y < minx);
end
return;


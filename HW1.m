% Initialize A & b
n = 10;
k = 5;
A = zeros(n,n,k);
b = zeros(n,k);
for s = 1:k
    for i = 1:n
        b(i,s) = exp(i/s) * sin(i*s);
        for j = 1:n
            if i ~= j
                A(i,j,s) = exp(min(i/j,j/i)) * cos(i*j) * sin(s);
            end
        end
    end
end
for s = 1:k
    for i =1:n
        A(i,i,s) = i/10 * abs(sin(s)) + sum(abs(A(i,:,s)));
    end
end

% C/sqrt(T) stepsize
T = 1000000;
C = 0.1;
x = ones(n,1);
best = zeros(T,1);
[best(1),sub_g] = f(x,k,A,b);
for t = 2:T
    x = x - C * sqrt(1/t) * sub_g/norm(sub_g);
    [v, sub_g] = f(x,k,A,b);
    best(t) = min(best(t-1),v);
end
gap = best - best(T)+ 1e-6;
loglog(gap)

% Update the optimal value
optimal = best(T);

% Polyak stepsize
x = ones(n,1);
best1 = zeros(T,1);
[v,sub_g] = f(x,k,A,b);
best1(1) = v;
for t = 2:T
    x = x - (v-optimal) * sub_g/norm(sub_g)^2;
    [v, sub_g] = f(x,k,A,b);
    best1(t) = min(best1(t-1),v);
end
gap1 = best1 - optimal;
loglog(gap1)

% Calculate f(x1)
f(ones(n,1),k,A,b)

function [value, subgrad] = f(x,k,A,b)
c = zeros(k,1);
    for i = 1:k
        c(i) = x'*A(:,:,i)*x - x'*b(:,i);
    end
[value, index] = max(c);
subgrad = 2*A(:,:,index)*x - b(:,index);
end

Mf = 1;
max_iter = 30;
epsilon = [1;0.1;0.01;0.005];
record = zeros(4, max_iter);
record_iter = 1:4;

for ind = 1:4
    iter = 1;
    x_ = zeros(10,1);
    x = zeros(10,1);
    [L, x_] = update(x, epsilon(ind), Mf);
    while L>1e-6
        [record(ind, iter), x_] = update(x, epsilon(ind), Mf);
        x=x_;
        L = record(ind, iter);
        iter = iter + 1;
    end
    record_iter(ind) = iter;
end

figure;
subplot(221);
plot(record(1,:));
hold on;
plot(record_iter(1), 0 , '*');
title("\epsilon = 1");
ylabel("\lambda(f,x)");
xlabel("Iteration");

subplot(222);
plot(record(2,:));
hold on;
plot(record_iter(2), 0 , '*');
title("\epsilon = 0.1");
ylabel("\lambda(f,x)");
xlabel("Iteration");

subplot(223);
plot(record(3,:));
hold on;
plot(record_iter(3), 0 , '*');
title("\epsilon = 0.01");
ylabel("\lambda(f,x)");
xlabel("Iteration");

subplot(224);
plot(record(4,:));
hold on;
plot(record_iter(4), 0 , '*');
title("\epsilon = 0.005");
ylabel("\lambda(f,x)");
xlabel("Iteration");


function nabla = Hess(x)
v=zeros(10,1);
for i = 1:10
    v(i) = 1/(1-x(i))^2 + 1/(1+x(i))^2;
end
nabla = diag(v);
end

function g = grad(x,eps)
g=zeros(10,1);
for i = 1:10
    g(i) = i/eps + 1/(1-x(i)) - 1/(1+x(i));
end
end

function [lambda, x_] = update(x,eps,Mf)
g = grad(x,eps);
H = Hess(x);
lambda = sqrt(g'* (H\g));
x_ = x - 1/(1+ Mf * lambda) * (H\g);
end
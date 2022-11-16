%Ryan Reschak, Scott Crowner
clear
clc

syms x1 x2 alpha

%x_0 = starting point
%func = Symbolic expression made of your vars
%vars = a vector of your symbolic variables ex: [x_1, x_2]
%k = max iterations before termination
x_0 = [0.5, 0.5];
x_set = randi([-1000,1000], 5, 2);
vars = [x1, x2];
k = 10;
func = 2*x1*x2 + 3*x2 - x1^2 -2*x2^2;
t = 0.5;
annel = 0.1;

%% Multiple Start Point Steepest Ascent
x_final = x_0;
x_prev = x_0+100;
grad = gradient(func,vars);
i = 0;
tic
for j = 1:length(x_set(:,1))
    x_final = x_set(j,:);
    while abs(norm(x_final-x_prev)) > 0.001
        %find alpha equation
        alpha_eq = x_final' + alpha*subs(grad, vars, x_final);
        alpha_opt = solve(diff(subs(func,vars,alpha_eq'))==0, alpha) ;
        
        x_prev = x_final;
        x = x_final' + alpha_opt*subs(grad, vars, x_final);
        x_final = x';
        i = i + 1;
    end
    if j == 1
        x_last_best = x_final;
    end
    if j~=1 & subs(grad, vars, x_final) < subs(grad, vars, x_last_best)
        x_last_best = x_final;
    end
end
toc

%% Multiple Start Point Newton's Method
x_final = x_0;
x_prev = x_0+100;
grad = gradient(func,vars);
H = hessian(func, vars);
i = 0;
tic
for j = 1:length(x_set(:,1))
    x_final = x_set(j,:);
    while abs(norm(x_final-x_prev)) > 0.001
        %find alpha equation
        alpha_eq = x_final' + alpha*inv(H)*subs(grad, vars, x_final);
        alpha_opt = solve(diff(subs(func,vars,alpha_eq'))==0, alpha);
        
        x_prev = x_final;
        x = x_final' + alpha_opt*inv(H)*subs(grad, vars, x_final);
        x_final = x';
        i = i + 1;
    end
    if j == 1
        x_last_best = x_final;
    end
    if j~=1 & subs(grad, vars, x_final) < subs(grad, vars, x_last_best)
        x_last_best = x_final;
    end
end
toc